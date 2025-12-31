from pathlib import Path
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Multimodal self-supervised pretraining')
parser.add_argument('--dataset', type=str, help='pretraining dataset', choices=['FB15K237', 'WN18RR', 'YAGO15K'])
parser.add_argument('--data1', type=str, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--data2', type=str, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=5000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=2048, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--lr', default=0.1, type=float) # no effect
parser.add_argument('--cos', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--dim_common', type=int, default=512)
parser.add_argument('--test', default=False)

def fix_random_seeds(seed=2024):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
def save_loss(losses, path):
    # 创建一个图形
    plt.figure(figsize=(10, 6))

    epochs = np.arange(1, len(losses) + 1)
    # 绘制损失变化图
    plt.plot(epochs, losses, marker='o', color='b', linestyle='-', label='Training Loss')

    # 添加标题和标签
    plt.title('Loss Curve during Training', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)

    # 显示网格
    plt.grid(True)

    # 添加图例
    plt.legend()

    # 显示图形
    plt.savefig(path, dpi=300)
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = 10000
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  # 重置计数器
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
class MyDataset(Dataset):
    def __init__(self, args):
        self.img_vec = torch.from_numpy(pickle.load(open(args.data1, 'rb'))).float() # (N, 30, 768)
        self.dscp_vec = torch.from_numpy(pickle.load(open(args.data2, 'rb'))).float() # (N, 768)
        self.num = self.img_vec.shape[0]

    def __getitem__(self, idx):
        return self.img_vec[idx], self.dscp_vec[idx] # (bs, 30, 768) - (bs, 768)

    def __len__(self):
        return self.num
class MultiHeadAttention(nn.Module):
    def __init__(self,
                 dims,
                 k_dims=None,
                 v_dims=None,
                 h_dims=None,
                 o_dims=None,
                 heads=4,
                 p=0.1,
                 bias=True):
        super(MultiHeadAttention, self).__init__()

        self._q_dims = dims
        self._k_dims = k_dims or dims
        self._v_dims = v_dims or dims
        self._h_dims = h_dims or dims
        self._o_dims = o_dims or dims
        self._heads = heads
        self._p = p
        self._bias = bias
        self._head_dims = self._h_dims // heads

        self.q = nn.Linear(self._q_dims, self._h_dims, bias=bias)
        self.k = nn.Linear(self._k_dims, self._h_dims, bias=bias)
        self.v = nn.Linear(self._v_dims, self._h_dims, bias=bias)
        self.m = nn.Linear(self._h_dims, self._o_dims, bias=bias)

        self.dropout = nn.Dropout(p=p)

    def forward(self, q, k=None, v=None, mask=None):
        v = v if torch.is_tensor(v) else k if torch.is_tensor(k) else q
        k = k if torch.is_tensor(k) else q

        q = self.q(q).transpose(0, 1).contiguous()
        k = self.k(k).transpose(0, 1).contiguous()
        v = self.v(v).transpose(0, 1).contiguous()

        b = q.size(1) * self._heads

        q = q.view(-1, b, self._head_dims).transpose(0, 1)
        k = k.view(-1, b, self._head_dims).transpose(0, 1)
        v = v.view(-1, b, self._head_dims).transpose(0, 1)

        att = torch.bmm(q, k.transpose(1, 2)) / self._head_dims**0.5

        if mask is not None:
            mask = torch.where(mask > 0, .0, float('-inf'))
            mask = mask.repeat_interleave(self._heads, dim=0)
            att += mask.unsqueeze(1)

        att = att.softmax(-1)

        if self.dropout is not None:
            att = self.dropout(att)

        m = torch.bmm(att, v).transpose(0, 1).contiguous()
        m = m.view(m.size(0), -1, self._h_dims).transpose(0, 1)
        m = self.m(m)
        return m
class DecoupleNet(nn.Module):
    def __init__(self, args):
        super(DecoupleNet, self).__init__()

        self.head = MultiHeadAttention(dims=768, heads=8, p=0.0, bias=False)

        self.img_transform = nn.Sequential(
            nn.Linear(in_features=768, out_features=768, bias=False),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=768, out_features=768, bias=False),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=768, out_features=768, bias=False)
        )

        self.dscp_transform = nn.Sequential(
            nn.Linear(in_features=768, out_features=768, bias=False),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=768, out_features=768, bias=False),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=768, out_features=768, bias=False)
        )

        self.temp = nn.Parameter(torch.ones([]) * np.array(0.07))

    def compute_fea(self, x1, x2, common_dim):
        img = self.head(x2.unsqueeze(1), x1, x1)
        assert img.shape == (x1.shape[0], 1, 768) and x2.unsqueeze(1).shape == (x1.shape[0], 1, 768)
        img, dscp = img.squeeze(1), x2
        img = self.img_transform(img)
        dscp = self.dscp_transform(dscp)

        img_common_emb, dscp_common_dim = img[:, :common_dim], dscp[:, :common_dim]
        img_spe_emb, dscp_spe_emb = img[:, common_dim:], dscp[:, common_dim:]
        return img_common_emb, dscp_common_dim, img_spe_emb, dscp_spe_emb

    def forward(self, x1, x2, common_dim):
        """
        x1: bs, 30, 768
        x2: bs, 768
        """
        img_common_emb, dscp_common_dim, img_spe_emb, dscp_spe_emb = self.compute_fea(x1, x2, common_dim)

        img_common_emb, dscp_common_dim = F.normalize(img_common_emb, p=2, dim=-1), F.normalize(dscp_common_dim, p=2, dim=-1)
        img_spe_emb, dscp_spe_emb = F.normalize(img_spe_emb, p=2, dim=-1), F.normalize(dscp_spe_emb, p=2, dim=-1)

        labels = torch.arange(x1.shape[0]).to(x1.device)
        logits_fea12fea2 = self.temp * img_common_emb @ dscp_common_dim.t()
        logits_fea22fea1 = self.temp * dscp_common_dim @ img_common_emb.t()
        loss_1 = F.cross_entropy(logits_fea12fea2, labels)
        loss_2 = F.cross_entropy(logits_fea22fea1, labels)
        loss_common = (loss_1 + loss_2) / 2
        loss_spe = torch.sum(img_spe_emb * dscp_spe_emb, dim=-1)
        loss_spe = torch.mean(loss_spe ** 2)
        return loss_common + loss_spe, loss_common, loss_spe
def main_worker(args, checkpoint='checkpoint.model', save_loss_plot=True):
    # if not os.path.isdir(args.checkpoint_dir):
    #     os.makedirs(args.checkpoint_dir, exist_ok=True)
    dataset = MyDataset(args=args)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)
    # test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)
    model = DecoupleNet(args=args).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, verbose=True, patience=50, min_lr=1e-2)
    early_stop = EarlyStopping(patience=50)
    if not args.test:
        print('Start training...')
        best_loss = 10000
        total_loss = []
        for epoch in range(0, args.epochs):
            loss_epoch = []
            loss_com_epoch = []
            loss_spe_epoch = []
            for step, (img, dscp) in enumerate(train_loader):
                # assert img.shape == (args.batch_size, 30, 768) and dscp.shape == (args.batch_size, 768)
                optimizer.zero_grad()
                img, dscp = img.to(torch.device("cuda")), dscp.to(torch.device("cuda"))
                loss, loss_common, loss_spe = model(img, dscp, args.dim_common)
                loss_epoch.append(loss.item())
                loss_com_epoch.append(loss_common.item())
                loss_spe_epoch.append(loss_spe.item())
                loss.backward()
                optimizer.step()
            loss_epoch = np.mean(loss_epoch)
            loss_com_epoch = np.mean(loss_com_epoch)
            loss_spe_epoch = np.mean(loss_spe_epoch)
            total_loss.append(loss_epoch)
            # scheduler.step(loss_epoch)
            print(f"epoch: {epoch + 1}, tatal loss: {loss_epoch}, common loss: {loss_com_epoch}, specific loss: {loss_spe_epoch}.")
            if loss_epoch < best_loss:
                torch.save(model.state_dict(), checkpoint)
                best_loss = loss_epoch
            early_stop(loss_epoch)
            if early_stop.early_stop or (epoch + 1) == args.epochs:
                print(f"Early stopping at epoch {epoch + 1}, best loss: {early_stop.best_loss}")
                print("Loading checkpoint for generating final features.")
                checkpoint_ = torch.load(checkpoint)
                model.load_state_dict(checkpoint_)
                img_vec = torch.from_numpy(pickle.load(open(args.data1, 'rb'))).float().cuda()  # (N, 30, 768)
                dscp_vec = torch.from_numpy(pickle.load(open(args.data2, 'rb'))).float().cuda()  # (N, 768)
                img_common_emb, dscp_common_dim, img_spe_emb, dscp_spe_emb = model.compute_fea(img_vec, dscp_vec, common_dim=args.dim_common)
                img_common_emb, dscp_common_dim = img_common_emb.detach().cpu().numpy(), dscp_common_dim.detach().cpu().numpy()
                assert len(img_common_emb.shape) == 2
                with open("./decouple_" + args.data1.split("/")[-1], 'wb') as out1:
                    pickle.dump(img_common_emb, out1)
                with open("./decouple_" + args.data2.split("/")[-1], 'wb') as out2:
                    pickle.dump(dscp_common_dim, out2)
                if save_loss_plot:
                    save_loss(losses=total_loss, path="./loss.png")
                break

    else:
        print("Loading checkpoint for generating final features.")
        checkpoint_ = torch.load(checkpoint)
        model.load_state_dict(checkpoint_)
        img_vec = torch.from_numpy(pickle.load(open(args.data1, 'rb'))).float().cuda()  # (N, 30, 768)
        dscp_vec = torch.from_numpy(pickle.load(open(args.data2, 'rb'))).float().cuda()  # (N, 768)
        img_common_emb, dscp_common_dim, img_spe_emb, dscp_spe_emb = model.compute_fea(img_vec, dscp_vec,
                                                                                       common_dim=args.dim_common)
        img_common_emb, dscp_common_dim = img_common_emb.detach().cpu().numpy(), dscp_common_dim.detach().cpu().numpy()
        assert len(img_common_emb.shape) == 2
        with open("./decouple_" + args.data1.split("/")[-1], 'wb') as out1:
            pickle.dump(img_common_emb, out1)
        with open("./decouple_" + args.data2.split("/")[-1], 'wb') as out2:
            pickle.dump(dscp_common_dim, out2)

def main():
    global args
    args = parser.parse_args()
    fix_random_seeds(args.seed)
    main_worker(args)

if __name__ == '__main__':
    main()