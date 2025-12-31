import torch
import tqdm
from torch import nn
from torch import optim
from models import KBCModel
from regularizers import Regularizer
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def save_to_txt(filename, data, mode='a'):
    with open(filename, mode) as f:
        np.savetxt(f, data, fmt='%f', delimiter='\t')

def plot_gmm(gmm, X, clean_index, save_path=''):
    plt.clf()
    ax = plt.gca()
    font1 = {'family': 'DejaVu Sans',
             'weight': 'normal',
             'size': 8,
             }
    x = np.linspace(np.min(X), np.max(X), 1000).reshape(-1, 1)
    log_prob = gmm.score_samples(x)
    pdf = np.exp(log_prob)  # 对数似然转换为概率密度函数
    ax.plot(x, pdf, '-k', label='Mixture PDF')

    loss_values = X.reshape(-1, 1)
    threshold = 0.01
    loss_pdf = gmm.score_samples(loss_values)
    loss_density = np.exp(loss_pdf)
    anomalies = loss_values[(loss_density < threshold) & (loss_values.reshape(-1) > np.max(gmm.means_.flatten()))]
    plt.scatter(anomalies, np.zeros_like(anomalies), color='black', marker='x', s=100, label='anomalies')
    ax.hist(X[clean_index == 1], bins=100, color='blue', alpha=0.4, label='Hard Samples', edgecolor='black')
    ax.hist(X[clean_index == 0], bins=100, color='red', alpha=0.4, label='Easy Samples', edgecolor='black')

    responsibilities = gmm.predict_proba(x)
    pdf_individual = responsibilities * pdf[:, np.newaxis]
    ax.plot(x, pdf_individual[:, 0], '--', label='Component A', color='pink')
    ax.plot(x, pdf_individual[:, 1], '--', label='Component B', color='orange')
    # ax.set_xlabel('Per-sample loss, epoch {}'.format(epoch), font1)
    ax.set_xlabel('Per-sample loss', font1)
    ax.set_ylabel('Density', font1)
    # x_ticks = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # plt.xticks(x_ticks)
    plt.tick_params(labelsize=11)
    ax.legend(loc='upper right', prop=font1)
    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

def split_prob(prob, threshld, mode='a'):
    pred = (prob > threshld)
    return pred # bool

def softmax(x):
    # 为了避免数值溢出，减去数组的最大值
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class KBCOptimizer(object):
    def __init__(
            self, model: KBCModel, regularizer: Regularizer, optimizer: optim.Optimizer, batch_size: int = 256,
            modality_split=True, fusion_img=True, fusion_label=True, fusion_dscp=True,
            verbose: bool = True, size=None, ep=None
    ):
        self.model = model
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.modality_split = modality_split
        self.fusion_img = fusion_img
        self.fusion_label = fusion_label
        self.fusion_dscp = fusion_dscp
        self.tv1_weights = np.ones(shape=(size,)) / size
        self.tv2_weights = np.ones(shape=(size,)) / size
        self.ts_weights = np.ones(shape=(size,)) / size
        self.vs_weights = np.ones(shape=(size,)) / size
        self.size = size
        self.ep = ep

    def get_loss(self, batch):
        total_loss_tv1, total_loss_tv2, total_loss_ts, total_loss_vs = self.model.per_sample_loss(batch)
        return total_loss_tv1, total_loss_tv2, total_loss_ts, total_loss_vs

    def epoch(self, examples: torch.LongTensor, epoch, save_csv=False, is_plot_gmm=False):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]  # 随机数
        if epoch >= self.ep:
            total_loss_tv1, total_loss_tv2, total_loss_ts, total_loss_vs = self.get_loss(self.batch_size) # (N, )

            print("\nFitting GMM, wait a moment...")
            loss_tv1 = total_loss_tv1.reshape(-1, 1)
            loss_tv2 = total_loss_tv2.reshape(-1, 1)
            loss_ts = total_loss_ts.reshape(-1, 1)
            loss_vs = total_loss_vs.reshape(-1, 1)

            gmm_tv1 = GaussianMixture(n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6)
            gmm_tv2 = GaussianMixture(n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6)
            gmm_ts = GaussianMixture(n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6)
            gmm_vs = GaussianMixture(n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6)

            gmm_tv1.fit(loss_tv1)
            gmm_tv2.fit(loss_tv2)
            gmm_ts.fit(loss_ts)
            gmm_vs.fit(loss_vs)

            print('Fitting GMM, Done.\n')

            prob_tv1 = gmm_tv1.predict_proba(loss_tv1)
            prob_tv2 = gmm_tv2.predict_proba(loss_tv2)
            prob_ts = gmm_ts.predict_proba(loss_ts)
            prob_vs = gmm_vs.predict_proba(loss_vs)

            prob_tv1 = prob_tv1[:, gmm_tv1.means_.argmax()] # (bs, ) probability of hard sample
            prob_tv2 = prob_tv2[:, gmm_tv2.means_.argmax()] # (bs, ) probability of hard sample
            prob_ts = prob_ts[:, gmm_ts.means_.argmax()] # (bs, ) probability of hard sample
            prob_vs = prob_vs[:, gmm_vs.means_.argmax()] # (bs, ) probability of hard sample

            if is_plot_gmm:
                index1 = split_prob(prob_tv1, threshld=0.5)
                index2 = split_prob(prob_tv2, threshld=0.5)
                index3 = split_prob(prob_ts, threshld=0.5)
                index4 = split_prob(prob_vs, threshld=0.5)
                if epoch % 5 == 0:
                    plot_gmm(gmm_tv1, loss_tv1.reshape(-1), clean_index=index1, save_path=f"./result/tv1_{epoch}.jpg")
                    plot_gmm(gmm_tv2, loss_tv2.reshape(-1), clean_index=index2, save_path=f"./result/tv2_{epoch}.jpg")
                    plot_gmm(gmm_ts, loss_ts.reshape(-1), clean_index=index3, save_path=f"./result/ts_{epoch}.jpg")
                    plot_gmm(gmm_vs, loss_vs.reshape(-1), clean_index=index4, save_path=f"./result/vs_{epoch}.jpg")

            hard_index_tv1 = split_prob(prob_tv1, threshld=0.5)
            hard_index_tv2 = split_prob(prob_tv2, threshld=0.5)
            hard_index_ts = split_prob(prob_ts, threshld=0.5)
            hard_index_vs = split_prob(prob_vs, threshld=0.5)

            min_error = 1e-10
            error_tv1 = max(np.sum(self.tv1_weights[hard_index_tv1]), min_error)
            alpha_tv1 = 0.5 * np.log((1 - error_tv1) / error_tv1)
            error_tv2 = max(np.sum(self.tv2_weights[hard_index_tv2]), min_error)
            alpha_tv2 = 0.5 * np.log((1 - error_tv2) / error_tv2)
            error_ts = max(np.sum(self.ts_weights[hard_index_ts]), min_error)
            alpha_ts = 0.5 * np.log((1 - error_ts) / error_ts)
            error_vs = max(np.sum(self.vs_weights[hard_index_vs]), min_error)
            alpha_vs = 0.5 * np.log((1 - error_vs) / error_vs)

            self.tv1_weights[hard_index_tv1] *= (np.exp(alpha_tv1) * prob_tv1[hard_index_tv1])
            self.tv1_weights[~hard_index_tv1] *= (np.exp(-alpha_tv1) * (1.0 - prob_tv1[~hard_index_tv1]))
            self.tv2_weights[hard_index_tv2] *= (np.exp(alpha_tv2) * prob_tv2[hard_index_tv2])
            self.tv2_weights[~hard_index_tv2] *= (np.exp(-alpha_tv2) * (1.0 - prob_tv2[~hard_index_tv2]))
            self.ts_weights[hard_index_ts] *= (np.exp(alpha_ts) * prob_ts[hard_index_ts])
            self.ts_weights[~hard_index_ts] *= (np.exp(-alpha_ts) * (1.0 - prob_ts[~hard_index_ts]))
            self.vs_weights[hard_index_vs] *= (np.exp(alpha_vs) * prob_vs[hard_index_vs])
            self.vs_weights[~hard_index_vs] *= (np.exp(-alpha_vs) * (1.0 - prob_vs[~hard_index_vs]))
            self.tv1_weights = softmax(self.tv1_weights)
            self.tv2_weights = softmax(self.tv2_weights)
            self.ts_weights = softmax(self.ts_weights)
            self.vs_weights = softmax(self.vs_weights)

        # examples shape: torch.Size([966284, 3])
        # actual_examples 是examples的乱序版
        loss = nn.CrossEntropyLoss(reduction='mean')
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose, ncols=80) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[b_begin:b_begin + self.batch_size].cuda()  # [batch, 3]
                truth = input_batch[:, 2]
                # truth shape: 1000
                if self.modality_split:
                    if self.fusion_img and self.fusion_dscp:
                        preds_str, fac_str, \
                        preds_img, fac_img, \
                        preds_dscp, fac_dscp, \
                        con_loss, l1_loss = self.model.forward(input_batch, epoch, self.tv1_weights, self.tv2_weights, self.ts_weights, self.vs_weights)
                        l_str_fit = loss(preds_str, truth)
                        l_str_reg = self.regularizer.forward(fac_str)
                        l_img_fit = loss(preds_img, truth)
                        l_img_reg = self.regularizer.forward(fac_img)
                        l_dscp_fit = loss(preds_dscp, truth)
                        l_dscp_reg = self.regularizer.forward(fac_dscp)
                        l = l_str_fit + l_str_reg + l_img_fit + l_img_reg + l_dscp_fit + l_dscp_reg \
                            + 1 * con_loss + 0.1 * l1_loss
                    elif self.fusion_img:
                        preds_str, fac_str, preds_img, fac_img = self.model.forward(input_batch)
                        l_str_fit = loss(preds_str, truth)
                        l_str_reg = self.regularizer.forward(fac_str)
                        l_img_fit = loss(preds_img, truth)
                        l_img_reg = self.regularizer.forward(fac_img)
                        l = l_str_fit + l_str_reg + l_img_fit + l_img_reg
                    elif self.fusion_dscp:
                        preds_str, fac_str, preds_dscp, fac_dscp = self.model.forward(input_batch)
                        l_str_fit = loss(preds_str, truth)
                        l_str_reg = self.regularizer.forward(fac_str)
                        l_dscp_fit = loss(preds_dscp, truth)
                        l_dscp_reg = self.regularizer.forward(fac_dscp)
                        l = l_str_fit + l_str_reg + l_dscp_fit + l_dscp_reg
                    else:
                        preds_str, fac_str = self.model.forward(input_batch)
                        l_str_fit = loss(preds_str, truth)
                        l_str_reg = self.regularizer.forward(fac_str)
                        l = l_str_fit + l_str_reg
                else:
                    preds_str, fac_str = self.model.forward(input_batch)
                    l_str_fit = loss(preds_str, truth)
                    l_str_reg = self.regularizer.forward(fac_str)
                    l = l_str_fit + l_str_reg

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.0f}')
        return l