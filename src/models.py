import pickle
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn.init as init
import torch
from torch import nn
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns

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
    e_x = torch.exp(x - torch.max(x))
    return e_x / e_x.sum()

class KBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    q = self.get_queries(these_queries)

                    scores = q @ rhs
                    targets = self.score(these_queries)

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks

class CP(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(CP, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.lhs = nn.Embedding(sizes[0], rank, sparse=True)
        self.rel = nn.Embedding(sizes[1], rank, sparse=True)
        self.rhs = nn.Embedding(sizes[2], rank, sparse=True)

        self.lhs.weight.data *= init_size
        self.rel.weight.data *= init_size
        self.rhs.weight.data *= init_size

    def score(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        return torch.sum(lhs * rel * rhs, 1, keepdim=True)

    def forward(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])
        return (lhs * rel) @ self.rhs.weight.t(), (lhs, rel, rhs)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs.weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        return self.lhs(queries[:, 0]).data * self.rel(queries[:, 1]).data

class ComplEx(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(ComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        return (
                       (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
                       (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
               ), (
                   torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                   torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                   torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
               )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
               chunk_begin:chunk_begin + chunk_size
               ].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)

class ComplExMDR(KBCModel):

    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3,
            modality_split=True, fusion_img=True, fusion_dscp=True,
            alpha=1,
            dataset='FB15K-237',
            scale=16,
            img_info='../data/FB15K-237/img_vec.pickle',
            dscp_info='../data/FB15K-237/dscp_vec.pickle',
            ep=None,
            temp=None
    ):
        super(ComplExMDR, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.tau = temp

        self.r_embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])

        self.r_embeddings[0].weight.data *= init_size  # entity
        self.r_embeddings[1].weight.data *= init_size  # relation

        self.aggr_img_vec = torch.zeros(sizes[0], 768)
        self.weights = nn.Parameter(torch.ones(sizes[0], 30))

        self.modality_split = modality_split
        self.fusion_img = fusion_img
        self.fusion_dscp = fusion_dscp

        self.alpha = alpha  # alpha = 1 means image/text emb does not fuse structure modality

        self.temp = scale

        self.ep = ep

        self.img_dimension = 768
        self.dscp_dimension = 768

        self.str_post_mats_img = nn.Parameter(torch.Tensor(2 * rank, self.img_dimension), requires_grad=True)
        self.str_post_mats_dscp = nn.Parameter(torch.Tensor(2 * rank, self.img_dimension), requires_grad=True)
        nn.init.xavier_uniform(self.str_post_mats_img)
        nn.init.xavier_uniform(self.str_post_mats_dscp)
        self.str_transform_img, self.str_transform_dscp = None, None
        self.scale_str = 1e-5

        if self.fusion_img:
            self.img_info = pickle.load(open(img_info, 'rb'))
            self.img_vec = torch.from_numpy(self.img_info).float().cuda()
            self.img_post_mats = nn.Parameter(torch.Tensor(self.img_dimension, 2 * rank), requires_grad=True)
            nn.init.xavier_uniform(self.img_post_mats)
            self.img_rel_embeddings = nn.Embedding(sizes[1], 2 * rank, sparse=True)
            self.img_rel_embeddings.weight.data *= init_size
            self.img_bn = nn.BatchNorm1d(self.img_dimension)
            self.sparse_img = nn.Sequential(
                nn.Linear(in_features=self.img_dimension, out_features=2 * rank, bias=True),
                nn.BatchNorm1d(2 * rank),
                nn.Sigmoid()
            )

        if self.fusion_dscp:
            self.dscp_info = pickle.load(open(dscp_info, 'rb'))
            self.dscp_vec = torch.from_numpy(self.dscp_info).float().cuda()
            self.dscp_post_mats = nn.Parameter(torch.Tensor(self.dscp_dimension, 2 * rank), requires_grad=True)
            nn.init.xavier_uniform(self.dscp_post_mats)
            self.dscp_rel_embeddings = nn.Embedding(sizes[1], 2 * rank, sparse=True)
            self.dscp_rel_embeddings.weight.data *= init_size
            self.sparse_dscp = nn.Sequential(
                nn.Linear(in_features=self.dscp_dimension, out_features=2 * rank, bias=True),
                nn.BatchNorm1d(2 * rank),
                nn.Sigmoid()
            )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)  # Xavier 初始化
            if m.bias is not None:
                init.zeros_(m.bias)  # 偏置初始化为零

    def score(self, x):
        lhs = self.r_embeddings[0](x[:, 0])
        rel = self.r_embeddings[1](x[:, 1])
        rhs = self.r_embeddings[0](x[:, 2])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        score_str = torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )
        return_value = []
        return_value.append(score_str)

        if self.modality_split:
            if self.fusion_img:
                mask_img = self.sparse_img(self.aggr_img_vec + self.scale_str * self.str_transform_img)  # (14541, 4000)
                img_embeddings = (self.aggr_img_vec).mm(self.img_post_mats)  # (14541, 4000)
                str_img_embeddings = img_embeddings * mask_img
                str_img_embeddings = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * str_img_embeddings
                str_img_embeddings /= self.temp

                lhs_i = str_img_embeddings[(x[:, 0])]
                rel_i = self.img_rel_embeddings(x[:, 1])
                rhs_i = str_img_embeddings[(x[:, 2])]

                lhs_i = lhs_i[:, :self.rank], lhs_i[:, self.rank:]
                rel_i = rel_i[:, :self.rank], rel_i[:, self.rank:]
                rhs_i = rhs_i[:, :self.rank], rhs_i[:, self.rank:]
                score_img = torch.sum(
                    (lhs_i[0] * rel_i[0] - lhs_i[1] * rel_i[1]) * rhs_i[0] +
                    (lhs_i[0] * rel_i[1] + lhs_i[1] * rel_i[0]) * rhs_i[1],
                    1, keepdim=True
                )
                return_value.append(score_img)

            if self.fusion_dscp:
                mask_dscp = self.sparse_dscp(self.dscp_vec + self.scale_str * self.str_transform_dscp)
                dscp_embeddings = (self.dscp_vec).mm(self.dscp_post_mats)
                str_dscp_embeddings = dscp_embeddings * mask_dscp
                str_dscp_embeddings = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * str_dscp_embeddings
                str_dscp_embeddings /= self.temp

                lhs_i = str_dscp_embeddings[(x[:, 0])]
                rel_i = self.dscp_rel_embeddings(x[:, 1])
                rhs_i = str_dscp_embeddings[(x[:, 2])]

                lhs_i = lhs_i[:, :self.rank], lhs_i[:, self.rank:]
                rel_i = rel_i[:, :self.rank], rel_i[:, self.rank:]
                rhs_i = rhs_i[:, :self.rank], rhs_i[:, self.rank:]
                score_dscp = torch.sum(
                    (lhs_i[0] * rel_i[0] - lhs_i[1] * rel_i[1]) * rhs_i[0] +
                    (lhs_i[0] * rel_i[1] + lhs_i[1] * rel_i[0]) * rhs_i[1],
                    1, keepdim=True
                )  # ComplEx的score function
                # [batch , 1]
                return_value.append(score_dscp)
        return tuple(return_value)

    def forward(self, x, epoch, tv1_weights, tv2_weights, ts_weights, vs_weights):
        lhs = self.r_embeddings[0](x[:, 0])
        rel = self.r_embeddings[1](x[:, 1])
        rhs = self.r_embeddings[0](x[:, 2])
        lhs_temp = lhs
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        to_score = self.r_embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        score_str = ((lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) + (
                    lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1))
        factors_str = (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2), torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                       torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))
        return_value = []
        return_value.append(score_str)
        return_value.append(factors_str)
        ############################################################################################################################
        str_fea = self.r_embeddings[0].weight.data.detach()
        self.str_transform_img = str_fea.mm(self.str_post_mats_img) # 14541, 768
        self.str_transform_dscp = str_fea.mm(self.str_post_mats_dscp)  # 14541, 768

        normalized_weights = nn.functional.softmax(self.weights, dim=1)
        self.aggr_img_vec = self.img_bn(torch.matmul(normalized_weights.unsqueeze(-1).transpose(1, 2), self.img_vec).squeeze()) # (14541, 768)
        mask_img = self.sparse_img(self.aggr_img_vec + self.scale_str * self.str_transform_img)  # (14541, 4000)
        img_embeddings = (self.aggr_img_vec).mm(self.img_post_mats) # (14541, 4000)
        str_img_embeddings = img_embeddings * mask_img
        str_img_embeddings = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * str_img_embeddings
        str_img_embeddings /= self.temp
        lhs_i = str_img_embeddings[(x[:, 0])]
        rel_i = self.img_rel_embeddings(x[:, 1])
        rhs_i = str_img_embeddings[(x[:, 2])]
        lhs_i_temp = lhs_i
        lhs_i = lhs_i[:, :self.rank], lhs_i[:, self.rank:]
        rel_i = rel_i[:, :self.rank], rel_i[:, self.rank:]
        rhs_i = rhs_i[:, :self.rank], rhs_i[:, self.rank:]
        to_score_i = str_img_embeddings
        to_score_i = to_score_i[:, :self.rank], to_score_i[:, self.rank:]
        score_img = ((lhs_i[0] * rel_i[0] - lhs_i[1] * rel_i[1]) @ to_score_i[0].transpose(0, 1) + (
                    lhs_i[0] * rel_i[1] + lhs_i[1] * rel_i[0]) @ to_score_i[1].transpose(0, 1))
        factors_img = (torch.sqrt(lhs_i[0] ** 2 + lhs_i[1] ** 2), torch.sqrt(rel_i[0] ** 2 + rel_i[1] ** 2),
                       torch.sqrt(rhs_i[0] ** 2 + rhs_i[1] ** 2))
        return_value.append(score_img)
        return_value.append(factors_img)
        ############################################################################################################################
        mask_dscp = self.sparse_dscp(self.dscp_vec + self.scale_str * self.str_transform_dscp)
        dscp_embeddings = (self.dscp_vec).mm(self.dscp_post_mats)
        str_dscp_embeddings = dscp_embeddings * mask_dscp
        str_dscp_embeddings = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * str_dscp_embeddings
        str_dscp_embeddings /= self.temp
        lhs_d = str_dscp_embeddings[(x[:, 0])]
        rel_d = self.dscp_rel_embeddings(x[:, 1])
        rhs_d = str_dscp_embeddings[(x[:, 2])]
        lhs_d_temp = lhs_d
        lhs_d = lhs_d[:, :self.rank], lhs_d[:, self.rank:]
        rel_d = rel_d[:, :self.rank], rel_d[:, self.rank:]
        rhs_d = rhs_d[:, :self.rank], rhs_d[:, self.rank:]
        to_score_d = str_dscp_embeddings
        to_score_d = to_score_d[:, :self.rank], to_score_d[:, self.rank:]
        score_dscp = ((lhs_d[0] * rel_d[0] - lhs_d[1] * rel_d[1]) @ to_score_d[0].transpose(0, 1) + (
                    lhs_d[0] * rel_d[1] + lhs_d[1] * rel_d[0]) @ to_score_d[1].transpose(0, 1))
        factors_dscp = (torch.sqrt(lhs_d[0] ** 2 + lhs_d[1] ** 2), torch.sqrt(rel_d[0] ** 2 + rel_d[1] ** 2),
                        torch.sqrt(rhs_d[0] ** 2 + rhs_d[1] ** 2))
        return_value.append(score_dscp)
        return_value.append(factors_dscp)
        ############################################################################################################################
        lhs_i_contrast = self.aggr_img_vec[x[:, 0]]
        lhs_d_contrast = self.dscp_vec[x[:, 0]]
        if epoch < self.ep:
            align_loss_tv1 = self.noisy_contrastive_loss(lhs_i_contrast, lhs_d_contrast, None, temperature=self.tau, mode='train')
            align_loss_tv2 = self.noisy_contrastive_loss(lhs_i_temp, lhs_d_temp, None, temperature=self.tau, mode='train')
            align_loss_ts = self.noisy_contrastive_loss(lhs_d_temp, lhs_temp, None, temperature=self.tau, mode='train')
            align_loss_vs = self.noisy_contrastive_loss(lhs_i_temp, lhs_temp, None, temperature=self.tau, mode='train')
            con_loss = align_loss_tv1 + align_loss_tv2 + align_loss_ts + align_loss_vs
        else:
            batch_prob_tv1 = tv1_weights[x[:, 0].cpu().numpy()]
            batch_prob_tv2 = tv2_weights[x[:, 0].cpu().numpy()]
            batch_prob_ts = ts_weights[x[:, 0].cpu().numpy()]
            batch_prob_vs = vs_weights[x[:, 0].cpu().numpy()]
            align_loss_tv1 = self.noisy_contrastive_loss(lhs_i_contrast, lhs_d_contrast, batch_prob_tv1, temperature=self.tau, mode='weighting')
            align_loss_tv2 = self.noisy_contrastive_loss(lhs_i_temp, lhs_d_temp, batch_prob_tv2, temperature=self.tau, mode='weighting')
            align_loss_ts = self.noisy_contrastive_loss(lhs_d_temp, lhs_temp, batch_prob_ts, temperature=self.tau, mode='weighting')
            align_loss_vs = self.noisy_contrastive_loss(lhs_i_temp, lhs_temp, batch_prob_vs, temperature=self.tau, mode='weighting')
            align_loss_tv = align_loss_tv2 + align_loss_tv1
            con_loss = (align_loss_tv + align_loss_ts + align_loss_vs) / 3

        kl_loss = torch.mean(mask_img) + torch.mean(mask_dscp)
        return_value.append(con_loss)
        return_value.append(kl_loss)
        return tuple(return_value)

    def compute_tv1_loss(self, index, temp=None):
        img_vec = F.normalize(self.aggr_img_vec, p=2, dim=-1)
        dscp_vec = F.normalize(self.dscp_vec, p=2, dim=-1)
        img_vec_batch = img_vec[index]
        dscp_vec_batch = dscp_vec[index]

        similarity_i2t = img_vec_batch @ dscp_vec.T # (1000, 14541)
        similarity_i2t_norm = (similarity_i2t - similarity_i2t.min(dim=-1, keepdim=True)[0]) / (similarity_i2t.max(dim=-1, keepdim=True)[0] - similarity_i2t.min(dim=-1, keepdim=True)[0])
        similarity_i2t_norm = (similarity_i2t_norm + 1e-10) / (similarity_i2t_norm.diag().view(similarity_i2t_norm.shape[0], -1) + 1e-10)
        similarity_i2t_norm = torch.clamp(similarity_i2t_norm, min=0, max=1)
        samples_i2t = torch.bernoulli(similarity_i2t_norm)

        similarity_t2i = dscp_vec_batch @ img_vec.T # (1000, 14541)
        similarity_t2i_norm = (similarity_t2i - similarity_t2i.min(dim=-1, keepdim=True)[0]) / (similarity_t2i.max(dim=-1, keepdim=True)[0] - similarity_t2i.min(dim=-1, keepdim=True)[0])
        similarity_t2i_norm = (similarity_t2i_norm + 1e-10) / (similarity_t2i_norm.diag().view(similarity_t2i_norm.shape[0], -1) + 1e-10)
        similarity_t2i_norm = torch.clamp(similarity_t2i_norm, min=0, max=1)
        samples_t2i = torch.bernoulli(similarity_t2i_norm)

        scores_i2t = torch.exp(torch.div(similarity_i2t, temp)) # (1000, 14541)
        scores_t2i = torch.exp(torch.div(similarity_t2i, temp)) # (1000, 14541)

        diagonal_i2t = scores_i2t.diag().view(scores_i2t.size(0), 1).t()
        diagonal_t2i = scores_t2i.diag().view(scores_t2i.size(0), 1).t()
        sum_i2t = (samples_i2t * scores_i2t).sum(1)
        sum_t2i = (samples_t2i * scores_t2i).sum(1)
        loss_i2t = -torch.log(torch.div(diagonal_i2t, sum_i2t)).view(-1)  # (bs,)
        loss_t2i = -torch.log(torch.div(diagonal_t2i, sum_t2i)).view(-1)  # (bs,)
        loss = (loss_i2t + loss_t2i) / 2
        loss = loss.view(-1)
        return loss.detach().cpu().numpy()

    def compute_tv2_loss(self, index, temp=None):
        mask_img = self.sparse_img(self.aggr_img_vec + self.scale_str * self.str_transform_img)  # (14541, 4000)
        img_embeddings = (self.aggr_img_vec).mm(self.img_post_mats) # (14541, 4000)
        str_img_embeddings = img_embeddings * mask_img
        str_img_embeddings = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * str_img_embeddings
        str_img_embeddings /= self.temp

        mask_dscp = self.sparse_dscp(self.dscp_vec + self.scale_str * self.str_transform_dscp)
        dscp_embeddings = (self.dscp_vec).mm(self.dscp_post_mats)
        str_dscp_embeddings = dscp_embeddings * mask_dscp
        str_dscp_embeddings = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * str_dscp_embeddings
        str_dscp_embeddings /= self.temp

        img_vec = F.normalize(str_img_embeddings, p=2, dim=-1)
        dscp_vec = F.normalize(str_dscp_embeddings, p=2, dim=-1)
        img_vec_batch = img_vec[index]
        dscp_vec_batch = dscp_vec[index]

        similarity_i2t = img_vec_batch @ dscp_vec.T # (1000, 14541)
        similarity_i2t_norm = (similarity_i2t - similarity_i2t.min(dim=-1, keepdim=True)[0]) / (similarity_i2t.max(dim=-1, keepdim=True)[0] - similarity_i2t.min(dim=-1, keepdim=True)[0])
        similarity_i2t_norm = (similarity_i2t_norm + 1e-10) / (similarity_i2t_norm.diag().view(similarity_i2t_norm.shape[0], -1) + 1e-10)
        similarity_i2t_norm = torch.clamp(similarity_i2t_norm, min=0, max=1)
        samples_i2t = torch.bernoulli(similarity_i2t_norm)

        similarity_t2i = dscp_vec_batch @ img_vec.T # (1000, 14541)
        similarity_t2i_norm = (similarity_t2i - similarity_t2i.min(dim=-1, keepdim=True)[0]) / (similarity_t2i.max(dim=-1, keepdim=True)[0] - similarity_t2i.min(dim=-1, keepdim=True)[0])
        similarity_t2i_norm = (similarity_t2i_norm + 1e-10) / (similarity_t2i_norm.diag().view(similarity_t2i_norm.shape[0], -1) + 1e-10)
        similarity_t2i_norm = torch.clamp(similarity_t2i_norm, min=0, max=1)
        samples_t2i = torch.bernoulli(similarity_t2i_norm)

        scores_i2t = torch.exp(torch.div(similarity_i2t, temp)) # (1000, 14541)
        scores_t2i = torch.exp(torch.div(similarity_t2i, temp)) # (1000, 14541)

        diagonal_i2t = scores_i2t.diag().view(scores_i2t.size(0), 1).t()
        diagonal_t2i = scores_t2i.diag().view(scores_t2i.size(0), 1).t()
        sum_i2t = (samples_i2t * scores_i2t).sum(1)
        sum_t2i = (samples_t2i * scores_t2i).sum(1)
        loss_i2t = -torch.log(torch.div(diagonal_i2t, sum_i2t)).view(-1)  # (bs,)
        loss_t2i = -torch.log(torch.div(diagonal_t2i, sum_t2i)).view(-1)  # (bs,)
        loss = (loss_i2t + loss_t2i) / 2
        loss = loss.view(-1)
        return loss.detach().cpu().numpy()

    def compute_vs_loss(self, index, temp=None):
        mask_img = self.sparse_img(self.aggr_img_vec + self.scale_str * self.str_transform_img)  # (14541, 4000)
        img_embeddings = (self.aggr_img_vec).mm(self.img_post_mats) # (14541, 4000)
        str_img_embeddings = img_embeddings * mask_img
        str_img_embeddings = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * str_img_embeddings
        str_img_embeddings /= self.temp

        img_vec = F.normalize(str_img_embeddings, p=2, dim=-1)
        dscp_vec = F.normalize(self.r_embeddings[0].weight, p=2, dim=-1)
        img_vec_batch = img_vec[index]
        dscp_vec_batch = dscp_vec[index]

        similarity_i2t = img_vec_batch @ dscp_vec.T # (1000, 14541)
        similarity_i2t_norm = (similarity_i2t - similarity_i2t.min(dim=-1, keepdim=True)[0]) / (similarity_i2t.max(dim=-1, keepdim=True)[0] - similarity_i2t.min(dim=-1, keepdim=True)[0])
        similarity_i2t_norm = (similarity_i2t_norm + 1e-10) / (similarity_i2t_norm.diag().view(similarity_i2t_norm.shape[0], -1) + 1e-10)
        similarity_i2t_norm = torch.clamp(similarity_i2t_norm, min=0, max=1)
        samples_i2t = torch.bernoulli(similarity_i2t_norm)

        similarity_t2i = dscp_vec_batch @ img_vec.T # (1000, 14541)
        similarity_t2i_norm = (similarity_t2i - similarity_t2i.min(dim=-1, keepdim=True)[0]) / (similarity_t2i.max(dim=-1, keepdim=True)[0] - similarity_t2i.min(dim=-1, keepdim=True)[0])
        similarity_t2i_norm = (similarity_t2i_norm + 1e-10) / (similarity_t2i_norm.diag().view(similarity_t2i_norm.shape[0], -1) + 1e-10)
        similarity_t2i_norm = torch.clamp(similarity_t2i_norm, min=0, max=1)
        samples_t2i = torch.bernoulli(similarity_t2i_norm)

        scores_i2t = torch.exp(torch.div(similarity_i2t, temp)) # (1000, 14541)
        scores_t2i = torch.exp(torch.div(similarity_t2i, temp)) # (1000, 14541)

        diagonal_i2t = scores_i2t.diag().view(scores_i2t.size(0), 1).t()
        diagonal_t2i = scores_t2i.diag().view(scores_t2i.size(0), 1).t()
        sum_i2t = (samples_i2t * scores_i2t).sum(1)
        sum_t2i = (samples_t2i * scores_t2i).sum(1)
        loss_i2t = -torch.log(torch.div(diagonal_i2t, sum_i2t)).view(-1)  # (bs,)
        loss_t2i = -torch.log(torch.div(diagonal_t2i, sum_t2i)).view(-1)  # (bs,)
        loss = (loss_i2t + loss_t2i) / 2
        loss = loss.view(-1)
        return loss.detach().cpu().numpy()

    def compute_ts_loss(self, index, temp=None):
        mask_dscp = self.sparse_dscp(self.dscp_vec + self.scale_str * self.str_transform_dscp)
        dscp_embeddings = (self.dscp_vec).mm(self.dscp_post_mats)
        str_dscp_embeddings = dscp_embeddings * mask_dscp
        str_dscp_embeddings = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * str_dscp_embeddings
        str_dscp_embeddings /= self.temp

        img_vec = F.normalize(self.r_embeddings[0].weight, p=2, dim=-1)
        dscp_vec = F.normalize(str_dscp_embeddings, p=2, dim=-1)
        img_vec_batch = img_vec[index]
        dscp_vec_batch = dscp_vec[index]

        similarity_i2t = img_vec_batch @ dscp_vec.T # (1000, 14541)
        similarity_i2t_norm = (similarity_i2t - similarity_i2t.min(dim=-1, keepdim=True)[0]) / (similarity_i2t.max(dim=-1, keepdim=True)[0] - similarity_i2t.min(dim=-1, keepdim=True)[0])
        similarity_i2t_norm = (similarity_i2t_norm + 1e-10) / (similarity_i2t_norm.diag().view(similarity_i2t_norm.shape[0], -1) + 1e-10)
        similarity_i2t_norm = torch.clamp(similarity_i2t_norm, min=0, max=1)
        samples_i2t = torch.bernoulli(similarity_i2t_norm)

        similarity_t2i = dscp_vec_batch @ img_vec.T # (1000, 14541)
        similarity_t2i_norm = (similarity_t2i - similarity_t2i.min(dim=-1, keepdim=True)[0]) / (similarity_t2i.max(dim=-1, keepdim=True)[0] - similarity_t2i.min(dim=-1, keepdim=True)[0])
        similarity_t2i_norm = (similarity_t2i_norm + 1e-10) / (similarity_t2i_norm.diag().view(similarity_t2i_norm.shape[0], -1) + 1e-10)
        similarity_t2i_norm = torch.clamp(similarity_t2i_norm, min=0, max=1)
        samples_t2i = torch.bernoulli(similarity_t2i_norm)

        scores_i2t = torch.exp(torch.div(similarity_i2t, temp)) # (1000, 14541)
        scores_t2i = torch.exp(torch.div(similarity_t2i, temp)) # (1000, 14541)

        diagonal_i2t = scores_i2t.diag().view(scores_i2t.size(0), 1).t()
        diagonal_t2i = scores_t2i.diag().view(scores_t2i.size(0), 1).t()
        sum_i2t = (samples_i2t * scores_i2t).sum(1)
        sum_t2i = (samples_t2i * scores_t2i).sum(1)
        loss_i2t = -torch.log(torch.div(diagonal_i2t, sum_i2t)).view(-1)  # (bs,)
        loss_t2i = -torch.log(torch.div(diagonal_t2i, sum_t2i)).view(-1)  # (bs,)
        loss = (loss_i2t + loss_t2i) / 2
        loss = loss.view(-1)
        return loss.detach().cpu().numpy()

    def per_sample_loss(self, batch_size):
        total_loss_tv1 = np.zeros(shape=(self.sizes[0],))
        total_loss_tv2 = np.zeros(shape=(self.sizes[0],))
        total_loss_ts = np.zeros(shape=(self.sizes[0],))
        total_loss_vs = np.zeros(shape=(self.sizes[0],))
        index = torch.arange(self.sizes[0]).cuda()
        with tqdm(total=self.sizes[0], unit='ex', disable=True, ncols=80) as bar:
            bar.set_description(f'Computing Loss for Each Entity...')
            b_begin = 0
            while b_begin < self.sizes[0]:
                batch_index = index[b_begin:b_begin + batch_size]
                loss_tv1 = self.compute_tv1_loss(batch_index, temp=self.tau)
                loss_ts = self.compute_ts_loss(batch_index, temp=self.tau)
                loss_vs = self.compute_vs_loss(batch_index, temp=self.tau)
                loss_tv2 = self.compute_tv2_loss(batch_index, temp=self.tau)
                total_loss_tv1[batch_index.cpu().numpy()] = loss_tv1
                total_loss_tv2[batch_index.cpu().numpy()] = loss_tv2
                total_loss_ts[batch_index.cpu().numpy()] = loss_ts
                total_loss_vs[batch_index.cpu().numpy()] = loss_vs
                b_begin += batch_size
                bar.update(batch_size)
        return total_loss_tv1, total_loss_tv2, total_loss_ts, total_loss_vs

    def noisy_contrastive_loss(self, fea1, fea2, prob, temperature=None, mode='train'):
        fea1 = F.normalize(fea1, p=2, dim=-1)
        fea2 = F.normalize(fea2, p=2, dim=-1)
        scores = fea1 @ fea2.T  # (bs, bs)
        if mode == 'train':
            scores = torch.exp(torch.div(scores, temperature))  # (bs, bs)
            diagonal = scores.diag().view(scores.size(0), 1).t().to(scores.device)
            sum_row = scores.sum(1)
            sum_col = scores.sum(0)
            loss_fea1_retrieval = -torch.log(torch.div(diagonal, sum_row)).view(-1)  # (bs,)
            loss_fea2_retrieval = -torch.log(torch.div(diagonal, sum_col)).view(-1)  # (bs,)
            loss = (loss_fea1_retrieval + loss_fea2_retrieval) / 2
            return loss.mean()
        elif mode == 'weighting':
            similarity_i2t_norm = (scores - scores.min(dim=-1, keepdim=True)[0]) / (
                        scores.max(dim=-1, keepdim=True)[0] - scores.min(dim=-1, keepdim=True)[0])
            similarity_i2t_norm = (similarity_i2t_norm + 1e-10) / (
                        similarity_i2t_norm.diag().view(similarity_i2t_norm.shape[0], -1) + 1e-10)
            similarity_i2t_norm = torch.clamp(similarity_i2t_norm, min=0, max=1)
            samples_i2t = torch.bernoulli(similarity_i2t_norm)

            similarity_t2i_norm = (scores.t() - scores.t().min(dim=-1, keepdim=True)[0]) / (
                        scores.t().max(dim=-1, keepdim=True)[0] - scores.t().min(dim=-1, keepdim=True)[0])
            similarity_t2i_norm = (similarity_t2i_norm + 1e-10) / (
                        similarity_t2i_norm.diag().view(similarity_t2i_norm.shape[0], -1) + 1e-10)
            similarity_t2i_norm = torch.clamp(similarity_t2i_norm, min=0, max=1)
            samples_t2i = torch.bernoulli(similarity_t2i_norm)

            scores_i2t = torch.exp(torch.div(scores, temperature))  # (1000, 14541)
            scores_t2i = torch.exp(torch.div(scores.t(), temperature))  # (1000, 14541)

            diagonal_i2t = scores_i2t.diag().view(scores_i2t.size(0), 1).t()
            diagonal_t2i = scores_t2i.diag().view(scores_t2i.size(0), 1).t()
            sum_i2t = (similarity_i2t_norm * samples_i2t * scores_i2t).sum(1)
            sum_t2i = (similarity_t2i_norm * samples_t2i * scores_t2i).sum(1)

            loss_fea1_retrieval = -torch.log(torch.div(diagonal_i2t, sum_i2t)).view(-1)  # (bs,)
            loss_fea2_retrieval = -torch.log(torch.div(diagonal_t2i, sum_t2i)).view(-1)  # (bs,)
            prob = torch.Tensor(prob).to(scores.device)
            prob = softmax(prob)  # (bs, )
            loss = ((loss_fea1_retrieval * prob).sum(0) + (loss_fea2_retrieval * prob).sum(0)) / 2
            return loss

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        rhs_str = self.r_embeddings[0].weight[
                  chunk_begin:chunk_begin + chunk_size
                  ].transpose(0, 1)
        return_value = []
        return_value.append(rhs_str)
        if self.modality_split:
            if self.fusion_img:
                mask_img = self.sparse_img(self.aggr_img_vec + self.scale_str * self.str_transform_img)  # (14541, 4000)
                img_embeddings = (self.aggr_img_vec).mm(self.img_post_mats)  # (14541, 4000)
                str_img_embeddings = img_embeddings * mask_img
                str_img_embeddings = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * str_img_embeddings
                str_img_embeddings /= self.temp

                rhs_img = str_img_embeddings[
                          chunk_begin:chunk_begin + chunk_size
                          ].transpose(0, 1)
                return_value.append(rhs_img)
            if self.fusion_dscp:
                mask_dscp = self.sparse_dscp(self.dscp_vec + self.scale_str * self.str_transform_dscp)
                dscp_embeddings = (self.dscp_vec).mm(self.dscp_post_mats)
                str_dscp_embeddings = dscp_embeddings * mask_dscp
                str_dscp_embeddings = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * str_dscp_embeddings
                str_dscp_embeddings /= self.temp

                rhs_dscp = str_dscp_embeddings[
                           chunk_begin:chunk_begin + chunk_size
                           ].transpose(0, 1)
                return_value.append(rhs_dscp)

        return tuple(return_value)

    def get_queries(self, queries: torch.Tensor):
        embedding = self.r_embeddings[0].weight
        lhs = embedding[(queries[:, 0])]
        rel = self.r_embeddings[1](queries[:, 1])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        queries_str = torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)
        return_value = []
        return_value.append(queries_str)

        if self.modality_split:
            if self.fusion_img:
                mask_img = self.sparse_img(self.aggr_img_vec + self.scale_str * self.str_transform_img)  # (14541, 4000)
                img_embeddings = (self.aggr_img_vec).mm(self.img_post_mats)  # (14541, 4000)
                str_img_embeddings = img_embeddings * mask_img
                str_img_embeddings = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * str_img_embeddings
                str_img_embeddings /= self.temp

                lhs_i = str_img_embeddings[(queries[:, 0])]
                rel_i = self.img_rel_embeddings(queries[:, 1])

                lhs_i = lhs_i[:, :self.rank], lhs_i[:, self.rank:]
                rel_i = rel_i[:, :self.rank], rel_i[:, self.rank:]

                queries_img = torch.cat([
                    lhs_i[0] * rel_i[0] - lhs_i[1] * rel_i[1],
                    lhs_i[0] * rel_i[1] + lhs_i[1] * rel_i[0]
                ], 1)
                return_value.append(queries_img)

            if self.fusion_dscp:
                mask_dscp = self.sparse_dscp(self.dscp_vec + self.scale_str * self.str_transform_dscp)
                dscp_embeddings = (self.dscp_vec).mm(self.dscp_post_mats)
                str_dscp_embeddings = dscp_embeddings * mask_dscp
                str_dscp_embeddings = (1 - self.alpha) * self.r_embeddings[0].weight + self.alpha * str_dscp_embeddings
                str_dscp_embeddings /= self.temp

                lhs_d = str_dscp_embeddings[(queries[:, 0])]
                rel_d = self.dscp_rel_embeddings(queries[:, 1])

                lhs_d = lhs_d[:, :self.rank], lhs_d[:, self.rank:]
                rel_d = rel_d[:, :self.rank], rel_d[:, self.rank:]

                queries_dscp = torch.cat([
                    lhs_d[0] * rel_d[0] - lhs_d[1] * rel_d[1],
                    lhs_d[0] * rel_d[1] + lhs_d[1] * rel_d[0]
                ], 1)
                return_value.append(queries_dscp)
        return tuple(return_value)

    def get_ranking_score(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1, filt=False
    ):
        if chunk_size < 0:
            chunk_size = self.sizes[2]  # eval时每次candidate都是所有实体 （chuck_size default = -1）
        scores_str_all = torch.zeros(size=(len(queries), self.sizes[2]))
        scores_img_all = torch.zeros(size=(len(queries), self.sizes[2]))
        scores_dscp_all = torch.zeros(size=(len(queries), self.sizes[2]))
        targets_str_all = torch.zeros(size=(len(queries), 1))
        targets_img_all = torch.zeros(size=(len(queries), 1))
        targets_dscp_all = torch.zeros(size=(len(queries), 1))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                pbar = tqdm(total=len(queries), ncols=80)
                while b_begin < len(queries):  # batch iter
                    these_queries = queries[b_begin:b_begin + batch_size]
                    rhs_str, rhs_img, rhs_dscp = self.get_rhs(c_begin, chunk_size)
                    # rhs: rank * ents
                    # q: batch * rank
                    q_str, q_img, q_dscp = self.get_queries(these_queries)
                    scores_str = q_str @ rhs_str  # batch, ents
                    scores_img = q_img @ rhs_img  # 500, 14951
                    scores_dscp = q_dscp @ rhs_dscp
                    targets_str, targets_img, targets_dscp = self.score(these_queries)  # batch,1
                    targets_str_all[b_begin:b_begin + batch_size] += targets_str.cpu()
                    targets_img_all[b_begin:b_begin + batch_size] += targets_img.cpu()
                    targets_dscp_all[b_begin:b_begin + batch_size] += targets_dscp.cpu()
                    if filt:
                        for i, query in enumerate(these_queries):
                            filter_out = filters[(query[0].item(), query[1].item())]
                            filter_out += [queries[b_begin + i, 2].item()]
                            if chunk_size < self.sizes[2]:  # if candidate is not all entity
                                filter_in_chunk = [
                                    int(x - c_begin) for x in filter_out
                                    if c_begin <= x < c_begin + chunk_size
                                ]
                                scores_str[i, torch.LongTensor(filter_in_chunk)] = -1e6
                                if self.fusion_img:
                                    scores_img[i, torch.LongTensor(filter_in_chunk)] = -1e6
                                if self.fusion_dscp:
                                    scores_dscp[i, torch.LongTensor(filter_in_chunk)] = -1e6
                            else:
                                scores_str[i, torch.LongTensor(filter_out)] = -1e6
                                if self.fusion_img:
                                    scores_img[i, torch.LongTensor(filter_out)] = -1e6
                                if self.fusion_dscp:
                                    scores_dscp[i, torch.LongTensor(filter_out)] = -1e6
                            pbar.update(1)
                    else:
                        pbar.update(batch_size)
                    scores_str_all[b_begin:b_begin + batch_size] += scores_str.cpu()
                    scores_img_all[b_begin:b_begin + batch_size] += scores_img.cpu()
                    scores_dscp_all[b_begin:b_begin + batch_size] += scores_dscp.cpu()
                    b_begin += batch_size
                c_begin += chunk_size
                pbar.close()
        return torch.stack([scores_str_all, scores_img_all, scores_dscp_all]), \
               torch.stack([targets_str_all, targets_img_all, targets_dscp_all])  # (M,T,E), (M,T)

    def get_ensemble_score_filtered_ranking(
            self, ensemble_score: torch.Tensor,
            mod_tri_weight: torch.Tensor,
            queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        if chunk_size < 0:
            chunk_size = self.sizes[2]  # eval时每次candidate都是所有实体 （chuck_size default = -1）
        ranks = torch.ones(len(queries))
        ranks_ent = torch.zeros(size=(len(queries), self.sizes[0]))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                pbar = tqdm(total=len(queries), ncols=80)
                while b_begin < len(queries):  # batch iter
                    these_queries = queries[b_begin:b_begin + batch_size]
                    these_scores = ensemble_score[b_begin:b_begin + batch_size].clone().cuda()
                    these_modalities = mod_tri_weight[:, b_begin:b_begin + batch_size, :].clone().cuda()  # 3, batch, 1
                    targets_str, targets_img, targets_dscp = self.score(these_queries)  # batch,1
                    targets_all = torch.stack([targets_str, targets_img, targets_dscp])  # 3, batch, 1
                    targets_ensemble = torch.sum(targets_all * these_modalities, dim=0)
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:  # if candidate is not all entity
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            these_scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            these_scores[i, torch.LongTensor(filter_out)] = -1e6
                        pbar.update(1)
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (these_scores >= targets_ensemble).float(), dim=1
                    ).cpu()
                    b_begin += batch_size
                c_begin += chunk_size
                pbar.close()
        return ranks

    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """

        if chunk_size < 0:
            chunk_size = self.sizes[2]  # eval时每次candidate都是所有实体 （chuck_size default = -1）
        ranks = torch.ones(len(queries))
        ranks_str = torch.ones(len(queries))
        ranks_img = torch.ones(len(queries))
        ranks_dscp = torch.ones(len(queries))
        ranks_fusion = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                pbar = tqdm(total=len(queries), ncols=80)
                while b_begin < len(queries):  # batch iter
                    if not self.modality_split or (
                            self.modality_split and not self.fusion_img and not self.fusion_dscp):
                        these_queries = queries[b_begin:b_begin + batch_size]
                        rhs = self.get_rhs(c_begin, chunk_size)[0]
                        q = self.get_queries(these_queries)[0]
                        scores_str = q @ rhs
                        scores = scores_str  # torch.Size([500, 14951])
                        targets = self.score(these_queries)[0]
                        for i, query in enumerate(these_queries):
                            filter_out = filters[(query[0].item(), query[1].item())]
                            filter_out += [queries[b_begin + i, 2].item()]
                            if chunk_size < self.sizes[2]:  # if candidate is not all entity
                                filter_in_chunk = [
                                    int(x - c_begin) for x in filter_out
                                    if c_begin <= x < c_begin + chunk_size
                                ]
                                scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                            else:
                                scores[i, torch.LongTensor(filter_out)] = -1e6
                            pbar.update(1)
                        ranks[b_begin:b_begin + batch_size] += torch.sum(
                            (scores >= targets).float(), dim=1
                        ).cpu()
                    else:
                        these_queries = queries[b_begin:b_begin + batch_size]
                        if self.fusion_img and self.fusion_dscp:
                            rhs_str, rhs_img, rhs_dscp = self.get_rhs(c_begin, chunk_size)
                            q_str, q_img, q_dscp = self.get_queries(these_queries)
                            scores_str = q_str @ rhs_str  # 500, 14951
                            scores_img = q_img @ rhs_img  # 500, 14951
                            scores_dscp = q_dscp @ rhs_dscp
                            targets_str, targets_img, targets_dscp = self.score(these_queries)  # 500,1
                        elif self.fusion_img:
                            rhs_str, rhs_img = self.get_rhs(c_begin, chunk_size)
                            q_str, q_img = self.get_queries(these_queries)
                            scores_str = q_str @ rhs_str  # 500, 14951
                            scores_img = q_img @ rhs_img  # 500, 14951
                            targets_str, targets_img = self.score(these_queries)  # 500,1
                        elif self.fusion_dscp:
                            rhs_str, rhs_dscp = self.get_rhs(c_begin, chunk_size)
                            q_str, q_dscp = self.get_queries(these_queries)
                            scores_str = q_str @ rhs_str  # 500, 14951
                            scores_dscp = q_dscp @ rhs_dscp  # 500, 14951
                            targets_str, targets_dscp = self.score(these_queries)  # 500,1
                        for i, query in enumerate(these_queries):
                            filter_out = filters[(query[0].item(), query[1].item())]
                            filter_out += [queries[b_begin + i, 2].item()]
                            if chunk_size < self.sizes[2]:  # if candidate is not all entity
                                filter_in_chunk = [
                                    int(x - c_begin) for x in filter_out
                                    if c_begin <= x < c_begin + chunk_size
                                ]
                                scores_str[i, torch.LongTensor(filter_in_chunk)] = -1e6
                                if self.fusion_img:
                                    scores_img[i, torch.LongTensor(filter_in_chunk)] = -1e6
                                if self.fusion_dscp:
                                    scores_dscp[i, torch.LongTensor(filter_in_chunk)] = -1e6
                            else:
                                scores_str[i, torch.LongTensor(filter_out)] = -1e6
                                if self.fusion_img:
                                    scores_img[i, torch.LongTensor(filter_out)] = -1e6
                                if self.fusion_dscp:
                                    scores_dscp[i, torch.LongTensor(filter_out)] = -1e6
                            pbar.update(1)

                        ranks_str[b_begin:b_begin + batch_size] += torch.sum(
                            (scores_str >= targets_str).float(), dim=1
                        ).cpu()
                        if self.fusion_img:
                            ranks_img[b_begin:b_begin + batch_size] += torch.sum(
                                (scores_img >= targets_img).float(), dim=1
                            ).cpu()
                        if self.fusion_dscp:
                            ranks_dscp[b_begin:b_begin + batch_size] += torch.sum(
                                (scores_dscp >= targets_dscp).float(), dim=1
                            ).cpu()
                    b_begin += batch_size

                c_begin += chunk_size
                pbar.close()

        if not self.modality_split or (self.modality_split and not self.fusion_img and not self.fusion_dscp):
            return ranks
        else:
            if self.fusion_img and self.fusion_dscp:
                ranks_fusion = torch.min(ranks_str, torch.min(ranks_img, ranks_dscp))
                print("ranks_str: {:.4f}, ranks_img: {:.4f}, ranks_dscp: {:.4f}".format(
                    sum(ranks_fusion == ranks_str) / ranks.shape[0],
                    sum(ranks_fusion == ranks_img) / ranks.shape[0],
                    sum(ranks_fusion == ranks_dscp) / ranks.shape[0]))
                # return ranks_str, ranks_img, ranks_dscp
            elif self.fusion_img:
                ranks_fusion = torch.min(ranks_str, ranks_img)
            elif self.fusion_dscp:
                ranks_fusion = torch.min(ranks_str, ranks_dscp)
            return ranks_fusion

    def get_meta_score_filtered_ranking(
            self, ensemble_score: torch.Tensor,
            ensemble_target: torch.Tensor,
            queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        if chunk_size < 0:
            chunk_size = self.sizes[2]  # eval时每次candidate都是所有实体 （chuck_size default = -1）
        ranks = torch.ones(len(queries))
        ranks_ent = torch.zeros(size=(len(queries), self.sizes[0]))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                pbar = tqdm(total=len(queries), ncols=80)
                while b_begin < len(queries):  # batch iter
                    these_queries = queries[b_begin:b_begin + batch_size]
                    these_scores = ensemble_score[b_begin:b_begin + batch_size].clone().cuda()
                    targets_ensemble = ensemble_target[b_begin:b_begin + batch_size].clone().cuda()  # batch
                    targets_ensemble = targets_ensemble.unsqueeze(-1)  # batch, 1
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:  # if candidate is not all entity
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            these_scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            these_scores[i, torch.LongTensor(filter_out)] = -1e6
                        pbar.update(1)
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (these_scores >= targets_ensemble).float(), dim=1
                    ).cpu()
                    b_begin += batch_size
                c_begin += chunk_size
                pbar.close()
        return ranks
