import torch
import torch.nn as nn
import torch.nn.functional as F


import settings
from evaluate.evaluator import *


class StepWeight(nn.Module):
    """
    对于一个(*, seq_l)的表示，在seq_l维度上分配一个加权
    """
    def __init__(self, seq_l: int):
        super(StepWeight, self).__init__()
        self.seq_l = seq_l
        self.step_weight = nn.Parameter(torch.full([self.seq_l], 0.6, dtype=torch.float32), requires_grad=True)

    def forward(self, repr: torch.Tensor):
        """

        :param repr: (*, seq_l) (3,4,5,6,seq_l)
        :return:
        """
        weighted_repr = repr * self.step_weight  # (*, seq_l)
        return weighted_repr


class GeoWeight(nn.Module):
    """
    对于一个 (bsz, n_seq, seq_l, hidden)的典型的cascade batch的表示
    在n_seq维度上以mini-batch为步长，按照几何分布进行加权
    """
    def __init__(self, mini_batch: int):
        super(GeoWeight, self).__init__()
        self.mini_batch = mini_batch
        self.geo_weight = nn.Parameter(torch.full([1], 0.6), requires_grad=True)

    def forward(self, repr: torch.Tensor, cascade_sizes: torch.Tensor):
        """

        :param repr: (bsz, n_seq, seq_l, hidden)
        :param cascade_sizes: (bsz)
        :return:
        """
        # 首先读取出几个维度的大小
        bsz, n_seq, seq_l, hidden = repr.shape
        # (7 // 2) + 1 = 4
        mini_seq = (n_seq // self.mini_batch) + (1 if n_seq % self.mini_batch != 0 else 0)

        # 开始对初始的geo_weight进行处理。
        #   1, 转化为一个概率值, w -> p， 然后按cascade sz进行正则化
        p_geo = F.sigmoid(self.geo_weight)  # (1)
        a_c = torch.pow(p_geo, torch.log2(cascade_sizes + 1)).unsqueeze(dim=-1)  # broadcast (bsz, 1)
        #   2, 按mini-batch展开
        geo_exponent = torch.arange(mini_seq).cuda()  # (mini_seq): (0, 1, 2, ..., mini_seq)
        geo_result = torch.pow(1 - a_c, geo_exponent)  # broadcast, (bsz, mini-seq)
        #   3, 按照repr的维度展开
        #   [[1,2,3], [1,2,3]] -> [[1,1,2,2,3,3], [1,1,2,2,3,3]] (mini-batch=2)
        geo_result = torch.stack([geo_result] * self.mini_batch).permute([1, 2, 0]).reshape(bsz, -1)  # (bsz, mini-batch * mini-seq)
        #   4, 将repr的维度与geo_result对齐。
        #   因为要对n_seq维度加权，所以将n_seq维度放在最后
        #   bsz维度放在倒数第二，因为geo_result按照每个cascade的size不同，值也不同
        repr = repr.permute([2, 3, 0, 1])  # (seq_l, hidden, bsz, n_seq)
        geo_result = geo_result[:, :n_seq]  # 去除多余维度
        repr = repr * geo_result  # (seq_l. hidden, bsz, n_seq)

        repr = repr.permute([2, 3, 0, 1])  # (bsz, n_seq, seq_l, hidden)
        return repr
