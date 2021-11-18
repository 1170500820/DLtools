"""
该文件将flat源码中的部分方法实现、调试、验证，然后转为使用DLtools的形式


"""
from analysis.record_tools import build_record, write_record
# ==    import    ==

# =  torch essentials  =
import torch
import torch.nn as nn
import torch.nn.functional as F

# =  math related  =
import math
import numpy as np


# ===    似乎是用来生成位置编码的  ===

# ==  原实现+注释整理  ==
def get_embedding(max_seq_len, embedding_dim, padding_idx=None, rel_pos_init=0):
    """
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".

    Giere: rel_pos_init决定来位置编码是从0开始还是从-max/2开始。后者可能会用于构建相对位置编码，所以需要负数的编码

    :param rel_pos_init:    如果是0，那么从-max_len到max_len的相对位置编码矩阵就按0-2*max_len来初始化，
                            如果是1，那么就按-max_len,max_len来初始化
    """
    num_embeddings = 2 * max_seq_len + 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    if rel_pos_init == 0:
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    else:
        emb = torch.arange(-max_seq_len,max_seq_len+1, dtype=torch.float).unsqueeze(1)*emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    if embedding_dim % 2 == 1:
        # zero pad
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0
    return np.array(emb)


# == 我的修改  ==
def build_sinusoidal_embeddings(seq_len, embed_dim, pad_idx=None, gen_negative=False):
    """
    生成一个sinusoidal矩阵。
    :param seq_len:
    :param embed_dim:
    :param pad_idx:
    :param gen_negative:
    :return:
    """


# 分别生成长度为100， 200， 300。dim为32， 64， 128/类型为0、1的矩阵

embedes = []
for seql in [100, 200, 300]:
    seq_lst = []
    for dim in [32, 64, 128]:
        dim_lst = []
        for rel in [0, 1]:
            dim_lst.append(get_embedding(seql, dim, rel_pos_init=rel))
        seq_lst.append(dim_lst)
    embedes.append(seq_lst)
name = 'sinusoidal_flat'
dimensions = [
    {'max_seq_len': [100, 200, 300]},
    {'embedding_dim': [32, 64, 128]},
    {'rel_pos_init': [0, 1]},
    "embedding_matrix"
]

record = build_record([embedes], [name], [dimensions])
write_record(record, '.', {'type':'ndarray'}, {'test'})

