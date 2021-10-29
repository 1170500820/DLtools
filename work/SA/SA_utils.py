from work.SA import SA_settings
import torch
from utils import tools


def SA_tensor2pred(sa_tensor: torch.Tensor, labels=SA_settings.sentiment_label):
    """

    :param sa_tensor: (*, sentiment_cnt)，最后一维是预测向量，其他都不在乎
    :param labels:
    :return:
    """
    _, idx = torch.max(sa_tensor, dim=-1)  # (*)
    idx_lst = idx.tolist()
    idx_lst = tools.convert_list_with_operation(lambda x: labels[x], idx_lst)
    return idx_lst
