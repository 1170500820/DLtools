"""
实现一些常用到的loss模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# from allennlp.modules import ConditionalRandomField
#
#
# class CRF_Loss(nn.Module):
#     def __init__(self):
#         super(CRF_Loss, self).__init__()
#
#     def forward(self, crf: ConditionalRandomField, token_scores, tags, token_mask):
#         batch_size = token_scores.shape[0]
#         loss = -crf(token_scores, tags, token_mask) / float(batch_size)
#         return loss


class Scalar_Loss(nn.Module):
    """
    标量loss针对最简单的loss
    """
    def __init__(self):
        super(Scalar_Loss, self).__init__()

    def forward(self, output: torch.Tensor, label: torch.Tensor):
        """

        :param output:
        :param label:
        :return:
        """
        return F.mse_loss(output, label)

