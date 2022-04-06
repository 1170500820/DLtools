import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertModel
from itertools import chain
import pickle

from type_def import *
from utils import tools, batch_tool, tokenize_tools
from utils.tokenize_tools import OffsetMapping
from utils.data import SimpleDataset
from evaluate.evaluator import BaseEvaluator, EE_F1Evaluator, SentenceWithEvent, Events, Event, Mention, Mentions
from models.model_utils import get_init_params
from analysis.recorder import NaiveRecorder

from work.EE import EE_settings, EE_utils


class PLMEE_Argument(nn.Module):
    """
    与PLMEE_Trigger的结构相似，
    """
    def __init__(self, plm_lr: float = EE_settings.plm_lr, linear_lr: float = EE_settings.others_lr,
                 plm_path: str = EE_settings.default_plm_path, role_types: str = EE_settings.role_types):
        super(PLMEE_Argument, self).__init__()
        self.init_params = get_init_params(locals())

        self.plm_lr = plm_lr
        self.linear_lr = linear_lr
        self.plm_path = plm_path
        self.role_types = role_types

        self.bert = BertModel.from_pretrained(self.plm_path)
        self.hidden = self.bert.config.hidden_size

        # 分别用于预测论元的开始与结束位置
        self.start_classifiers = nn.ModuleList(nn.Linear(self.hidden, 2) for i in range(len(self.role_types)))
        self.end_classifiers = nn.ModuleList(nn.Linear(self.hidden, 2) for i in range(len(self.role_types)))

        self.init_weights()

    def init_weights(self):
        for elem_cls in self.start_classifiers:
            torch.nn.init.xavier_uniform_(elem_cls.weight)
            elem_cls.bias.data.fill_(0)
        for elem_cls in self.end_classifiers:
            torch.nn.init.xavier_uniform_(elem_cls.weight)
            elem_cls.bias.data.fill_(0)

    def get_optimizers(self):
        linear_start_params = self.start_classifiers.parameters()
        linear_end_params = self.end_classifiers.parameters()
        bert_params = self.bert.parameters()
        linear_optimizer = AdamW(params=chain(linear_start_params, linear_end_params), lr=self.linear_lr)
        bert_optimizer = AdamW(parser=bert_params, lr=self.plm_lr)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor):
        """

        :param input_ids: (bsz, seq_l)
        :param token_type_ids: (bsz, seq_l)
        :param attention_mask: (bsz, seq_l)
        :return:
        """
        output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        embed = output[0]  # (bsz, seq_l, hidden)

        pred_starts = []
        pred_ends = []
        for cls in self.start_classifiers:
            pred_starts.append(cls(embed))  # (bsz, seq_l, 2)
        for cls in self.end_classifiers:
            pred_ends.append(cls(embed))  # (bsz, seq_l, 2)

        return {
            'pred_starts': pred_starts,
            'pred_ends': pred_ends,
            'mask': attention_mask
        }