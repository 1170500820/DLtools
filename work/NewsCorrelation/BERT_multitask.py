import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertModel, AutoTokenizer, AutoModel
from itertools import chain
import random
from torch.utils.data.distributed import DistributedSampler
from itertools import permutations, combinations

from type_def import *
from utils.data import SimpleDataset
from work.NewsCorrelation import newsco_settings
from work.NewsCorrelation import newsco_utils
from models.Loss.regular_loss import Scalar_Loss
from evaluate.evaluator import Pearson_Evaluator
from analysis.recorder import NaiveRecorder
from utils import batch_tool, tools, tokenize_tools


class BERT_for_Sentence_Similarity(nn.Module):
    def __init__(self, pretrain_path: str = newsco_settings.pretrain_path, linear_lr: float = newsco_settings.linear_lr, plm_lr: float = newsco_settings.plm_lr):
        """

        :param pretraine_path:
        """
        super(BERT_for_Sentence_Similarity, self).__init__()
        self.linear_lr = linear_lr
        self.plm_lr = plm_lr
        self.bert = AutoModel.from_pretrained(pretrain_path)
        self.hidden = self.bert.config.hidden_size

        self.layer_size = 200
        self.linear1 = nn.Linear(self.hidden, self.layer_size)
        self.relu = nn.LeakyReLU(inplace=True)
        self.linear2 = nn.Linear(self.layer_size, 1)

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        self.linear1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        self.linear2.bias.data.fill_(0)

    def get_optimizers(self):
        bert_params = self.bert.parameters()
        linear1_params = self.linear1.parameters()
        linear2_params = self.linear2.parameters()
        bert_optimizer = AdamW(params=bert_params, lr=self.plm_lr)
        linear_optimizer = AdamW(params=chain(linear1_params, linear2_params), lr=self.linear_lr)
        return [bert_optimizer, linear_optimizer]

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor):
        """

        :param input_ids: (bsz, seq_l)
        :param attention_mask: (bsz, seq_l)
        :return:
        """
        bert_output = self.bert(input_ids, attention_mask)
        hidden_output = bert_output[0]  # (bsz, seq_l, hidden_size)
        hidden_mean = torch.mean(hidden_output, dim=1)  # (bsz, hidden_size)
        linear_output = self.linear1(hidden_mean)  # (bsz, layer)
        relu_output = self.relu(linear_output)  # (bsz, layer)
        relu_output = self.linear2(relu_output)  # (bsz, 1)
        if self.training:
            return {
                "output": relu_output
            }
        else:
            relu_output = float(relu_output)
            return {
                "pred": relu_output
            }