import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertModel, AutoTokenizer, AutoModel
from itertools import chain
import random

from torch.utils.data.distributed import DistributedSampler

from type_def import *
from utils.data import SimpleDataset, RandomPairDataset
from work.NewsCorrelation import newsco_settings
from work.NewsCorrelation.raw_BERT import BERT_for_Sentence_Similarity, Pearson_Loss, convert_parsed_sample_to_model_sample, eval_dataset_factory
from work.NewsCorrelation import newsco_utils
from models.Loss.regular_loss import Scalar_Loss
from evaluate.evaluator import Pearson_Evaluator
from analysis.recorder import NaiveRecorder
from utils import batch_tool, tools, tokenize_tools


class BERT_compare(nn.Module):
    def __init__(self, pretrain_path: str = newsco_settings.pretrain_path, linear_lr: float = newsco_settings.linear_lr, plm_lr: float = newsco_settings.plm_lr):
        """

        :param pretraine_path:
        """
        super(BERT_compare, self).__init__()
        self.plm_path = pretrain_path
        self.linear_lr = linear_lr
        self.plm_lr = plm_lr

        self.bert_embed1 = BERT_for_Sentence_Similarity(self.plm_path, self.linear_lr, self.plm_lr)
        self.bert_embed2 = BERT_for_Sentence_Similarity(self.plm_path, self.linear_lr, self.plm_lr)
        self.hidden = self.bert_embed1.bert.config.hidden_size

    def get_optimizers(self):
        bert1_optimizers = self.bert_embed1.get_optimizers()
        bert2_optimizers = self.bert_embed2.get_optimizers()
        return bert1_optimizers + bert2_optimizers

    def forward(self,
                input_ids1: torch.Tensor,
                attention_mask1: torch.Tensor,
                input_ids2: torch.Tensor = None,
                attention_mask2: torch.Tensor = None):
        """

        :param input_ids1:
        :param attention_mask1:
        :param input_ids2:
        :param attention_mask2:
        :return:
        """
        sim1 = self.bert_embed1(input_ids1, attention_mask1)  #
        if self.training:
            sim2 = self.bert_embed2(input_ids2, attention_mask2)  #

        if self.training:
            return {
                'sim1': sim1['output'],
                'sim2': sim2['output']
            }
        else:
            pred = float(sim1['pred'])  # 默认选第一个的输出
            return {
                "pred": pred
            }


class Compare_Loss(nn.Module):
    """
    统计计算两个新闻对的loss，并且还要考虑相互之间的大小关系
    """
    def __init__(self, reference_value: float = 1, gap_weight: float = 0.3):
        super(Compare_Loss, self).__init__()
        self.pearsonloss = Pearson_Loss(0)
        self.ref = torch.tensor(reference_value, dtype=torch.float)
        self.gap_weight = gap_weight

    def forward(self,
                sim1: torch.Tensor,
                sim2: torch.Tensor,
                label1: torch.Tensor,
                label2: torch.Tensor):
        """

        :param sim1:
        :param sim2:
        :param label1:
        :param label2:
        :return:
        """
        # 分别计算两部分的loss
        sim_loss1 = self.pearsonloss(sim1, label1)
        sim_loss2 = self.pearsonloss(sim2, label2)

        bsz = sim1.size(0)
        total_gap_loss = 0
        for elem_batch in range(bsz):
            bsim1, bsim2, blabel1, blabel2 = sim1[elem_batch], sim2[elem_batch], label1[elem_batch], label2[elem_batch]
            # 如果sim1与sim2的大小关系与label1、label2相同
            if blabel1 == blabel2:
                gap_loss = F.mse_loss(bsim1, bsim2)
            elif (float(bsim1) > float(bsim2) and blabel1 > blabel2) or (float(bsim1) < float(bsim2) and blabel1 < blabel2):
                gap_loss = F.mse_loss(bsim1 - bsim2, self.ref * (blabel1 - blabel2))
            else:  # 否则
                if float(bsim1) > float(bsim2):
                    gap_loss = 2 * F.mse_loss(bsim2 - bsim1, self.ref)
                elif float(bsim1) == float(bsim2):
                    gap_loss = 0
                else:
                    gap_loss = 2 * F.mse_loss(bsim1 - bsim2, self.ref)
            total_gap_loss += gap_loss
        loss = (1 - self.gap_weight) * (sim_loss1 + sim_loss2) + self.gap_weight * total_gap_loss

        return loss


def dataset_factory(
        crawl_dir: str = newsco_settings.crawl_file_dir,
        newspair_file: str = newsco_settings.newspair_file,
        pretrain_path: str = newsco_settings.pretrain_path,
        bsz: int = newsco_settings.bsz,
        local_rank: int = -1):
    print('get local rank ' + str(local_rank))
    if local_rank in [0, -1]:
        print('[dataset_factory]building samples...', end=' ... ')
    samples = newsco_utils.build_news_samples(crawl_dir, newspair_file)
    random.shuffle(samples)
    # language_pair, id1, id2, crawl1, crawl2, Overall, ...

    # 划分训练、评价
    train_samples, val_samples = batch_tool.train_val_split(samples, newsco_settings.train_ratio)
    if local_rank in [0, -1]:
        print('finish')

    if local_rank in [0, -1]:
        print('[dataset_factory]preparing data for training...', end=' ... ')
    train_dataloader = train_dataset_factory(train_samples, pretrain_path, bsz)
    if local_rank in [0, -1]:
        print('finish')

    if local_rank in [0, -1]:
        print('[dataset_factory]preparing data for evaluating...', end=' ... ')
        val_dataloader = eval_dataset_factory(val_samples, pretrain_path)
        print('finish')
    else:  # 非主卡上不需要eval数据
        val_dataloader = None

    return train_dataloader, val_dataloader


def train_dataset_factory(samples: List[Dict[str, Any]], pretrain_path: str = newsco_settings.pretrain_path, bsz: int = newsco_settings.bsz):
    """
    参考raw_BERT.py
    :param samples:
    :param pretrain_path:
    :param bsz:
    :return:
    """
    train_samples = convert_parsed_sample_to_model_sample(samples, pretrain_path)
    train_dataset = RandomPairDataset(train_samples)
    train_dataset_sampler = DistributedSampler(train_dataset, shuffle=True)

    def collate_fn(lst):
        label1 = torch.Tensor(list(x[0]['label'] for x in lst))  # (bsz, 1)
        input_ids1 = torch.tensor(list(x[0]['input_ids'] for x in lst), dtype=torch.int)  # (bsz, max_seq_l)
        attention_mask1 = torch.tensor(list(x[0]['attention_mask'] for x in lst), dtype=torch.int)  # (bsz, max_seq_l)
        label2 = torch.Tensor(list(x[1]['label'] for x in lst))  # (bsz, 1)
        input_ids2 = torch.tensor(list(x[1]['input_ids'] for x in lst), dtype=torch.int)  # (bsz, max_seq_l)
        attention_mask2 = torch.tensor(list(x[1]['attention_mask'] for x in lst), dtype=torch.int)  # (bsz, max_seq_l)

        return {
            'input_ids1': input_ids1,
            'attention_mask1': attention_mask1,
            "input_ids2": input_ids2,
            'attention_mask2': attention_mask2
               }, {
            'label1': label1,
            'label2': label2
        }

    train_dataloader = DataLoader(train_dataset, batch_size=bsz, collate_fn=collate_fn, sampler=train_dataset_sampler)
    return train_dataloader


model_registry = {
    "model": BERT_compare,
    "evaluator": Pearson_Evaluator,
    "loss": Compare_Loss,
    'train_val_data': dataset_factory,
    'recorder': NaiveRecorder
}

