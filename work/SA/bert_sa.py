# base
from type_def import *
from utils import tools

# mission related
from work.SA import SA_settings
from work.SA import SA_utils

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import numpy as np


# optimize
from torch.optim import AdamW

# evaluate
from evaluate.evaluator import BaseEvaluator, KappaEvaluator

# data
from itertools import chain
from torch.utils.data import DataLoader
from utils.data import SimpleDataset

# analysis
from analysis.recorder import AutoRecorder


# 这次我想试试能不能写一个通用模型框架
class BertForSA(nn.Module):
    def __init__(self, 
                 plm_path=SA_settings.default_plm, 
                 sentiment_cnt=len(SA_settings.sentiment_label), 
                 plm_lr=SA_settings.default_plm_lr, 
                 others_lr=SA_settings.default_others_lr):
        super(BertForSA, self).__init__()
        self.plm_path = plm_path
        self.bert = BertModel.from_pretrained(plm_path)
        self.hidden = self.bert.config.hidden_size
        self.sentiment_cnt = sentiment_cnt
        self.plm_lr = plm_lr
        self.others_lr = others_lr

        # 识别情感类别
        self.classifier = nn.Linear(self.hidden, sentiment_cnt)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0)

    def forward(self,
                input_ids: torch.Tensor,
                token_type_ids: torch.Tensor,
                attention_mask: torch.Tensor):
        result = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        output, pooled = result[0], result[1]
        # pooled (bsz, hidden), output (bsz, seq_l, hidden)

        sentiment_result = self.classifier(pooled)  # (bsz, sentiment_cnt))

        return {
            "sentiment_output": sentiment_result  # (bsz, sentiment_cnt)
        }

    def get_optimizers(self):
        plm_params = self.bert.parameters()
        senti_params = self.classifier.parameters()

        plm_optimizer = AdamW(params=plm_params, lr=self.plm_lr)
        others_optimizer = AdamW(params=senti_params, lr=self.others_lr)
        return [plm_optimizer, others_optimizer]


class SA_cross_entropy_Loss(nn.Module):
    def __init__(self):
        super(SA_cross_entropy_Loss, self).__init__()
        self.weight = 5  # 加在非2项的weight

    def forward(self,
                sentiment_output: torch.Tensor,
                sentiment_target: torch.Tensor):
        """

        :param sentiment_output: (bsz, sentiment_cnt) dtype=torch.float
        :param sentiment_target: (bsz) dtype=torch.long
        :return:
        """
        ce_loss = F.cross_entropy(sentiment_output, sentiment_target, reduction='none')
        weight_tensor = tools.convert_list_with_operation(lambda x: {0: self.weight, 1: self.weight, 2: 1}[x], sentiment_target.clone().detach().tolist())
        weight_tensor = torch.tensor(weight_tensor).cuda()
        weighted_loss = torch.mean(ce_loss * weight_tensor)
        return weighted_loss


class SA_evaluator_combined(BaseEvaluator):
    """
    将会集成多个evaluator
    """
    def __init__(self):
        super(SA_evaluator_combined, self).__init__()
        self.kappa_evaluator = KappaEvaluator()

    def eval_single(self, sentiment_output: torch.tensor, sentiment_gt: int):
        """

        :param sentiment_output: (bsz, sentiment_cnt)
        :param sentiment_gt:
        :return:
        """
        sentiment_pred = SA_utils.SA_tensor2pred(sentiment_output)
        sentiment_pred = sentiment_pred[0]  # int。 默认一次只输入一个sample
        self.kappa_evaluator.eval_single(sentiment_pred, sentiment_gt)

    def eval_step(self) -> Dict[str, Any]:
        score = self.kappa_evaluator.eval_step()
        return score


def train_dataset_factory(data_dicts: List[Dict[str, Any]], bsz, shuffle=True):
    # data_dicts content
    # [text, BIO_anno, class]

    # tokenize
    data_dict = tools.transpose_list_of_dict(data_dicts)
    lst_tokenizer = tools.bert_tokenizer()
    text_result = lst_tokenizer(data_dict['text'])
    text_result = tools.transpose_list_of_dict(text_result)
    data_dict.update(text_result)
    data_dicts = tools.transpose_dict_of_list(data_dict)
    # [text, input_ids, token_type_ids, attention_mask, offset_mappings, BIO_anno, class]

    train_dataset = SimpleDataset(data_dicts)

    def collate_fn(lst):
        dict_of_data = tools.transpose_list_of_dict(lst)
        input_ids = torch.tensor(tools.batchify_ndarray([np.array(x) for x in dict_of_data['input_ids']]))
        token_type_ids = torch.tensor(tools.batchify_ndarray([np.array(x) for x in dict_of_data['token_type_ids']]))
        attention_mask = torch.tensor(tools.batchify_ndarray([np.array(x) for x in dict_of_data['attention_mask']]))
        sentiment_target = torch.tensor(np.array(dict_of_data['class'], dtype=np.long))  # (bsz)
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
               }, {
            'sentiment_target': sentiment_target
        }

    train_dataloader = DataLoader(train_dataset, batch_size=bsz, shuffle=shuffle, collate_fn=collate_fn)
    return train_dataloader


def val_dataset_factory(data_dicts: List[Dict[str, Any]]):
    # data_dicts content
    # [text, BIO_anno, class]

    # tokenize
    data_dict = tools.transpose_list_of_dict(data_dicts)
    lst_tokenizer = tools.bert_tokenizer()
    text_result = lst_tokenizer(data_dict['text'])
    text_result = tools.transpose_list_of_dict(text_result)
    data_dict.update(text_result)
    data_dicts = tools.transpose_dict_of_list(data_dict)
    # [text, input_ids, token_type_ids, attention_mask, offset_mappings, BIO_anno, class]

    val_dataset = SimpleDataset(data_dicts)

    def collate_fn(lst):
        dict_of_data = tools.transpose_list_of_dict(lst)
        input_ids = torch.tensor(tools.batchify_ndarray([np.array(x) for x in dict_of_data['input_ids']]))
        token_type_ids = torch.tensor(tools.batchify_ndarray([np.array(x) for x in dict_of_data['token_type_ids']]))
        attention_mask = torch.tensor(tools.batchify_ndarray([np.array(x) for x in dict_of_data['attention_mask']]))
        sentiment_gt = dict_of_data['class'][0]
        return {
            'input_ids': input_ids,
            "token_type_ids": token_type_ids,
            'attention_mask': attention_mask
               }, {
            'sentiment_gt': sentiment_gt
        }

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return val_dataloader


# * 虽然bert_sa旨在适用所有的sentiment analysis任务，但是数据处理
# * 肯定是没法通用的，只能来一个搭一个
# * 可以专门写一批情感分析的处理代码，然后点菜式调用
def dataset_factory(file_dir: str, bsz: int = SA_settings.default_bsz, shuffle=True):
    trainfile_name = 'train_data_public.csv'
    if file_dir[-1] != '/':
        file_dir += '/'
    filename = file_dir + trainfile_name
    data_dicts = tools.read_csv_as_datadict(filename)
    train_datadicts, val_datadicts = tools.split_list_with_ratio(data_dicts, 0.9)

    train_dataloader = train_dataset_factory(train_datadicts, bsz, shuffle)
    val_dataloader = val_dataset_factory(val_datadicts)

    return train_dataloader, val_dataloader


model_registry = {
    "model": BertForSA,
    "evaluator": SA_evaluator_combined,
    "loss": SA_cross_entropy_Loss,
    "dataset": dataset_factory,
    'args': [
        {'name': "--file_dir", 'dest': 'file_dir', 'type': str, 'help': '训练/测试数据文件的路径'},
    ]
}
