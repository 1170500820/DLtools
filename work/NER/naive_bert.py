"""
用BERT实现一个最简版本的NER
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from torch.optim import AdamW
from transformers import BertModel, BertTokenizer
from work.NER import CCF_settings
from itertools import chain
from work.NER import NER_utils
from evaluate.evaluator import BaseEvaluator, F1_Evaluator, KappaEvaluator
from type_def import *
from utils import tools
from analysis.recorder import AutoRecorder
from utils.data import SimpleDataset


class NaiveBertHybrid(nn.Module):
    def __init__(self,
                 plm_path=CCF_settings.default_plm,
                 sentiment_cnt=CCF_settings.sentiment_label_cnt,
                 ner_cnt=len(CCF_settings.ner_tags),
                 plm_lr=CCF_settings.plm_lr,
                 others_lr=CCF_settings.others_lr):
        super(NaiveBertHybrid, self).__init__()
        self.plm_path = plm_path
        self.bert = BertModel.from_pretrained(plm_path)
        self.hidden = self.bert.config.hidden_size
        self.ner_cnt = ner_cnt
        self.sentiment_cnt = sentiment_cnt
        self.plm_lr = plm_lr
        self.others_lr = others_lr

        # 识别ner tag
        self.ner_cls = nn.Linear(self.hidden, ner_cnt)

        # 识别情感分类
        self.sentiment_cls = nn.Linear(self.hidden, sentiment_cnt)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.ner_cls.weight)
        self.ner_cls.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.sentiment_cls.weight)
        self.sentiment_cls.bias.data.fill_(0)

    def forward(self,
                input_ids: torch.Tensor,
                token_type_ids: torch.Tensor,
                attention_mask: torch.Tensor):
        result = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        output, pooled = result[0], result[1]
        # pooled (bsz, hidden), output (bsz, seq_l, hidden)

        sentiment_result = self.sentiment_cls(pooled)  # (bsz, sentiment_cnt))
        ner_result = self.ner_cls(output)  # (bsz, seq_l, ner_cnt)

        # sentiment_result = F.softmax(sentiment_output, dim=-1)  # (bsz, sentiment_cnt)
        # ner_result = F.softmax(ner_output, dim=-1)  # (bsz, seq_l, ner_cnt)

        return {
            "ner_result": ner_result,  # (bsz, seq_l, ner_cnt)
            "sentiment": sentiment_result  # (bsz, sentiment_cnt)
        }

    def get_optimizers(self):
        plm_params = self.bert.parameters()

        ner_params = self.ner_cls.parameters()
        sentiment_params = self.sentiment_cls.parameters()

        plm_optimizer = AdamW(params=plm_params, lr=self.plm_lr)
        others_optimizer = AdamW(params=chain(ner_params, sentiment_params), lr=self.others_lr)
        return [plm_optimizer, others_optimizer]


class NaiveBertHybrid_Loss(nn.Module):
    def __init__(self, lmd=CCF_settings.lmd, enable_focal=False, gamma=2):
        super(NaiveBertHybrid_Loss, self).__init__()
        self.enable_focal = enable_focal
        self.gamma = gamma
        self.lmd = lmd
        self.RECORDER_loss = {
            'ner_result': None,
            '_ner_result': ['batch', 'np.ndarray (seq_l, ner_type)'],
            'ner_target': None,
            '_ner_target': ['batch', 'np.ndarray (seq_l)'],
            'sentiment': None,
            '_sentiment': ['batch', 'sentiment_type'],
            'sentiment_target': None,
            '_sentiment_target': ['batch'],
            'loss': None,
            '_loss': [{'loss_component': ('ner_loss', 'sentiment_loss', 'total_loss')}, 'float']
        }

    def forward(self,
                ner_result: torch.Tensor,
                sentiment: torch.Tensor,
                ner_target: torch.Tensor,
                sentiment_target: torch.Tensor):
        """

        :param ner_result: (bsz, seq_l, ner_cnt)
        :param sentiment: (bsz, sent_cnt)
        :param ner_target: (bsz, seq_l)
        :param sentiment_target: (bsz)
        :return:
        """
        self.RECORDER_loss['ner_result'] = np.array(ner_result.clone().detach().cpu())
        self.RECORDER_loss['ner_target'] = np.array(ner_target.clone().detach().cpu())
        self.RECORDER_loss['sentiment'] = np.array(sentiment.clone().detach().cpu())
        self.RECORDER_loss['sentiment_target'] = np.array(sentiment_target.clone().detach().cpu())
        bsz, seq_l, ner_cnt = ner_result.shape
        ner_result = ner_result.view(bsz * seq_l, ner_cnt)
        ner_target = ner_target.view(bsz * seq_l)
        ner_loss = F.cross_entropy(ner_result, ner_target)

        if self.enable_focal:
            ce_loss = F.cross_entropy(sentiment, sentiment_target, reduction='none')
            pt = torch.exp(ce_loss)
            sentiment_loss = torch.mean(((1 - pt) ** self.gamma) * ce_loss)
        else:
            sentiment_loss = F.cross_entropy(sentiment, sentiment_target)

        loss = self.lmd * ner_loss + (1 - self.lmd) * sentiment_loss
        self.RECORDER_loss['loss'] = [float(ner_loss), float(sentiment_loss), float(loss)]

        return loss


class NaiveNER_HybridEvaluator(BaseEvaluator):
    def __init__(self):
        super(NaiveNER_HybridEvaluator, self).__init__()
        self.f1_evaluator = F1_Evaluator()
        self.kappa_evaluator = KappaEvaluator()
        self.RECORDER_preds = {
            "ner_pred_lst": [],
            '_ner_pred_lst': ['sample', 'List[ner label str], ner label of a sentence'],
            'ner_gt_lst': [],
            '_ner_gt_lst': ['sample', 'List[ner label str], ner label of a sentence'],
            'sentiment_pred_lst': [],
            '_sentiment_pred_lst': ['sample', 'int, sentiment label of a sentence'],
            'sentiment_gt_lst': [],
            '_sentiment_gt_lst': ['sample', 'int, sentiment label of a sentence'],
            'f1_score': {},
            '_f1_score': ['Dict[metric type str], metric value'],
            "kappa_score": {},
            '_kappa_score': ['Dict[metric type str], metric value']
        }

    def eval_single(self,
                    ner_result: torch.Tensor,
                    ner_string_gt: List[str],
                    sentiment: torch.Tensor,
                    sentiment_gt: int):
        """
        bsz = 1
        :param ner_result: (bsz, seq_l, ner_cnt)
        :param ner_string_gt:
        :param sentiment: (bsz, sent_cnt)
        :param sentiment_gt:
        :return:
        """
        ner_result = ner_result.squeeze(dim=0)  # (seq_l, ner_cnt)
        ner_tag_result = NER_utils.tensor_to_ner_label(ner_result)  # List[str]
        sentiment = sentiment.squeeze(dim=0)  # (sent_cnt)
        _, i = torch.max(sentiment, dim=0)
        sentiment_result = i.tolist()  # int
        self.f1_evaluator.eval_single(ner_tag_result, ner_string_gt)  # TODO'
        self.kappa_evaluator.eval_single(sentiment_result, sentiment_gt)

    def eval_step(self) -> Dict[str, Any]:
        self.RECORDER_preds['ner_pred_lst'] = self.f1_evaluator.pred_lst
        self.RECORDER_preds['ner_gt_lst'] = self.f1_evaluator.gt_lst
        self.RECORDER_preds['sentiment_pred_lst'] = self.kappa_evaluator.pred_lst
        self.RECORDER_preds['sentiment_gt_lst'] = self.kappa_evaluator.gt_lst
        sf1 = self.f1_evaluator.eval_step()
        skappa = self.kappa_evaluator.eval_step()
        self.RECORDER_preds['f1_score'] = sf1
        self.RECORDER_preds['kappa_score'] = skappa

        result = {}
        result.update(sf1)
        result.update(skappa)
        return result


def train_dataset_factory(data_dicts: List[Dict[str, Any]], bsz: int = CCF_settings.default_bsz, shuffle = CCF_settings.default_shuffle):
    """

    :param data_dicts: text, BIO_anno, class
    :param bsz:
    :param shuffle:
    :return:
    """
    # data_dicts content
    # [text, BIO_anno, class]

    # tokenize
    data_dict = tools.transpose_list_of_dict(data_dicts)
    lst_tokenizer = tools.bert_tokenizer()
    text_result = lst_tokenizer(data_dict['text'])
    text_result = tools.transpose_list_of_dict(text_result)
    data_dict.update(text_result)
    data_dicts = tools.transpose_dict_of_list(data_dict)
    # [text, input_ids, token_type_ids, attention_mask, offset_mapping, BIO_anno, class]

    # generate ner label
    def generate_ner_label(datadict: Dict[str, Any]):
        mappings = datadict['offset_mapping']
        charBIO = datadict['BIO_anno'].split(' ')
        tokenBIO_tailed = NER_utils.charBIO_to_tokenBIO_with_tail(charBIO, mappings)
        datadict['BIO_anno_token'] = tokenBIO_tailed
        return [datadict]
    data_dicts = tools.map_operation_to_list_elem(generate_ner_label, data_dicts)
    # [text, input_ids, token_type_ids, attention_mask, offset_mappings, BIO_anno, BIO_anno_token, class]

    # convert ner label to tensor
    def convert_ner_label_to_tensor(datadict: Dict[str, Any]):
        tokenBIO = datadict['BIO_anno_token']
        ner_target = np.zeros([len(tokenBIO)], dtype=np.long)
        for idx, token in enumerate(tokenBIO):
            tag_idx = CCF_settings.ner_tags_idx[token]
            ner_target[idx] = tag_idx
        datadict['ner_target'] = ner_target  # (seq_l, ner_cnt)
        return [datadict]
    data_dicts = tools.map_operation_to_list_elem(convert_ner_label_to_tensor, data_dicts)
    # [text, input_ids, token_type_ids, attention_mask, offset_mapping, BIO_anno, BIO_anno_token, class, ner_target]

    train_dataset = SimpleDataset(data_dicts)

    def collate_fn(lst):
        dict_of_data = tools.transpose_list_of_dict(lst)
        input_ids = torch.tensor(tools.batchify_ndarray([np.array(x) for x in dict_of_data['input_ids']]))
        token_type_ids = torch.tensor(tools.batchify_ndarray([np.array(x) for x in dict_of_data['token_type_ids']]))
        attention_mask = torch.tensor(tools.batchify_ndarray([np.array(x) for x in dict_of_data['attention_mask']]))
        ner_target = torch.tensor(tools.batchify_ndarray(dict_of_data['ner_target']))  # (bsz, seq_l)
        sentiment_target = torch.tensor(np.array(dict_of_data['class'], dtype=np.long))  # (bsz)
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
               }, {
            'ner_target': ner_target,
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
        ner_string_gt = dict_of_data['BIO_anno'][0].split(' ')
        sentiment_gt = dict_of_data['class'][0]
        return {
            'input_ids': input_ids,
            "token_type_ids": token_type_ids,
            'attention_mask': attention_mask
               }, {
            'ner_string_gt': ner_string_gt,
            'sentiment_gt': sentiment_gt
        }

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return val_dataloader


def dataset_factory(file_dir: str, bsz: int = CCF_settings.default_bsz, shuffle = CCF_settings.default_shuffle):
    """

    :param file_dir:
    :param bsz:
    :param shuffle:
    :return:
    """
    trainfile_name = 'train_data_public.csv'
    if file_dir[-1] != '/':
        file_dir += '/'
    filename = file_dir + trainfile_name
    data_dicts = tools.read_csv_as_datadict(filename)
    train_datadicts, val_datadicts = tools.split_list_with_ratio(data_dicts, CCF_settings.default_train_val_split)

    train_dataloader = train_dataset_factory(train_datadicts, bsz, shuffle)
    val_dataloader = val_dataset_factory(val_datadicts)
    return train_dataloader, val_dataloader



model_registry = {
    'model': NaiveBertHybrid,
    "evaluator": NaiveNER_HybridEvaluator,
    "loss": NaiveBertHybrid_Loss,
    'dataset': dataset_factory,
    'recorder': AutoRecorder,
    'args': [
        {'name': "--file_dir", 'dest': 'file_dir', 'type': str, 'help': '训练/测试数据文件的路径'},
        {'name': "--save_path", 'dest': 'save_path', 'type': str, 'default': 'work/NER', 'help': '训练/测试数据文件的路径'},
        {'name': "--focal_gamma", 'dest': 'gamma', 'type': float, 'default': 2, 'help': 'focal loss的gamma参数'},
    ]
}
