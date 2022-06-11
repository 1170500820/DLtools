import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

from transformers import BertModel, BertTokenizerFast, AdamW


from type_def import *
from evaluate.evaluator import BaseEvaluator, F1_Evaluator
from work.NER import NER_utils, NER_settings
from utils import tools
from utils.data import SimpleDataset


class BERT_NER(nn.Module):
    def __init__(self,
                 bert_plm_path: str = NER_settings.default_plm,
                 ner_cnt: int = len(NER_settings.msra_ner_tags),
                 plm_lr: float = NER_settings.plm_lr,
                 others_lr: float = NER_settings.others_lr):
        super(BERT_NER, self).__init__()

        # 保存初始化参数
        self.plm_path = bert_plm_path
        self.ner_cnt = ner_cnt
        self.plm_lr = plm_lr
        self.others_lr = others_lr

        # 记载bert预训练模型
        self.bert = BertModel.from_pretrained(bert_plm_path)
        self.hidden = self.bert.config.hidden_size

        # 识别ner tag
        self.ner_cls = nn.Linear(self.hidden, ner_cnt)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.ner_cls.weight)
        self.ner_cls.bias.data.fill_(0)

    def get_optimizers(self):
        plm_params = self.bert.parameters()

        ner_params = self.ner_cls.parameters()

        plm_optimizer = AdamW(params=plm_params, lr=self.plm_lr)
        others_optimizer = AdamW(params=ner_params, lr=self.others_lr)
        return [plm_optimizer, others_optimizer]


    def forward(self, input_ids, token_type_ids, attention_mask):
        """

        :param input_ids: (bsz, seq_l (+2))
        :param token_type_ids: (bsz, seq_l (+2))
        :param attention_mask: (bsz, seq_l (+2))
        :return:
        """
        result = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        output, pooled = result[0], result[1]
        # pooled (bsz, hidden), output (bsz, seq_l, hidden)

        ner_result = self.ner_cls(output)  # (bsz, seq_l, ner_cnt)

        return {
            "ner_result": ner_result  # (bsz, seq_l, ner_cnt)
        }


class BERT_NER_Loss(nn.Module):
    def __init__(self):
        super(BERT_NER_Loss, self).__init__()

    def forward(self,
                ner_result: torch.Tensor,
                ner_target: torch.Tensor):
        """
        先实现一个最简单的版本
        :param ner_result: (bsz, seq_l, ner_cnt)
        :param ner_target: (bsz, seq_l)
        :return:
        """
        bsz, seq_l, ner_cnt = ner_result.shape
        ner_result = ner_result.view(bsz * seq_l, ner_cnt)
        ner_target = ner_target.view(bsz * seq_l)
        ner_loss = F.cross_entropy(ner_result, ner_target)

        return ner_loss


class BERT_NER_Evaluator(BaseEvaluator):
    def __init__(self):
        super(BERT_NER_Evaluator, self).__init__()
        self.f1_evaluator = F1_Evaluator()

    def eval_single(self,
                    ner_result: torch.Tensor,
                    ner_gt: List[str]):
        """

        :param ner_result:
        :param ner_string_gt:
        :return:
        """
        ner_result = ner_result.squeeze(dim=0)  # (seq_l, ner_cnt)
        ner_tag_result = NER_utils.tensor_to_ner_label(ner_result)  # List[str]
        self.f1_evaluator.eval_single(ner_tag_result, ner_gt)

    def eval_step(self) -> Dict[str, Any]:
        sf1 = self.f1_evaluator.eval_step()
        return sf1


def train_dataset_factory(data_dicts: List[Dict[str, Any]], bsz: int = NER_settings.default_bsz, shuffle: bool = NER_settings.default_shuffle, data_type='msra'):
    """
    each data_dict contains:
    - token List[str]
    - input_ids np.ndarray
    - token_type_ids np.ndarray
    - attention_mask np.ndarray
    - offset_mapping List[Tuple[int, int]]
    - chars List[str]
    - tags List[str]
    - token_tags List[str]
    :param data_dicts:
    :param bsz:
    :param shuffle:
    :param data_type: msra, weibo (, ontonotes, resume待添加)
    :return:
    """
    # 用token_tags构建target ndarray
    data_dict = tools.transpose_list_of_dict(data_dicts)
    if data_type == 'msra':
        data_dict['target'] = list(NER_utils.tags_to_ndarray(x, NER_settings.msra_ner_tags_idx) for x in data_dict['token_tags'])
    elif data_type == 'weibo':
        data_dict['target'] = list(NER_utils.tags_to_ndarray(x, NER_settings.weibo_nert_tags_idx) for x in data_dict['token_tags'])
    else:
        raise Exception(f'[train_dataset_factory]未知的训练数据来源！[{data_type}]')
    # add "target"
    data_dicts = tools.transpose_dict_of_list(data_dict)

    # build dataset and dataloader
    train_dataset = SimpleDataset(data_dicts)
    def collate_fn(lst):
        dict_of_data = tools.transpose_list_of_dict(lst)
        batchified_data = {}
        for elem_data in ['input_ids', 'token_type_ids', 'attention_mask', 'target']:
            batchified_data[elem_data] = torch.tensor(tools.batchify_ndarray_1d(dict_of_data[elem_data]))
        return {
            'input_ids': batchified_data['input_ids'],  # (bsz, seq_l)
            'token_type_ids': batchified_data['token_type_ids'],  # (bsz, seq_l)
            'attention_mask': batchified_data['attention_mask']  # (bsz, seq_l)
               }, {
            "ner_target": batchified_data['target']  # (bsz, seq_l)
        }
    train_dataloader = DataLoader(train_dataset, batch_size=bsz, shuffle=shuffle, collate_fn=collate_fn)

    return train_dataloader


def val_dataset_factory(data_dicts: List[Dict[str, Any]]):
    """
    each data_dict contains:
    - token List[str]
    - input_ids np.ndarray
    - token_type_ids np.ndarray
    - attention_mask np.ndarray
    - offset_mapping List[Tuple[int, int]]
    - chars List[str]
    - tags List[str]
    - token_tags List[str]
    :param data_dicts:
    :return:
    """
    # build dataset and dataloader
    dev_dataset = SimpleDataset(data_dicts)
    def collate_fn(lst):
        dict_of_data = tools.transpose_list_of_dict(lst)
        batchified_data = {}
        for elem_data in ['input_ids', 'token_type_ids', 'attention_mask']:
            batchified_data[elem_data] = torch.tensor(tools.batchify_ndarray_1d(dict_of_data[elem_data]))
        return {
            'input_ids': batchified_data['input_ids'],  # (bsz, seq_l)
            'token_type_ids': batchified_data['token_type_ids'],  # (bsz, seq_l)
            'attention_mask': batchified_data['attention_mask']  # (bsz, seq_l)
               }, {
            "ner_gt": dict_of_data['token_tags']  # List[List[str]]
        }
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    return dev_dataloader


def dataset_factory(file_dir: str, bsz=NER_settings.default_bsz, shuffle=NER_settings.default_shuffle, data_type='msra'):
    """
    目前只有msra和weibo数据集，然后都只使用训练集和开发集，测试集暂不启用
    weibo数据集提供了分词信息，但是暂时不使用

    dataset_factory需要将任意数据集转化为同一种数据格式，最终传入train/dev_dataset_factory的数据应当与data_type无关
    (但是实际上做不到完全无关。因为转为target array需要tag idx，而这一步在dataset_factory做又冗余了)
    :param file_dir:
    :param bsz:
    :param shuffle:
    :param data_type: msra, weibo (, ontonotes, resume待添加)
    :return:
    """
    # load data
    if data_type == 'msra':
        data_dict = NER_utils.load_msra_ner(file_dir)
        train_chars, dev_chars = tools.split_list_with_ratio(data_dict['msra_train_chars'], NER_settings.default_train_val_split)
        train_tags, dev_tags = tools.split_list_with_ratio(data_dict['msra_train_tags'], NER_settings.default_train_val_split)
    elif data_type == 'weibo':
        data_dict = NER_utils.load_weibo_ner(file_dir)
        train_chars, dev_chars = data_dict['weibo_train_chars'], data_dict['weibo_dev_chars']
        train_tags, dev_tags = data_dict['weibo_train_tags'], data_dict['weibo_dev_tags']
    else:
        raise Exception(f'[dataset_factory]未知的训练数据来源！[{data_type}]')
    train_data = {
        'chars': train_chars,
        "tags": train_tags
    }
    dev_data = {
        "chars": dev_chars,
        "tags": dev_tags
    }

    # tokenize
    lst_tokenizer = tools.bert_tokenizer()
    train_tokenized = lst_tokenizer(list(''.join(x) for x in train_data['chars']))
    train_data.update(tools.transpose_list_of_dict(train_tokenized))
    dev_tokenized = lst_tokenizer(list(''.join(x) for x in dev_data['chars']))
    dev_data.update(tools.transpose_list_of_dict(dev_tokenized))
    # {'token', 'input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'chars', 'tags'}

    # convert char_tags to token_tags
    train_data['token_tags'] = list(NER_utils.charBIO_to_tokenBIO_with_tail(x, y) for (x, y) in zip(train_data['tags'], train_data['offset_mapping']))
    dev_data['token_tags'] = list(NER_utils.charBIO_to_tokenBIO_with_tail(x, y) for (x, y) in zip(dev_data['tags'], dev_data['offset_mapping']))
    # add 'token_tags'

    # convert input_ids, ... to ndarray
    for elem_key in ['input_ids', 'token_type_ids', 'attention_mask']:
        for elem_dict in [train_data, dev_data]:
            elem_dict[elem_key] = list(np.array(x) for x in elem_dict[elem_key])

    # convert to list of dict
    train_datadicts = tools.transpose_dict_of_list(train_data)
    dev_datadicts = tools.transpose_dict_of_list(dev_data)

    # get dataloaders
    train_dataloader = train_dataset_factory(train_datadicts, bsz, shuffle)
    dev_dataloader = val_dataset_factory(dev_datadicts)

    return train_dataloader, dev_dataloader


model_registry = {
    'model': BERT_NER,
    "evaluator": BERT_NER_Evaluator,
    'loss': BERT_NER_Loss,
    "dataset": dataset_factory,
    'args': [
        {'name': "--file_dir", 'dest': 'file_dir', 'type': str, 'help': '训练/测试数据文件的路径'}]
}


if __name__ == '__main__':
    dataset_factory('../../../data/NLP/MSRA/')
