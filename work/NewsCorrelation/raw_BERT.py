import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler
from transformers import BertModel, AutoTokenizer, AutoModel
from itertools import chain, combinations
import numpy as np
import random
import pickle

from type_def import *
from utils.data import SimpleDataset
from work.NewsCorrelation import newsco_settings
from work.NewsCorrelation import newsco_utils
from models.Loss.regular_loss import Scalar_Loss
from evaluate.evaluator import Pearson_Evaluator
from analysis.recorder import NaiveRecorder
from utils import batch_tool, tools, tokenize_tools
from models.model_utils import get_init_params


class BERT_for_Sentence_Similarity(nn.Module):
    def __init__(self, pretrain_path: str = newsco_settings.pretrain_path, linear_lr: float = newsco_settings.linear_lr, plm_lr: float = newsco_settings.plm_lr):
        """

        :param pretraine_path:
        """
        super(BERT_for_Sentence_Similarity, self).__init__()
        self.plm_path = pretrain_path
        self.init_params = get_init_params(locals())
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
        relu_output = self.linear2(relu_output)  # (bsz, 7)
        if self.training:
            return {
                "output": relu_output  # (bsz, 7)
            }
        else:
            relu_output = float(relu_output)
            return {
                "pred": relu_output
            }


class BERT_for_SS_Two_Tower(nn.Module):
    def __init__(self, pretrain_path: str = newsco_settings.pretrain_path, linear_lr: float = newsco_settings.linear_lr, plm_lr: float = newsco_settings.plm_lr):
        """

        :param pretraine_path:
        """
        super().__init__()
        self.plm_path = pretrain_path
        self.init_params = get_init_params(locals())
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
                input_ids1: torch.Tensor,
                attention_mask1: torch.Tensor,
                input_ids2: torch.Tensor,
                attention_mask2: torch.Tensor):
        """

        :param input_ids: (bsz, seq_l)
        :param attention_mask: (bsz, seq_l)
        :return:
        """
        bert_output = self.bert(input_ids1, attention_mask1)
        hidden_output1 = bert_output[0]  # (bsz, seq_l, hidden_size)
        bert_output2 = self.bert(input_ids2, attention_mask2)
        hidden_output2 = bert_output2[0]  # (bsz, seq_l, hidden_size)
        hidden_output = torch.cat([hidden_output1, hidden_output2], dim=1)  # bsz, seq_l * 2, hidden_size)
        hidden_mean = torch.mean(hidden_output, dim=1)  # (bsz, hidden_size)
        linear_output = self.linear1(hidden_mean)  # (bsz, layer)
        relu_output = self.relu(linear_output)  # (bsz, layer)
        relu_output = self.linear2(relu_output)  # (bsz, 7)
        if self.training:
            return {
                "output": relu_output  # (bsz, 7)
            }
        else:
            relu_output = float(relu_output)
            return {
                "pred": relu_output
            }


class Pearson_Loss(nn.Module):
    """
    为了pearson系数优化的loss
    """
    def __init__(self, var_weight: float = 0.3):
        super(Pearson_Loss, self).__init__()
        self.var_weight = var_weight
        self.scalar_loss = Scalar_Loss()

    def forward(self, output: torch.Tensor, label: torch.Tensor):
        """

        :param output:
        :param label:
        :return:
        """
        scalarl = self.scalar_loss(output, label)
        var_loss = F.mse_loss(torch.var(output), torch.var(label))
        loss = (1 - self.var_weight) * scalarl + self.var_weight * var_loss
        return loss


def news_triplet_loss(output_triplet, label_triplet):
    o_s, o_m, o_e = output_triplet
    l_s, l_m, l_e = label_triplet
    start_distance = F.mse_loss(o_m - o_s, l_m - l_s)
    end_distance = F.mse_loss(o_e - o_m, l_e - l_m)
    return start_distance + end_distance


def news_triplet_triangle_loss(output_triplet, label_triplet):
    o_s, o_m, o_e = output_triplet
    l_s, l_m, l_e = label_triplet
    s_distance = F.mse_loss(o_m - o_s, l_m - l_s)
    m_distance = F.mse_loss(o_e - o_m, l_e - l_m)
    e_distance = F.mse_loss(o_s - o_e, l_s - l_e)
    return s_distance + m_distance + e_distance


class NewsTriplet_Loss(nn.Module):
    """
    对于传入的batch，在batch之内搜索三元组，并

    - 对于单机多卡模式不兼容，因为需要在batch之内寻找三元组
    针对于semeval问题的TripletLoss需不需要做一些修改？
    """
    def __init__(self, triplet_maximum: int = 5, triangle_loss: bool = False):
        """

        :param triplet_maximum: 在batch中搜索triplet的最大数目
        """
        super(NewsTriplet_Loss, self).__init__()
        self.triplet_maximum = triplet_maximum
        self.triangle_loss = triangle_loss

    def forward(self, output: torch.Tensor, label: torch.Tensor):
        """

        :param output: (bsz, 1)
        :param label: (bsz, 1)
        :return:
        """
        bsz = output.size(0)
        if bsz < 3:
            raise Exception(f'[NewsTripletLoss]batch大小必须>=3!输入大小为:{bsz}')

        # 生成所有的三元组排列，打乱，并选取triplet_maximum个
        triplets = list(combinations(list(range(bsz)), 3))
        random.shuffle(triplets)
        triplets = triplets[:self.triplet_maximum]

        triple_loss = 0
        for triplet in triplets:
            triplet_index = torch.LongTensor(triplet)  # () of 3 element
            if output.is_cuda:
                triplet_index = triplet_index.cuda()
            o_s, o_m, o_e = output.index_select(0, triplet_index)
            l_s, l_m, l_e = label.index_select(0, triplet_index)
            if self.triangle_loss:
                cur_triple_loss = news_triplet_triangle_loss((o_s, o_m, o_e), (l_s, l_m, l_e))
            else:
                cur_triple_loss = news_triplet_loss((o_s, o_m, o_e), (l_s, l_m, l_e))
            triple_loss += cur_triple_loss
        return triple_loss


class RawBERT_Loss(nn.Module):
    def __init__(self, tl_weight: float = 0.2, tl_max: int = 5):
        super(RawBERT_Loss, self).__init__()
        self.tl_weight = tl_weight
        self.scalar_loss = Scalar_Loss()
        self.triplet_loss = NewsTriplet_Loss(triplet_maximum=tl_max)

    def forward(self, output: torch.Tensor, label: torch.Tensor):
        """

        :param output: (bsz, 1)
        :param label: (bsz, 1)
        :return:
        """
        scalarloss = self.scalar_loss(output, label)
        tripleloss = self.triplet_loss(output, label)
        loss = (1 - self.tl_weight) * scalarloss + self.tl_weight * tripleloss
        return loss


def dataset_factory(
        crawl_dir: str = newsco_settings.crawl_file_dir,
        newspair_file: str = newsco_settings.newspair_file,
        pretrain_path: str = newsco_settings.pretrain_path,
        bsz: int = newsco_settings.bsz,
        local_rank: int = -1,
        control_name: str = 'default',
        newspair_valid_file: str = 'None'):
    """

    :param crawl_dir:
    :param newspair_file:
    :param pretrain_path:
    :param bsz:
    :param local_rank:
    :param control_name: 尝试加入的参数
    :return:
    """
    print(f'get local rank: {local_rank}')
    print('[dataset_factory]building samples...', end=' ... ')
    samples = newsco_utils.build_news_samples(crawl_dir, newspair_file)

    if newspair_valid_file != 'None':
        train_samples = samples
        val_samples = newsco_utils.build_news_samples(crawl_dir, newspair_valid_file)
        pickle.dump(train_samples + val_samples, open(f'{control_name}-CachedSamples.pk', 'wb'))  # [临时]将初步处理完成的数据写入
    else:
        random.shuffle(samples)
        pickle.dump(samples, open(f'{control_name}-CachedSamples.pk', 'wb'))  # [临时]将初步处理完成的数据写入
        train_samples, val_samples = batch_tool.train_val_split(samples, newsco_settings.train_ratio)


    # 划分训练、评价
    # train_samples, val_samples = batch_tool.train_val_split(samples, newsco_settings.train_ratio)
    train_indices, val_indices = list(x['index'] for x in train_samples), list(x['index'] for x in val_samples)
    pickle.dump({
        "train_indices": train_indices,
        "val_indices": val_indices,
    }, open(f'{control_name}-RecordedInfos.pk', 'wb'))
    train_dataloader = train_dataset_factory(train_samples, pretrain_path, bsz, local_rank=local_rank)

    if local_rank in [-1, 0]:
        val_dataloader = eval_dataset_factory(val_samples, pretrain_path)
    else:
        val_dataloader = None

    return train_dataloader, val_dataloader


def dataset_factory_tow_tower(
        crawl_dir: str = newsco_settings.crawl_file_dir,
        newspair_file: str = newsco_settings.newspair_file,
        pretrain_path: str = newsco_settings.pretrain_path,
        bsz: int = newsco_settings.bsz,
        local_rank: int = -1,
        control_name: str = 'default',
        newspair_valid_file: str = 'None'):
    """

    :param crawl_dir:
    :param newspair_file:
    :param pretrain_path:
    :param bsz:
    :param local_rank:
    :param control_name: 尝试加入的参数
    :return:
    """
    print(f'get local rank: {local_rank}')
    print('[dataset_factory]building samples...', end=' ... ')
    samples = newsco_utils.build_news_samples(crawl_dir, newspair_file)

    if newspair_valid_file != 'None':
        train_samples = samples
        val_samples = newsco_utils.build_news_samples(crawl_dir, newspair_valid_file)
        pickle.dump(train_samples + val_samples, open(f'{control_name}-CachedSamples.pk', 'wb'))  # [临时]将初步处理完成的数据写入
    else:
        random.shuffle(samples)
        pickle.dump(samples, open(f'{control_name}-CachedSamples.pk', 'wb'))  # [临时]将初步处理完成的数据写入
        train_samples, val_samples = batch_tool.train_val_split(samples, newsco_settings.train_ratio)


    # 划分训练、评价
    # train_samples, val_samples = batch_tool.train_val_split(samples, newsco_settings.train_ratio)
    train_indices, val_indices = list(x['index'] for x in train_samples), list(x['index'] for x in val_samples)
    pickle.dump({
        "train_indices": train_indices,
        "val_indices": val_indices,
    }, open(f'{control_name}-RecordedInfos.pk', 'wb'))
    train_dataloader = train_dataset_factory(train_samples, pretrain_path, bsz, local_rank=local_rank, two_tower=True)

    if local_rank in [-1, 0]:
        val_dataloader = eval_dataset_factory(val_samples, pretrain_path, two_tower=True)
    else:
        val_dataloader = None

    return train_dataloader, val_dataloader


def sample2input(sample: dict, addon: Dict[str, Any]):
    """
    将一个新闻sample转换为能够输入到eval模型的状态
    :param sample:
        - crawl1
            -- title
            -- text
        - crawl2
            -- title
            -- text
    :param addon:
        - title_tokenizer
        - text_tokenizer
    :return:
    """
    crawl1, crawl2 = sample['crawl1'], sample['crawl2']
    title1, text1, title2, text2 = crawl1['title'], crawl1['text'], crawl2['title'], crawl2['text']
    title_tokenizer, text_tokenizer = addon['title_tokenizer'], addon['text_tokenizer']

    title1_result = tools.transpose_list_of_dict(title_tokenizer([title1]))  # Dict[str, list]
    title2_result = tools.transpose_list_of_dict(title_tokenizer([title2]))  # Dict[str, list]
    text1_result = tools.transpose_list_of_dict(text_tokenizer([text1]))  # Dict[str, list]
    text2_result = tools.transpose_list_of_dict(text_tokenizer([text2]))  # Dict[str, list]

    tokenize_result = {
        'title1_input_ids': title1_result['input_ids'],
        'title2_input_ids': title2_result['input_ids'],
        'text1_input_ids': text1_result['input_ids'],
        'text2_input_ids': text2_result['input_ids'],
    }
    tokenize_result_dicts = tools.transpose_dict_of_list(tokenize_result)  # List[dict]
    concat_results = list(newsco_utils.xlmr_sentence_concat_cza(
        [x['title1_input_ids'], x['text1_input_ids'], x['title2_input_ids'], x['text2_input_ids']]
    ) for x in tokenize_result_dicts)  # List[IntList] list of concatenated input_ids
    attention_masks = list(newsco_utils.generate_attention_mask(x, newsco_settings.max_seq_length) for x in concat_results)
    padded_input_ids = list(newsco_utils.pad_input_ids(x, newsco_settings.max_seq_length) for x in concat_results)

    # breakpoint()

    padded_input_ids = torch.tensor(padded_input_ids, dtype=torch.long).cuda()
    attention_masks = torch.tensor(attention_masks, dtype=torch.long).cuda()

    sample_input = {
        "input_ids": padded_input_ids,  # (1, seq_l)
        "attention_mask": attention_masks  # (1, seq_l)
    }
    return sample_input


def convert_parsed_sample_to_model_sample(samples, plm_path, two_tower: bool = False) -> List[Dict[str, Any]]:
    """
    将utils中从文件读取并合并的样本，转化为适合模型输入处理的样本
    :param samples:
    :param plm_path:
    :return:
    """
    # 首先提取出train所需的数据
    sample_dict = tools.transpose_list_of_dict(samples)
    crawl1s, crawl2s = sample_dict['crawl1'], sample_dict['crawl2']
    crawl1_dict, crawl2_dict = tools.transpose_list_of_dict(crawl1s), tools.transpose_list_of_dict(crawl2s)

    scores = list(float(x) for x in sample_dict['Overall'])  # List[float]
    # all_scores = []  # List[List[float]]
    # for elem_score_type in ['Overall', 'Geography', 'Entities', 'Time', 'Narrative', 'Style', 'Tone']:
    #     all_scores.append(list(float(x) for x in sample_dict[elem_score_type]))  # appended List[float]
    # all_scores = np.array(all_scores)  # (7, train_sample_cnt)
    title1_lst, title2_lst, text1_lst, text2_lst = \
        crawl1_dict['title'], crawl2_dict['title'], crawl1_dict['text'], crawl2_dict['text']

    if two_tower:
        text_tokenizer = tokenize_tools.xlmr_tokenizer(max_len=newsco_settings.max_seq_length - 4 - newsco_settings.title_len, plm_path=plm_path)
        title_tokenizer = tokenize_tools.xlmr_tokenizer(max_len=newsco_settings.title_len, plm_path=plm_path)
    else:
        title_tokenizer = tokenize_tools.xlmr_tokenizer(max_len=newsco_settings.title_len, plm_path=plm_path)
        text_tokenizer = tokenize_tools.xlmr_tokenizer(max_len=newsco_settings.text_len, plm_path=plm_path)
    title1_result = tools.transpose_list_of_dict(title_tokenizer(title1_lst))  # Dict[str, list]
    title2_result = tools.transpose_list_of_dict(title_tokenizer(title2_lst))  # Dict[str, list]
    text1_result = tools.transpose_list_of_dict(text_tokenizer(text1_lst))  # Dict[str, list]
    text2_result = tools.transpose_list_of_dict(text_tokenizer(text2_lst))  # Dict[str, list]

    tokenize_result = {
        'title1_input_ids': title1_result['input_ids'],
        'title2_input_ids': title2_result['input_ids'],
        'text1_input_ids': text1_result['input_ids'],
        'text2_input_ids': text2_result['input_ids'],
    }
    tokenize_result_dicts = tools.transpose_dict_of_list(tokenize_result)  # List[dict]
    if two_tower:
        concat_results1 = list(newsco_utils.xlmr_sentence_concat([x['title1_input_ids'], x['text1_input_ids']]) for x in tokenize_result_dicts)
        concat_results2 = list(newsco_utils.xlmr_sentence_concat([x['title2_input_ids'], x['text2_input_ids']]) for x in tokenize_result_dicts)
        attention_masks1 = list(newsco_utils.generate_attention_mask(x, newsco_settings.max_seq_length) for x in concat_results1)
        attention_masks2 = list(newsco_utils.generate_attention_mask(x, newsco_settings.max_seq_length) for x in concat_results2)
        padded_input_ids1 = list(newsco_utils.pad_input_ids(x, newsco_settings.max_seq_length) for x in concat_results1)
        padded_input_ids2 = list(newsco_utils.pad_input_ids(x, newsco_settings.max_seq_length) for x in concat_results2)
        train_samples = {
            'label': scores,  # List[float]
            "input_ids1": padded_input_ids1,  # List[IntList]
            "attention_mask1": attention_masks1,  # List[IntList]
            "input_ids2": padded_input_ids2,  # List[IntList]
            "attention_mask2": attention_masks2  # List[IntList]
        }
        train_samples = tools.transpose_dict_of_list(train_samples)
        return train_samples
    else:
        concat_results = list(newsco_utils.xlmr_sentence_concat_cza(
            [x['title1_input_ids'], x['text1_input_ids'], x['title2_input_ids'], x['text2_input_ids']]
        ) for x in tokenize_result_dicts)  # List[IntList] list of concatenated input_ids
        attention_masks = list(newsco_utils.generate_attention_mask(x, newsco_settings.max_seq_length) for x in concat_results)
        padded_input_ids = list(newsco_utils.pad_input_ids(x, newsco_settings.max_seq_length) for x in concat_results)

        train_samples = {
            "label": scores,  # List[float]
            "input_ids": padded_input_ids,  # List[IntList]
            "attention_mask": attention_masks  # List[IntList]
        }
        train_samples = tools.transpose_dict_of_list(train_samples)
        return train_samples


def train_dataset_factory(samples: List[Dict[str, Any]], pretrain_path: str, bsz: int, local_rank: int, two_tower: bool = False):
    """

    :param samples:
    :return:
    """
    # 首先提取出train所需的数据
    train_samples = convert_parsed_sample_to_model_sample(samples, pretrain_path, two_tower)
    train_dataset = SimpleDataset(train_samples)
    if local_rank != -1:
        train_dataset_sampler = DistributedSampler(train_dataset, shuffle=True)

    def collate_fn(lst):
        """
        :param lst: [
            {
            "label",
            "input_ids",
            "attention_mask",
            }
        ]
        :return:
        """
        label = torch.Tensor(list(x['label'] for x in lst))  # (bsz, 1)
        input_ids = torch.tensor(list(x['input_ids'] for x in lst), dtype=torch.long)  # (bsz, max_seq_l)
        attention_mask = torch.tensor(list(x['attention_mask'] for x in lst), dtype=torch.long)  # (bsz, max_seq_l)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }, {
            "label": label
        }

    def collate_fn_two_tower(lst):
        """

        :param lst: [{
            "label",
            "input_ids1",
            "input_ids2",
            "attention_mask1",
            "attention_mask2"
        }]
        :return:
        """
        label = torch.Tensor(list(x['label'] for x in lst))  # (bsz, 1)
        input_ids1 = torch.tensor(list(x['input_ids1'] for x in lst), dtype=torch.long)  # (bsz, max_seq_l)
        attention_mask1 = torch.tensor(list(x['attention_mask1'] for x in lst), dtype=torch.long)  # (bsz, max_seq_l)
        input_ids2 = torch.tensor(list(x['input_ids2'] for x in lst), dtype=torch.long)  # (bsz, max_seq_l)
        attention_mask2 = torch.tensor(list(x['attention_mask2'] for x in lst), dtype=torch.long)  # (bsz, max_seq_l)
        return {
            "input_ids1": input_ids1,
            "attention_mask1": attention_mask1,
            "input_ids2": input_ids2,
            "attention_mask2": attention_mask2
        }, {
            "label": label
        }
    if two_tower:
        collate_fn = collate_fn_two_tower
    if local_rank == -1:
        train_dataloader = DataLoader(train_dataset, batch_size=bsz, shuffle=True, collate_fn=collate_fn, drop_last=True)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=bsz, collate_fn=collate_fn, sampler=train_dataset_sampler)
    return train_dataloader


def eval_dataset_factory(samples: List[Dict[str, Any]], pretrain_path: str, two_tower: bool = False):
    """

    :param samples:
    :return:
    """
    val_samples = convert_parsed_sample_to_model_sample(samples, pretrain_path, two_tower=two_tower)
    # label, input_ids, attention_mask
    val_dataset = SimpleDataset(val_samples)
    def collate_fn(lst):
        target = lst[0]['label']
        input_ids = torch.tensor(list(x['input_ids'] for x in lst), dtype=torch.long)  # (1, max_seq_l)
        attention_mask = torch.tensor(list(x['attention_mask'] for x in lst), dtype=torch.long)  # (1, max_seq_l)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
               }, {
            "gt": target
        }
    def collate_fn_two_tower(lst):
        target = lst[0]['label']
        input_ids1 = torch.tensor(list(x['input_ids1'] for x in lst), dtype=torch.long)  # (1, max_seq_l)
        attention_mask1 = torch.tensor(list(x['attention_mask1'] for x in lst), dtype=torch.long)  # (1, max_seq_l)
        input_ids2 = torch.tensor(list(x['input_ids2'] for x in lst), dtype=torch.long)  # (1, max_seq_l)
        attention_mask2 = torch.tensor(list(x['attention_mask2'] for x in lst), dtype=torch.long)  # (1, max_seq_l)
        return {
            "input_ids1": input_ids1,
            "attention_mask1": attention_mask1,
            "input_ids2": input_ids2,
            "attention_mask2": attention_mask2
               }, {
            "gt": target
        }
    if two_tower:
        collate_fn = collate_fn_two_tower
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return val_dataloader


class UseModel:
    def __init__(self, state_dict_path: str, init_params_path):
        init_params = pickle.load(open(init_params_path, 'rb'))
        self.model = BERT_for_Sentence_Similarity(**init_params)
        self.model.load_state_dict(torch.load(open(state_dict_path, 'rb'), map_location=torch.device('cpu')))
        self.model.eval()
        self.model.cuda()

        self.title_tokenizer = tokenize_tools.xlmr_tokenizer(max_len=newsco_settings.title_len, plm_path=self.model.plm_path)
        self.text_tokenizer = tokenize_tools.xlmr_tokenizer(max_len=newsco_settings.text_len, plm_path=self.model.plm_path)

    def __call__(self, title1: str, text1: str, title2: str, text2: str):
        # 这里先按照爬取文件的格式，构造sample
        crawl1 = {
            "title": title1,
            "text": text1
        }
        crawl2 = {
            "title": title2,
            "text": text2
        }
        sample = {
            "crawl1": crawl1,
            "crawl2": crawl2
        }


        # 然后将sample转化为input的模式
        inp = sample2input(sample, {
            "title_tokenizer": self.title_tokenizer,
            "text_tokenizer": self.text_tokenizer
        })
        output = self.model(**inp)  # dict
        return output


model_registry = {
    "model": BERT_for_SS_Two_Tower,
    "evaluator": Pearson_Evaluator,
    "loss": RawBERT_Loss,
    "train_data": train_dataset_factory,
    "val_data": eval_dataset_factory,
    'train_val_data': dataset_factory_tow_tower,
    'recorder': NaiveRecorder,
    'use': UseModel
}

