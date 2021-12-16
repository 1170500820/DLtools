import copy

import numpy as np
import json

from type_def import *
from work.RE import RE_settings
from utils import format_convert, tools, tokenize_tools
from utils.tokenize_tools import OffsetMapping



def load_NYT_re(file_dir: str):
    """
    加载NYT的经过预处理格式的数据。（预处理格式即将id都转化为词，并且分为normal、epo等类别的数据）
    将每一条数据转换为
    {
    'text': 原句，
    'triplets': [
            {
            "subject": str,
            'object': str,
            'relation': str
            }, ...
        ]
    }
    :param file_dir:
    :return:
    """
    if file_dir[-1] != '/':
        file_dir += '/'
    filenames = {
        'test': 'new_test.json',
        # 'test_epo': 'new_test_epo.json',
        # 'test_seo': 'new_test_seo.json',
        # 'test_normal': 'new_test_normal.json',
        'train': 'new_train.json',
        # 'train_epo': 'new_train_epo.json',
        # 'train_seo': 'new_train_seo.json',
        # 'train_normal': 'new_train_normal.json',
        'valid': 'new_valid.json',
        # 'valid_epo': 'new_valid_epo.json',
        # 'valid_seo': 'new_valid_seo.json',
        # 'valid_normal': 'new_valid_normal.json',
    }
    loaded = {}
    for k, v in filenames.items():
        data_lst = []
        dicts = list(map(json.loads, open(file_dir + v, 'r').read().strip().split('\n')))
        for elem in dicts:
            triplets = []  # 当前句子所对应的所有triplet
            text = elem['sentText']
            relations = elem['relationMentions']
            for elem_rel in relations:
                sub, obj, rel = elem_rel['em1Text'], elem_rel['em2Text'], elem_rel['label']
                triplets.append({
                    "subject": sub,
                    "object": obj,
                    "relation": rel
                })
            data_lst.append({
                "text": text,
                "triplets": triplets
            })
        loaded[k] = data_lst
    return loaded


def load_WebNLG_re(file_dir: str):
    """
    与load_NYT_re类似，只不过读取的是WebNLG数据集
    :param file_dir:
    :return:
    """
    if file_dir[-1] != '/':
        file_dir += '/'
    filenames = {
        'train': 'new_train.json',
        # 'train_epo': 'new_train_epo.json',
        # 'train_seo': 'new_train_seo.json',
        # 'train_normal': 'new_train_normal.json',
        'valid': 'new_valid.json',
        # 'valid_epo': 'new_valid_epo.json',
        # 'valid_seo': 'new_valid_seo.json',
        # 'valid_normal': 'new_valid_normal.json',
    }
    loaded = {}
    for k, v in filenames.items():
        data_lst = []
        dicts = list(map(json.loads, open(file_dir + v, 'r').read().strip().split('\n')))
        for elem in dicts:
            triplets = []  # 当前句子所对应的所有triplet
            text = elem['sentText']
            relations = elem['relationMentions']
            for elem_rel in relations:
                sub, obj, rel = elem_rel['em1Text'], elem_rel['em2Text'], elem_rel['label']
                triplets.append({
                    "subject": sub,
                    "object": obj,
                    "relation": rel
                })
            data_lst.append({
                "text": text,
                "triplets": triplets
            })
        loaded[k] = data_lst
    return loaded


def load_duie_re(file_dir: str):
    """
    与WebNLG和NYT相比，duie有两点不同
    1，duie的test数据不包含标签。由于duie是一个仍在进行的比赛的数据集，因此test部分没有提供标签。本函数也会读取，不够triplets字段就为空了
    2，包含复杂object。duie的object部分有时候会包含多个，比如关系"票房"的object包括一个object本体和一个"inArea"
    {
    'text': 原句，
    'triplets': [
            {
            "subject": str,
            'object': str,
            'relation': str,
            'other_objects': {
                "type": str
            }
            }, ...
        ] or None
    }
    :param file_dir:
    :return:
    """
    if file_dir[-1] != '/':
        file_dir += '/'
    filenames = {
        "valid": 'duie_dev.json/duie_dev.json',
        "train": 'duie_train.json/duie_train.json',
        "test": "duie_test2.json/duie_test2.json"
    }
    loaded = {}
    for k, v in filenames.items():
        data_lst = []
        dicts = list(map(json.loads, open(file_dir + v, 'r').read().strip().split('\n')))
        if k == 'test':
            for elem in dicts:
                text = elem['text']
                text = text.replace('\t', ' ').replace('  ', ' ').replace('\u3000', ' ').replace('\xa0', ' ')
                data_lst.append({
                    "text": text,
                    "triplets": None
                })
            loaded[k] = data_lst
        else:
            for elem in dicts:
                triplets = []  # 当前句子所对应的所有triplet
                text = elem['text']
                text = text.replace('\t', ' ').replace('  ', ' ').replace('\u3000', ' ').replace('\xa0', ' ')
                relations = elem['spo_list']
                for elem_rel in relations:
                    sub, obj, rel = elem_rel['subject'], elem_rel['object']['@value'], elem_rel['predicate']
                    sub = sub.replace('\t', ' ').replace('  ', ' ').replace('\u3000', ' ').replace('\xa0', ' ')
                    obj = obj.replace('\t', ' ').replace('  ', ' ').replace('\u3000', ' ').replace('\xa0', ' ')
                    _, other_objects = tools.split_dict(elem_rel['object'], ['@value'], keep_origin=True)
                    for elem_k in other_objects.keys():
                        other_objects[elem_k] = other_objects[elem_k].replace('\t', ' ').replace('  ', ' ').replace('\u3000', ' ').replace('\xa0', ' ')
                    triplets.append({
                        "subject": sub,
                        "object": obj,
                        "relation": rel,
                        "other_objects": other_objects
                    })
                data_lst.append({
                    "text": text,
                    "triplets": triplets
                })
            loaded[k] = data_lst
    return loaded


def tokenize_re_dataset(data_dicts: List[Dict[str, Any]], lst_tokenizer):
    """

    :param data_dicts:
    :param lst_tokenizer:
    :return:
    """
    data_dict = tools.transpose_list_of_dict(data_dicts)
    tokenized = lst_tokenizer(data_dict['text'])
    data_dict.update(tools.transpose_list_of_dict(tokenized))
    # input_ids, token_type_ids, attention_mask, offset_mapping
    data_dicts = tools.transpose_dict_of_list(data_dict)
    return data_dicts


def add_char_span_re_dataset(data_dicts: List[Dict[str, Any]], find_all=False):
    """
    为关系抽取的数据的dict添加index信息。
    如果一个词在句子中多次出现且find_all=False，则选择第一次出现
    :param data_dicts: 符合下面格式的一个dict，
    {
    'text': 原句，
    'triplets': [
            {
            "subject": str,
            'object': str,
            'relation': str,
            'other_objects': {
                "type": str
            }
            }, ...
        ] or None
    }
    :param find_all: 一个词在句子中可能多次出现，如果find_all为False，则选择第一次出现的span，否则选择spanlist
    :return:
    """
    for d in data_dicts:
        text = d['text']
        triplets = d['triplets']
        for elem_triplet in triplets:
            sub_occur = tools.get_word_occurrences_in_sentence(text, elem_triplet['subject'])
            obj_occur = tools.get_word_occurrences_in_sentence(text, elem_triplet['object'])
            if find_all:
                elem_triplet['subject_occur'] = sub_occur
                elem_triplet['object_occur'] = obj_occur
            else:
                elem_triplet['subject_occur'] = sub_occur[0]
                elem_triplet['object_occur'] = obj_occur[0]
            if 'other_objects' in elem_triplet:
                for k, v in elem_triplet['other_objects']:
                    v_occur = tools.get_word_occurrences_in_sentence(text, v)
                    if find_all:
                        elem_triplet['other_objects'][k + '_occur'] = v_occur
                    else:
                        elem_triplet['other_objects'][k + '_occur'] = v_occur[0]
    return data_dicts


def add_token_span_re_dataset(data_dicts: List[Dict[str, Any]]):
    """

    :param data_dicts:
    需要先进行tokenize以及charSpan计算
    {
    'text': 原句，
    'input_ids': ,
    'token_type_ids': ,
    'attention_mask': ,
    'offset_mapping': ,
    'triplets': [
            {
            "subject": str,
            'subject_occur': ,
            'object': str,
            'object_occur': ,
            'relation': str,
            'other_objects': {
                "type": str
            }
            }, ...
        ] or None
    }
    :return:
    """
    for d in data_dicts:
        triplets = d['triplets']
        offset_mapping = d['offset_mapping']
        for elem_triplet in triplets:
            if isinstance(elem_triplet['subject_occur'], list):
                elem_triplet['subject_token_span'] = list(
                    tokenize_tools.charSpan_to_tokenSpan(x, offset_mapping) for x in elem_triplet['subject_occur'])
                elem_triplet['object_token_span'] = list(
                    tokenize_tools.charSpan_to_tokenSpan(x, offset_mapping) for x in elem_triplet['object_occur'])
            else:
                elem_triplet['subject_token_span'] = tokenize_tools.charSpan_to_tokenSpan(elem_triplet['subject_occur'],
                                                                                          offset_mapping)
                elem_triplet['object_token_span'] = tokenize_tools.charSpan_to_tokenSpan(elem_triplet['object_occur'],
                                                                                         offset_mapping)
    return data_dicts


def generate_key_labels(data_dicts: List[Dict[str, Any]], key='subject'):
    """
    根据span生成单个tensor
    比如span是(1, 3)，input_ids长度为6，则生成的是
    [0, 1, 0, 1, 0, 0]
    :param data_dicts:
    - tokenized
    - contains key tokenSpan
    :param key: {'subject', 'object'}
    :return:
    """
    for d in data_dicts:
        seq_l = len(d['input_ids'])
        subject_label = np.zeros(seq_l)
        subject_token_spans = list(x[key + '_token_span'] for x in d['triplets'])
        for elem_span in subject_token_spans:
            if isinstance(elem_span, list):
                for elem_elem_span in elem_span:
                    subject_label[elem_elem_span[0]] = 1
                    subject_label[elem_elem_span[1]] = 1
            else:
                subject_label[elem_span[0]] = 1
                subject_label[elem_span[1]] = 1
        d[key + '_label'] = subject_label
    return data_dicts


def generate_key_labels_separate(data_dicts: List[Dict[str, Any]], key='subject'):
    """
    与generate_key_labels相似，只不过span中的start与end要放在不同的两个array当中
    比如span是(1, 3)，input_ids长度为6，则生成的是
    start: [0, 1, 0, 0, 0, 0]
    end: [0, 0, 0, 1, 0, 0]
    :param data_dicts:
    :param key:
    :return:
    """
    for d in data_dicts:
        seq_l = len(d['input_ids'])
        subject_start_label = np.zeros(seq_l)
        subject_end_label = np.zeros(seq_l)
        subject_token_spans = list(x[key + '_token_span'] for x in d['triplets'])
        for elem_span in subject_token_spans:
            if isinstance(elem_span, list):
                for elem_elem_span in elem_span:
                    subject_start_label[elem_elem_span[0]] = 1
                    subject_end_label[elem_elem_span[1]] = 1
            else:
                subject_start_label[elem_span[0]] = 1
                subject_end_label[elem_span[1]] = 1
        d[key + '_start_label'] = subject_start_label
        d[key + '_end_label'] = subject_end_label
    return data_dicts


def generate_subject_labels(data_dicts: List[Dict[str, Any]], separate=True):
    if separate:
        return generate_key_labels_separate(data_dicts, key='subject')
    else:
        return generate_key_labels(data_dicts, key='subject')


def generate_object_labels(data_dicts: List[Dict[str, Any]], separate=True):
    if separate:
        return generate_key_labels_separate(data_dicts, key='object')
    else:
        return generate_key_labels(data_dicts, key='object')


def generate_pair_labels(data_dicts: List[Dict[str, Any]]):
    pass


def generate_relation_labels(data_dicts: List[Dict[str, Any]], relation2index: Dict[str, int]):
    pass


def rearrange_by_subject(data_dicts: List[Dict[str, Any]]):
    """
    按照subject重组
    :param data_dicts:
    :return:
    """
    new_data_dicts = []
    for d in data_dicts:
        triplets_dict, common_dict = tools.split_dict(d, ['triplets'])
        # triplets_dict只包含triplets，而common_dict包含text、input_ids等剩下的字段

        subject_dict = {}
        subject_related_dict = {}
        # subject_dict用于存放subject所对应的token_span等信息
        # 而subject_related_dict则用于存放对应的relation-object pair list
        for elem_triplet in triplets_dict['triplets']:
            subject = elem_triplet['subject']
            if subject in subject_dict:
                subject_related_dict[subject].append(tools.split_dict(elem_triplet, ['subject', 'subject_occur', 'subject_token_span'])[1])
            else:
                subject_part, relation_n_object_part = tools.split_dict(elem_triplet, ['subject', 'subject_occur', 'subject_token_span'])
                subject_dict[subject] = subject_part
                subject_related_dict[subject] = [relation_n_object_part]

        for subject, subject_value in subject_dict.items():
            relation_n_objects = subject_related_dict[subject]
            common_dict_copy = copy.deepcopy(common_dict)
            # text, input_ids, subject_label, ...
            common_dict_copy.update(subject_value)
            common_dict_copy['relation_object'] = relation_n_objects

            new_data_dicts.append(common_dict_copy)
    return new_data_dicts


def generate_relation_to_object_labels(data_dicts: List[Dict[str, Any]], relation_lst: StrList):
    """

    :param data_dicts:
    :param relation_lst:
    :return:
    """
    relation_cnt = len(relation_lst)
    relation_index = {i: x for (x, i) in enumerate(relation_lst)}
    for d in data_dicts:
        seq_l = len(d['input_ids'])
        relation_start_arrays, relation_end_arrays = np.zeros([relation_cnt, seq_l]), np.zeros([relation_cnt, seq_l])
        for elem_rel_obj in d['relation_object']:
            cur_relation = elem_rel_obj['relation']
            cur_object_span = elem_rel_obj['object_token_span']
            relation_idx = relation_index[cur_relation]
            relation_start_arrays[relation_idx][cur_object_span[0]] = 1
            relation_end_arrays[relation_idx][cur_object_span[1]] = 1
        d['relation_to_object_start_label'] = relation_start_arrays
        d['relation_to_object_end_label'] = relation_end_arrays
    return data_dicts


def generate_subject_gt_for_relation_object_pair_sample(data_dict: Dict[str, Any]):
    seq_l = len(data_dict['input_ids'])
    subject_start_gt, subject_end_gt = np.zeros(seq_l), np.zeros(seq_l)
    span = data_dict['subject_token_span']
    if isinstance(span, list):
        for elem_span in span:
            subject_end_gt[elem_span[1]] = 1
            subject_start_gt[elem_span[0]] = 1
    else:
        subject_end_gt[span[1]] = 1
        subject_start_gt[span[0]] = 1
    data_dict['subject_start_gt'] = subject_start_gt
    data_dict['subject_end_gt'] = subject_end_gt


def generate_subject_gt_for_relation_object_pair(data_dicts: List[Dict[str, Any]]):
    """

    :param data_dicts:
    :return:
    """
    for d in data_dicts:
        generate_subject_gt_for_relation_object_pair_sample(d)
    return d
