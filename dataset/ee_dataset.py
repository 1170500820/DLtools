import copy

import numpy as np
import json

from type_def import *
from work.RE import RE_settings
from utils import format_convert, tools, tokenize_tools
from utils.tokenize_tools import OffsetMapping


def load_FewFC_ee(file_dir: str, splitted=True):
    if file_dir[-1] != '/':
        file_dir += '/'
    if splitted:
        test_file = 'test.json'
        train_file = 'train.json'
        val_file = 'val.json'
        test_data = list(json.loads(x) for x in open(file_dir + test_file, 'r').read().strip().split('\n'))
        train_data = list(json.loads(x) for x in open(file_dir + train_file, 'r').read().strip().split('\n'))
        val_data = list(json.loads(x) for x in open(file_dir + val_file, 'r').read().strip().split('\n'))
        return {
            "train": train_data,
            "test": test_data,
            "valid": val_data
        }
    else:
        raise NotImplementedError


def load_Duee_ee(file_dir: str):
    if file_dir[-1] != '/':
        file_dir += '/'
    test_file = 'duee_test2.json/duee_test2.json'
    train_file = 'duee_train.json/duee_train.json'
    valid_file = 'duee_dev.json/duee_dev.json'
    test_data = list(json.loads(x) for x in open(file_dir + test_file, 'r').read().strip().split('\n'))
    train_data = list(json.loads(x) for x in open(file_dir + train_file, 'r').read().strip().split('\n'))
    val_data = list(json.loads(x) for x in open(file_dir + valid_file, 'r').read().strip().split('\n'))
    return {
        "train": train_data,
        "test": test_data,
        "valid": val_data
    }


def convert_Duee_to_FewFC_format_sample(data_dict: Dict[str, Any]):
    """
    将一个Duee的sample的key值转化为FewFC格式，对于FewFC中不包含的key值则直接忽略
    :param data_dict:
    :return:
    """
    new_dict = {}
    new_dict['content'] = data_dict['text']
    new_dict['id'] = data_dict['id']
    if 'event_list' not in data_dict:
        return new_dict
    events = []
    for elem_event in data_dict['event_list']:
        # 初始化一个dict，作为FewFC格式的一个event
        # FewFC格式的一个event dict，包括type与mentions
        new_event = {}

        # 装载 type
        new_event['type'] = elem_event['event_type']

        # 装载 mentions
        # 先构造trigger，然后放入mentions列表
        trigger_span = [elem_event['trigger_start_index'], elem_event['trigger_start_index'] + len(elem_event['trigger'])]
        new_mentions = [{
            "word": elem_event['trigger'],
            "span": trigger_span,
            "role": 'trigger'
        }]
        # 接下来构造每一个argument，然后放入mentions列表
        for elem_arg in elem_event['arguments']:
            arg = elem_arg['argument']
            role = elem_arg['role']
            span = [elem_arg['argument_start_index'], elem_arg['argument_start_index'] + len(arg)]
            new_mentions.append({
                "word": arg,
                'span': span,
                'role': role,
            })
        new_event['mentions'] = new_mentions
        events.append(new_event)
    new_dict['events'] = events
    return new_dict


def convert_Duee_to_FewFC_format(duee_dicts: Dict[str, List[Dict[str, Any]]]):
    """
    Duee的每个dict的key与FewFC不同，这里直接改成FewFC的形式
    对于FewFC中不包含的key，直接忽略
    :param data_dicts:
    :return:
    """
    new_duee_dicts = {}
    for elem_dataset_type in ['train', 'valid', 'test']:
        new_data_dicts = []
        for elem_dict in duee_dicts[elem_dataset_type]:
            new_data_dicts.append(convert_Duee_to_FewFC_format_sample(elem_dict))
        new_duee_dicts[elem_dataset_type] = new_data_dicts
    return new_duee_dicts


def preprocess_modify_sample(data_dict: Dict[str, Any]):
    pass


def preprocess_filter_sample(data_dict: Dict[str, Any]):
    return remove_illegal_length(data_dict)


def remove_illegal_length(data_dict: Dict[str, Any]):
    if len(data_dict['content']) >= 254:
        return []
    else:
        return [data_dict]


def preprocess_filter(data_dicts: List[Dict[str, Any]]):
    """
    对训练数据进行过滤，过滤掉其中过长的句子。
    :param data_dicts:
    :return:
    """
    result = []
    for d in data_dicts:
        result += preprocess_filter_sample(d)
    return result


def tokenize_ee_dataset(data_dicts: List[Dict[str, Any]], lst_tokenizer):
    """
    FewFC格式的tokenize
    FewFC格式下句子是content，别的格式的key值可能不太一样，不过好改
    :param data_dicts:
    :param lst_tokenizer:
    :return:
    """
    data_dict = tools.transpose_list_of_dict(data_dicts)
    tokenized = lst_tokenizer(data_dict['content'])
    data_dict.update(tools.transpose_list_of_dict(tokenized))
    # input_ids, token_type_ids, attention_mask, offset_mapping
    data_dicts = tools.transpose_dict_of_list(data_dict)
    return data_dicts


def add_token_span_to_ee_dataset_sample(data_dict: Dict[str, Any]):
    """
    根据offset_mapping，由charSpan计算tokenSpan
    首先FewFC的span格式并不是标准的span：
    FewFC的span是该词的第一个字的index与最后一个字的后一个字的index。而标准的span分别是第一个字和最后一个字的index
    所以计算时需要先对end index减一
    :param data_dict:
    {
        'content': ,
        "id": ,
        "input_ids": ,
        ... ,
        "offset_mapping": ,
        "events": [
            {
               "type": ,
               "mentions": [
                    {
                        "span": ,
                        ...
                    }
               ]
            },
            ...
        ]
    }
    :return:
    """
    offset_mapping = data_dict['offset_mapping']
    for elem_event in data_dict['events']:
        for elem_mention in elem_event['mentions']:
            real_span = (elem_mention['span'][0], elem_mention['span'][1] - 1)
            token_span = tokenize_tools.charSpan_to_tokenSpan(real_span, offset_mapping)
            elem_mention['token_span'] = token_span


def add_token_span_to_ee_dataset(data_dicts: List[Dict[str, Any]]):
    for d in data_dicts:
        add_token_span_to_ee_dataset_sample(d)
    return data_dicts


if __name__ == '__main__':
    duee_dicts = load_Duee_ee('../data/NLP/EventExtraction/duee/')
    fewfc_dicts = load_FewFC_ee('../data/NLP/EventExtraction/FewFC-main')
    converted_duee = convert_Duee_to_FewFC_format(duee_dicts)
    preprocessed_duee = preprocess_filter(converted_duee['train'])

    tokenizer = tokenize_tools.bert_tokenizer()
    tokenized_duee = tokenize_ee_dataset(preprocessed_duee, tokenizer)
    duee = add_token_span_to_ee_dataset(tokenized_duee)
