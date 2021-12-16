import copy

import numpy as np
import json
from tqdm import tqdm

import work.EE.EE_settings
from type_def import *
from work.RE import RE_settings
from utils import format_convert, tools, tokenize_tools
from utils.tokenize_tools import OffsetMapping
from utils.data import SimpleDataset


duee_dicts_sample = {
    "train": [
        {
            'text': '雀巢裁员4000人：时代抛弃你时，连招呼都不会打！',
            'id': '409389c96efe78d6af1c86e0450fd2d7',
            'event_list': [
                {
                    'event_type': '组织关系-裁员',
                    'trigger': '裁员',
                    'trigger_start_index': 2,
                    'arguments': [
                        {
                            'argument_start_index': 0, 'role': '裁员方', 'argument': '雀巢', 'alias': []
                        }, {
                            'argument_start_index': 4, 'role': '裁员人数', 'argument': '4000人', 'alias': []
                        }],
                    'class': '组织关系'
                }]
        },
        {
            'text': '美国“未来为”子公司大幅度裁员，这是为什么呢？任正非正式回应',
            'id': '5aec2b5b759c5f8f42f9c0156eb3c924',
            'event_list': [
                {
                    'event_type': '组织关系-裁员',
                    'trigger': '裁员',
                    'trigger_start_index': 13,
                    'arguments': [
                        {
                            'argument_start_index': 0, 'role': '裁员方', 'argument': '美国“未来为”子公司', 'alias': []
                        }],
                    'class': '组织关系'
                }]
        },
        {
            'text': '这一全球巨头“凉凉” “捅刀”华为后 裁员5000 现市值缩水800亿',
            'id': '82c4db0b0b209565485a1776b6f1b580',
            'event_list': [
                {
                    'event_type': '组织关系-裁员',
                    'trigger': '裁员',
                    'trigger_start_index': 19,
                    'arguments': [
                        {
                            'argument_start_index': 21, 'role': '裁员人数', 'argument': '5000', 'alias': []
                        }],
                    'class': '组织关系'}]
        }
    ]
}

fewfc_dicts_sample = {
    "train": [
        {
            'id': 'b8ca6cfbc85d11bced69be1e5e90021e',
            'content': '现中洲置业全资子公司惠州市筑品房地产开发有限公司拟收购中保盈隆所持有的目标公司70%股权。',
            'events': [
                {
                    'type': '收购',
                    'mentions': [
                        {
                            'word': '惠州市筑品房地产开发有限公司', 'span': [10, 24], 'role': 'sub-org'
                        }, {
                            'word': '收购', 'span': [25, 27], 'role': 'trigger'
                        }, {
                            'word': '70%', 'span': [39, 42], 'role': 'proportion'
                        }]
                }]
        },
        {
            'id': 'd257001948d1029d071680387bee28d6',
            'content': '公司判断因收购杭州优投科技有限公司及杭州多义乐网络科技有限公司存在大额计提商誉及无形资产减值准备的迹象,经测算,预计上述两家公司计提减值准备的金额预计在6亿元左右。',
            'events': [
                {
                    'type': '收购',
                    'mentions': [
                        {
                            'word': '收购', 'span': [5, 7], 'role': 'trigger'
                        }, {
                            'word': '杭州优投科技有限公司', 'span': [7, 17], 'role': 'obj-org'
                        }]
                }, {
                    'type': '收购',
                    'mentions': [
                        {
                            'word': '收购', 'span': [5, 7], 'role': 'trigger'
                        }, {
                            'word': '杭州多义乐网络科技有限公司', 'span': [18, 31], 'role': 'obj-org'
                        }]
                }]
        },
        {
            'id': 'c64bf55a2b36764adeb706f228e99f6a',
            'content': '2018年5月29日,恺英网络董事会会议又通过现金收购浙江九翎网络科技有限公司(下称:浙江九翎)部分股权的议案。',
            'events': [
                {
                    'type': '投资',
                    'mentions': [
                        {
                            'word': '收购', 'span': [25, 27], 'role': 'trigger'
                        }, {
                            'word': '恺英网络', 'span': [11, 15], 'role': 'sub'
                        }, {
                            'word': '浙江九翎网络科技有限公司', 'span': [27, 39], 'role': 'obj'
                        }, {
                            'word': '2018年5月29日', 'span': [0, 10], 'role': 'date'
                        }]
                }, {
                    'type': '股份股权转让', 'mentions': [
                        {
                            'word': '收购', 'span': [25, 27], 'role': 'trigger'
                        }, {
                            'word': '恺英网络', 'span': [11, 15], 'role': 'obj-org'
                        }, {
                            'word': '股权', 'span': [50, 52], 'role': 'collateral'
                        }, {
                            'word': '2018年5月29日', 'span': [0, 10], 'role': 'date'
                        }, {
                            'word': '浙江九翎网络科技有限公司', 'span': [27, 39], 'role': 'target-company'
                        }]
                }]
        }]
}

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


def add_event_types_to_ee_dataset_sample(data_dict: Dict[str, Any]):
    # 获取所有事件类型
    event_types = list(x['type'] for x in data_dict['events'])
    # 去重
    event_types = list(set(event_types))
    # 写入dict
    data_dict['event_types'] = event_types


def add_event_types_to_ee_dataset(data_dicts: List[Dict[str, Any]]):
    for d in data_dicts:
        add_event_types_to_ee_dataset_sample(d)
    return data_dicts


def add_event_type_label_to_ee_dataset_sample(data_dict: Dict[str, Any], type_index: Dict[str, Any]):
    # 获取所有事件类型
    event_types = list(x['type'] for x in data_dict['events'])
    # 去重
    event_types = list(set(event_types))
    # 创建label nparray
    type_cnt = len(type_index)
    label = np.zeros(type_cnt)
    for elem_event_type in event_types:
        label[type_index[elem_event_type]] = 1
    # 写入dict
    data_dict['label'] = label


def add_event_type_label_to_ee_dataset(data_dicts: List[Dict[str, Any]], type_index: Dict[str, Any]):
    for d in tqdm(data_dicts):
        add_event_type_label_to_ee_dataset_sample(d, type_index)
    return data_dicts



def building_event_detection_dataset(data_dir: str, data_type: str, tokenizer_path: str):
    if data_type == 'duee':
        data_dicts = load_Duee_ee(data_dir)
        data_dicts = convert_Duee_to_FewFC_format(data_dicts)
    elif data_type == 'fewfc':
        data_dicts = load_FewFC_ee(data_dir)
    else:
        raise Exception(f'[building_event_detection_train_dataset]没有{data_type}类型的数据集！')

    tokenizer = tokenize_tools.bert_tokenizer(tokenizer_path)
    train_data_dicts, valid_data_dicts = data_dicts['train'], data_dicts['valid']

    # 先生成train
    train_data_dicts = preprocess_filter(train_data_dicts)
    train_data_dicts = tokenize_ee_dataset(train_data_dicts, tokenizer)
    train_data_dicts = add_event_type_label_to_ee_dataset(train_data_dicts)

    # 生成valid
    valid_data_dicts = tokenize_ee_dataset(valid_data_dicts, tokenizer)
    valid_data_dicts = add_event_types_to_ee_dataset(valid_data_dicts)

    return {
        "train": train_data_dicts,
        "valid": valid_data_dicts
    }


if __name__ == '__main__':
    duee_dicts = load_Duee_ee('../data/NLP/EventExtraction/duee/')
    fewfc_dicts = load_FewFC_ee('../data/NLP/EventExtraction/FewFC-main')
    converted_duee = convert_Duee_to_FewFC_format(duee_dicts)
    train_duee, valid_duee = converted_duee['train'], converted_duee['valid']
    train_duee = preprocess_filter(train_duee)

    tokenizer = tokenize_tools.bert_tokenizer()
    train_duee = tokenize_ee_dataset(train_duee, tokenizer)
    valid_duee = tokenize_ee_dataset(valid_duee, tokenizer)
    duee = add_token_span_to_ee_dataset(train_duee)

    print('generating labels')
    train_duee = add_event_type_label_to_ee_dataset(train_duee, work.EE.EE_settings.duee_event_index)
    valid_duee = add_event_types_to_ee_dataset(valid_duee)


