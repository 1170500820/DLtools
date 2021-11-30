import numpy as np
import json

from type_def import *
from work.RE import RE_settings
from utils import format_convert, tools


Triplet = Tuple[int, int, int, int, int]  # hashable


def convert_lists_to_triplet_casrel(subjects: SpanList, objects_with_relation: List[List[SpanList]]) -> List[Triplet]:
    """
    将CASREL框架下的subjects与objects list转化为由与具体标注无关的index表示的triplet格式(hashable)
    转化之后既可以通过index与标注的对应关系转换为标注结果，也可以直接用f1计算函数计算。
    因为计算函数是不需要具体的标注信息的
    :param subjects:
    :param objects_with_relation:
    :return: list of triplets
        Triplet: (relation_idx, sub_start, sub_end, obj_start, obj_end)
    """
    triplets = []  # List[Tuple[int, int, int, int, int]] a.k.a List[Triplet]
    for (subject, object_w_rel) in zip(subjects, objects_with_relation):
        # subject -> Span
        # object_w_rel -> List[SpanList]
        sub_start, sub_end = subject
        for rel_idx, elem_objs in enumerate(object_w_rel):
            # elem_objs -> SpanList
            for elem_obj in elem_objs:
                # elem_obj -> Span
                obj_start, obj_end = elem_obj
                triplets.append((rel_idx, sub_start, sub_end, obj_start, obj_end))
    return triplets


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
        'test_epo': 'new_test_epo.json',
        'test_seo': 'new_test_seo.json',
        'test_normal': 'new_test_normal.json',
        'train': 'new_train.json',
        'train_epo': 'new_train_epo.json',
        'train_seo': 'new_train_seo.json',
        'train_normal': 'new_train_normal.json',
        'valid': 'new_valid.json',
        'valid_epo': 'new_valid_epo.json',
        'valid_seo': 'new_valid_seo.json',
        'valid_normal': 'new_valid_normal.json',
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
                "triplets": relations
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
        'train_epo': 'new_train_epo.json',
        'train_seo': 'new_train_seo.json',
        'train_normal': 'new_train_normal.json',
        'valid': 'new_valid.json',
        'valid_epo': 'new_valid_epo.json',
        'valid_seo': 'new_valid_seo.json',
        'valid_normal': 'new_valid_normal.json',
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
                "triplets": relations
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
        "test": "duit_test2.json/duie_test2.json"
    }
    loaded = {}
    for k, v in filenames.items():
        data_lst = []
        dicts = list(map(json.loads, open(file_dir + v, 'r').read().strip().split('\n')))
        if k == 'test':
            for elem in dicts:
                text = elem['text']
                data_lst.append({
                    "text": text,
                    "triplets": None
                })
        else:
            for elem in dicts:
                triplets = []  # 当前句子所对应的所有triplet
                text = elem['text']
                relations = elem['spo_list']
                for elem_rel in relations:
                    sub, obj, rel = elem_rel['subject'], elem_rel['object']['@value'], elem_rel['predicate']
                    _, other_objects = tools.split_dict(elem_rel['object'], ['@value'], keep_origin=True)
                    triplets.append({
                        "subject": sub,
                        "object": obj,
                        "relation": rel,
                        "other_objects": other_objects
                    })
                data_lst.append({
                    "text": text,
                    "triplets": relations
                })
            loaded[k] = data_lst
        return data_lst

