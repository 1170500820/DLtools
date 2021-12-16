import numpy as np
import json

from type_def import *
from work.RE import RE_settings
from utils import format_convert, tools, tokenize_tools
from utils.tokenize_tools import OffsetMapping


Triplet = Tuple[int, int, int, int, int]  # hashable


# 用于将模型预测转换为index和字符串的两个函数

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


def convert_lists_to_words_casrel(sentence: str, offset_mapping: OffsetMapping, relations: List[str], subjects: SpanList, objects_with_relation: List[List[SpanList]]) -> Tuple[str, str, str]:
    """
    将CASREL框架下的subjects与objects list转换为词本体
    (relation, subject, object)
    :param sentence:
    :param offset_mapping:
    :param subjects:
    :param objects_with_relation:
    :return:
    """
    triplets = convert_lists_to_triplet_casrel(subjects, objects_with_relation)
    words = []
    for elem_triplet in triplets:
        rel = relations[elem_triplet[0]]
        subject_span = tokenize_tools.tokenSpan_to_charSpan(elem_triplet[1: 3], offset_mapping)
        object_span = tokenize_tools.tokenSpan_to_charSpan(elem_triplet[3: 5], offset_mapping)
        words.append((rel, subject_span, object_span))
    return words


# 专用加载数据集的三个函数，分别加载NYT，WebNLG，duie


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





def add_index_to_re_data(d: dict, find_all=False) -> dict:
    """
    为关系抽取的数据的dict添加index信息。
    如果一个词在句子中多次出现且find_all=False，则选择第一次出现
    :param d: 符合下面格式的一个dict，
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
    :param find_all: 若为True，则position是一个list，保存所有存在
    :return:
    """
    text = d['text']
    triplets = d['triplets']
    for elem_triplet in triplets:
        sub_occur = get_word_occurrences_in_sentence(text, elem_triplet['subject'])
        obj_occur = get_word_occurrences_in_sentence(text, elem_triplet['object'])
        if find_all:
            elem_triplet['subject_occur'] = sub_occur
            elem_triplet['object_occur'] = obj_occur
        else:
            elem_triplet['subject_occur'] = sub_occur[0]
            elem_triplet['object_occur'] = obj_occur[0]
        if 'other_objects' in elem_triplet:
            for k, v in elem_triplet['other_objects']:
                v_occur = get_word_occurrences_in_sentence(text, v)
                if find_all:
                    elem_triplet['other_objects'][k + '_occur'] = v_occur
                else:
                    elem_triplet['other_objects'][k + '_occur'] = v_occur[0]
    return d


def add_token_span_to_re_data(d: dict, offset_mapping: OffsetMapping) -> dict:
    """
    将d中的char index都转换为token span
    :param d:
    :param offset_mapping:
    :return:
    """
    triplets = d['triplets']
    for elem_triplet in triplets:
        if isinstance(elem_triplet['subject_occur'], list):
            elem_triplet['subject_token_span'] = list(tokenize_tools.charSpan_to_tokenSpan(x, offset_mapping) for x in elem_triplet['subject_occur'])
            elem_triplet['object_token_span'] = list(tokenize_tools.charSpan_to_tokenSpan(x, offset_mapping) for x in elem_triplet['object_occur'])
        else:
            elem_triplet['subject_token_span'] = tokenize_tools.charSpan_to_tokenSpan(elem_triplet['subject_occur'], offset_mapping)
            elem_triplet['object_token_span'] = tokenize_tools.charSpan_to_tokenSpan(elem_triplet['object_occur'], offset_mapping)
    return d


def count_multi_occurrences(data_type: str, data_dir: str):
    """
    用于统计重复出现的一个小函数
    :param data_type: NYT, WebNLG, duie
    :param data_dir:
    :return:
    """
    if data_type == 'NYT':
        d = load_NYT_re(data_dir)
    elif data_type == 'WebNLG':
        d = load_WebNLG_re(data_dir)
    elif data_type == 'duie':
        d = load_duie_re(data_dir)
    else:
        raise Exception
    for key, value in d.items():
        if data_type == 'duie' and key == 'test':
            continue
        print(f'{data_type}--{key}')
        entity, entity_multi = 0, 0
        for triplet_dict in value:
            text = triplet_dict['text']
            relations = triplet_dict['triplets']
            for elem_rel in relations:
                entity += 2
                if len(get_word_occurrences_in_sentence(text, elem_rel['subject'])) > 1:
                    entity_multi += 1
                if len(get_word_occurrences_in_sentence(text, elem_rel['object'])) > 1:
                    entity_multi += 1
        print(f'total entity: {entity}\nentity with multi occur: {entity_multi}\nratio: {entity_multi / entity if entity != 0 else 0}')
        print('_' * 20)


if __name__ == '__main__':
    d = load_NYT_re('../../data/NLP/InformationExtraction/NYT/generated/')
    # count_multi_occurrences('duie', '../../data/NLP/InformationExtraction/duie/')
    # count_multi_occurrences('NYT', '../../data/NLP/InformationExtraction/NYT/generated/')
    # count_multi_occurrences('WebNLG', '../../data/NLP/InformationExtraction/WebNLG/generated')

