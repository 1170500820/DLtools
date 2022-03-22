import json
import pickle
from tqdm import tqdm
import numpy as np
import copy

from type_def import *
from work.RE import RE_utils, RE_settings
from utils import tokenize_tools, tools
from loguru import logger

"""
基础定义部分
"""

# 存放中间数据的位置
temp_path = 'temp_data/'

# 源数据路径
initial_dataset_path = '../../../data/NLP/InformationExtraction/duie/'
# nyt: '../../data/NLP/InformationExtraction/NYT/generated/'
# webnlg: '../../data/NLP/InformationExtraction/WebNLG/generated/'

# 数据类型
dataset_type = 'duie'

# tokenizer所使用的配置
tokenier_plm = 'bert-base-chinese'

"""
对数据进行一些简单分析
"""


"""
数据处理的一些工具
"""

def naive_char_to_token_span(span: Tuple[int, int], offset_mapping: List[Tuple[int, int]]):
    """
    简易的转换函数

    """
    token2origin = {}
    text_length = -1
    for idx, elem in enumerate(offset_mapping):
        token2origin[idx] = list(range(elem[0], elem[1]))
        if elem[1] > text_length:
            text_length = elem[1]
    origin2token = {x: -1 for x in range(text_length)}
    for key, value in token2origin.items():
        for elem_index in value:
            origin2token[elem_index] = key
    if origin2token[0] == -1:
        origin2token[0] = 0
    for idx in range(text_length - 1):
        if origin2token[idx] == -1:
            origin2token[idx] = origin2token[idx -1]

    token_index1 = origin2token[span[0]]
    token_index2 = origin2token[span[1]]
    token_span = (token_index1, token_index2)
    return token_span


def load_jsonl(filename: str) -> List[dict]:
    """
    读取一个jsonl格式的文件。该文件的每一行都是一个合法的json

    :param filename 要读取的文件名
    """
    data = list(json.loads(x) for x in open(filename, 'r', encoding='utf-8').read().strip().split('\n'))
    return data


def dump_jsonl(data_dicts: List[dict], filename: str):
    """
    将dict的list输出为jsonl文件

    :param data_dicts: dict的list
    :param filename: 输出的文件名
    """
    f = open(filename, 'w', encoding='utf-8')
    for elem in data_dicts:
        f.write(json.dumps(elem, ensure_ascii=False) + '\n')
    f.close()


"""
处理数据的函数
"""


def add_char_span_re_dataset(data_dicts: List[Dict[str, Any]], find_all=False, ignore_other_objects=True):
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
            if 'other_objects' in elem_triplet and not ignore_other_objects:
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
    for d in tqdm(data_dicts):
        triplets = d['triplets']
        offset_mapping = d['offset_mapping']
        for elem_triplet in triplets:
            if isinstance(elem_triplet['subject_occur'], list):
                elem_triplet['subject_token_span'] = list(
                    naive_char_to_token_span(x, offset_mapping) for x in elem_triplet['subject_occur'])
                elem_triplet['object_token_span'] = list(
                    naive_char_to_token_span(x, offset_mapping) for x in elem_triplet['object_occur'])
            else:
                elem_triplet['subject_token_span'] = naive_char_to_token_span(elem_triplet['subject_occur'],
                                                                                          offset_mapping)
                elem_triplet['object_token_span'] = naive_char_to_token_span(elem_triplet['object_occur'],
                                                                                         offset_mapping)
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
        # subject_start_label = np.zeros(seq_l)
        # subject_end_label = np.zeros(seq_l)
        subject_token_spans = list(x[key + '_token_span'] for x in d['triplets'])
        # for elem_span in subject_token_spans:
        #     subject_start_label[elem_span[0]] = 1
        #     subject_end_label[elem_span[1]] = 1
        # d[key + '_start_label'] = subject_start_label
        # d[key + '_end_label'] = subject_end_label
        #
        d[key + '_start_label'] = {
            'seq_l': seq_l,
            'label_indexes': list(x[0] for x in subject_token_spans)
        }
        d[key + '_end_label'] = {
            'seq_l': seq_l,
            'label_indexes': list(x[1] for x in subject_token_spans)
        }

    return data_dicts


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
        # relation_start_arrays, relation_end_arrays = np.zeros([relation_cnt, seq_l]), np.zeros([relation_cnt, seq_l])
        start_label_per_relation = []
        end_label_per_relation = []
        for idx in range(relation_cnt):
            start_label_per_relation.append([])
            end_label_per_relation.append([])
        for elem_rel_obj in d['relation_object']:
            cur_relation = elem_rel_obj['relation']
            cur_object_span = elem_rel_obj['object_token_span']
            relation_idx = relation_index[cur_relation]
            # relation_start_arrays[relation_idx][cur_object_span[0]] = 1
            # relation_end_arrays[relation_idx][cur_object_span[1]] = 1
            start_label_per_relation[relation_idx].append(cur_object_span[0])
            end_label_per_relation[relation_idx].append(cur_object_span[1])
        # d['relation_to_object_start_label'] = relation_start_arrays
        # d['relation_to_object_end_label'] = relation_end_arrays
        d['relation_to_object_start_label'] = {
            'seq_l': seq_l,
            'relation_cnt': relation_cnt,
            'label_per_relation': start_label_per_relation
        }
        d['relation_to_object_end_label'] = {
            'seq_l': seq_l,
            'relation_cnt': relation_cnt,
            'label_per_relation': end_label_per_relation
        }
    return data_dicts


def generate_subject_gt_for_relation_object_pair_sample(data_dict: Dict[str, Any]):
    seq_l = len(data_dict['input_ids'])
    # subject_start_gt, subject_end_gt = np.zeros(seq_l), np.zeros(seq_l)
    span = data_dict['subject_token_span']

    # for elem_span in span:
    #     subject_end_gt[elem_span[1]] = 1
    #     subject_start_gt[elem_span[0]] = 1
    # data_dict['subject_start_gt'] = subject_start_gt
    # data_dict['subject_end_gt'] = subject_end_gt

    data_dict['subject_start_gt'] = {
        'seq_l': seq_l,
        'label_index': span[0]
    }
    data_dict['subject_end_gt'] = {
        'seq_l': seq_l,
        'label_index': span[1]
    }


def generate_eval_gt(data_dict: Dict[str, Any], relation_index: Dict[str, Any]):
    """
    为每一个句子生成eval阶段用于评测的gt

    casrel在评测阶段会把模型抽取出的结果统一转换为triplet span格式，也就是：
    (relation_idx, sub_start, sub_end, obj_start, obj_end)
    因此本函数也需要将数据中的triplet转换为该格式
    """
    results = []

    triplets = data_dict['triplets']
    for elem in triplets:
        subject_token = elem['subject_token_span']
        object_token = elem['object_token_span']
        relation_idx = relation_index[elem['relation']]
        results.append([relation_idx, subject_token[0], subject_token[1], object_token[0], object_token[1]])
    data_dict['eval_triplets'] = results



"""
对数据处理的包装函数
"""

def data_filter(input_data_filename: str, output_data_filename, dataset_type: str = dataset_type, subset_name: str = 'train'):
    """
    数据的格式如下
    {
        text: 文本,
        triplets: {
            subject: 主实体
            object: 客实体
            relation: 关系名字
        }
    }

    """
    if dataset_type == 'NYT':
        data = RE_utils.load_NYT_re(input_data_filename)
        data_dicts = data[subset_name]
    elif dataset_type == 'WebNLG':
        data = RE_utils.load_WebNLG_re(input_data_filename)
        data_dicts = data[subset_name]
    elif dataset_type == 'duie':
        data = RE_utils.load_duie_re(input_data_filename)
        data_dicts = data[subset_name]
    else:
        raise Exception(f'[dataset_factory]遇到未知的数据集！[{dataset_type}]')

    # 删除长度大于256的样本
    new_data_dicts = []
    for elem in data_dicts:
        if len(elem['text']) > 256:
            continue
        else:
            new_data_dicts.append(elem)

    dump_jsonl(new_data_dicts, temp_path + output_data_filename)



def tokenize_data(input_data_filename: str, output_data_filename, plm_path: str = tokenier_plm):
    """
    只需要对其中text进行tokenize即可
    """
    data_dicts = load_jsonl(temp_path + input_data_filename)

    lst_tokenizer = tokenize_tools.bert_tokenizer(plm_path=plm_path)
    data_dict = tools.transpose_list_of_dict(data_dicts)
    tokenized = lst_tokenizer(data_dict['text'])
    data_dict.update(tools.transpose_list_of_dict(tokenized))
    data_dicts = tools.transpose_dict_of_list(data_dict)

    dump_jsonl(data_dicts, temp_path + output_data_filename)


def get_span(input_data_filename: str, output_data_filename: str):
    """
    获取entity，也就是subject和object的span
    该步骤分为两步，第一步获取词语在text中出现的位置，也就是char_span。第二步将char_span通过tokenizer给出的mapping转化为token_span

    第一步当中，一个词可能出现多次，默认选取第一次出现
    """
    data_dicts = load_jsonl(temp_path + input_data_filename)

    results = add_char_span_re_dataset(data_dicts)
    results = add_token_span_re_dataset(results)

    dump_jsonl(results, temp_path + output_data_filename)



def get_subject_label(input_data_filename: str, output_data_filename: str):
    data_dicts = load_jsonl(temp_path + input_data_filename)

    data_dicts = generate_key_labels_separate(data_dicts, 'subject')

    # dump_jsonl(data_dicts, temp_path + output_data_filename)
    pickle.dump(data_dicts, open(temp_path + output_data_filename, 'wb'))


def reassemble(input_data_filename: str, output_data_filename: str):
    data_dicts = pickle.load(open(temp_path + input_data_filename, 'rb'))

    data_dicts = rearrange_by_subject(data_dicts)

    pickle.dump(data_dicts, open(temp_path + output_data_filename, 'wb'))


def get_object_relation_label(input_data_filename: str, output_data_filename: str, dataset_type: str):
    data_dicts = pickle.load(open(temp_path + input_data_filename, 'rb'))

    if dataset_type == 'duie':
        relation_list = RE_settings.duie_relations
    else:
        raise Exception(f'[get_object_relation_label]不存在的数据集：{dataset_type}')

    data_dicts = generate_relation_to_object_labels(data_dicts, relation_list)

    pickle.dump(data_dicts, open(temp_path + output_data_filename, 'wb'))


def get_train_gt(input_data_filename: str, output_data_filename: str):
    data_dicts = pickle.load(open(temp_path + input_data_filename, 'rb'))

    for elem in data_dicts:
        generate_subject_gt_for_relation_object_pair_sample(elem)

    pickle.dump(data_dicts, open(temp_path + output_data_filename, 'wb'))


def get_eval_gt(input_data_filename: str, output_data_filename: str, dataset_type: str):
    if dataset_type == 'duie':
        relations_index = RE_settings.duie_relations_idx
    else:
        raise Exception(f'[get_eval_gt]未知的数据集！{dataset_type}')
    data_dicts = load_jsonl(temp_path + input_data_filename)

    for elem in data_dicts:
        generate_eval_gt(elem, relations_index)

    pickle.dump(data_dicts, open(temp_path + output_data_filename, 'wb'))

"""
主函数入口
"""


def main():
    logger.info('数据预处理中')

    # 输出一下信息
    logger.info(f'数据存放路径:{temp_path}')
    logger.info(f'数据集读取路径:{initial_dataset_path}')
    logger.info(f'数据集类型:{dataset_type}')

    logger.info(f'正在生成训练数据')
    # logger.info(f'[Step 1]过滤')
    # data_filter(initial_dataset_path, f'train.{dataset_type}.filterd.jsonl', dataset_type, 'train')

    # logger.info(f'[Step 2]tokenize')
    # tokenize_data(f'train.{dataset_type}.filterd.jsonl', f'train.{dataset_type}.tokenized.jsonl', tokenier_plm)

    # logger.info(f'[Step 3]获取span')
    # get_span(f'train.{dataset_type}.tokenized.jsonl', f'train.{dataset_type}.span.jsonl')

    # logger.info(f'[Step 4]生成label')
    # get_subject_label(f'train.{dataset_type}.span.jsonl', f'train.{dataset_type}.labeled.pk')
    #
    # logger.info(f'[Step 5]重组数据')
    # reassemble(f'train.{dataset_type}.labeled.pk', f'train.{dataset_type}.rearranged.pk')

    # logger.info(f'[Step 6]为relation+object生成label')
    # get_object_relation_label(f'train.{dataset_type}.rearranged.pk', f'train.{dataset_type}.ro_labeled.pk', dataset_type)
    #
    # logger.info(f'[Step 7]生成gt')
    # get_train_gt(f'train.{dataset_type}.ro_labeled.pk', f'train.{dataset_type}.final.pk')

    logger.info(f'正在生成评测数据')
    # logger.info(f'[Step 1]过滤')
    # data_filter(initial_dataset_path, f'valid.{dataset_type}.filterd.jsonl', dataset_type, 'valid')

    # logger.info(f'[Step 2]tokenize')
    # tokenize_data(f'valid.{dataset_type}.filterd.jsonl', f'valid.{dataset_type}.tokenized.jsonl', tokenier_plm)

    # logger.info(f'[Step 3]获取span')
    # get_span(f'valid.{dataset_type}.tokenized.jsonl', f'valid.{dataset_type}.span.jsonl')

    logger.info(f'[Step 4]直接获取eval gt')
    get_eval_gt(f'valid.{dataset_type}.span.jsonl', f'valid.{dataset_type}.eval_final.pk', dataset_type)

if __name__ == '__main__':
    main()

# cache
# simple load/dump
#
