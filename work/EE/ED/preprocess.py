from type_def import *

import json
import pickle
from loguru import logger
from tqdm import tqdm

from utils import tools, tokenize_tools
from work.EE.EE_utils import data_filter, divide_by_event_type
from work.EE.EE_utils import load_jsonl, dump_jsonl
from work.EE.JointEE_rebuild import jointee_settings
from work.EE import EE_settings


temp_path = 'temp_data/'
pretrained_path =  'hfl/chinese-roberta-wwm-ext-large'

initial_dataset_path = '../../../data/NLP/EventExtraction/FewFC-main'
dataset_type = 'FewFC'

# initial_dataset_path = '../../../data/NLP/EventExtraction/duee'
# dataset_type = 'Duee'


def remove_illegal_characters_p(data_dict: dict):
    content = data_dict['content']
    content = content.replace(' ', '_')
    data_dict['content'] = content
    return data_dict


def generate_event_detection_label_p(data_dict: dict, dataset_type: str):
    if dataset_type == 'FewFC':
        event_types = EE_settings.event_types_full
    elif dataset_type == 'Duee':
        event_types = EE_settings.duee_event_types
    else:
        raise Exception(f'{dataset_type}数据集不存在！')
    event_type_idx = {x: i for (i, x) in enumerate(event_types)}

    data_dict['event_types_label'] = list(event_type_idx[x] for x in data_dict['event_types'])
    return data_dict


def tokenize_content(last_output_name: str, output_name: str, temp_path: str, dataset_type: str, plm_path: str = EE_settings.default_plm_path):
    data_dicts = load_jsonl(temp_path + last_output_name)

    # tokenize
    data_dict = tools.transpose_list_of_dict(data_dicts)
    lst_tokenizer = tokenize_tools.bert_tokenizer(plm_path=plm_path)
    content_result = lst_tokenizer(data_dict['content'])
    content_result = tools.transpose_list_of_dict(content_result)
    data_dict.update(content_result)
    data_dicts = tools.transpose_dict_of_list(data_dict)

    # dump_jsonl(data_dicts, temp_path + output_name)
    pickle.dump(data_dicts, open(temp_path + output_name, 'wb'))


def remove_illegal_characters(last_output_name: str, output_name: str, temp_path: str, dataset_type: str):
    data_dicts = load_jsonl(temp_path + last_output_name)

    results = []
    for elem in data_dicts:
        results.append(remove_illegal_characters_p(elem))

    dump_jsonl(results, temp_path + output_name)


def extract_event_types(last_output_name: str, output_name: str, temp_path: str, dataset_type: str):
    data_dicts = load_jsonl(temp_path + last_output_name)

    for elem in data_dicts:
        events = elem['events']
        event_types = []
        for elem_event in events:
            event_types.append(elem_event['type'])
        elem['event_types'] = event_types

    dump_jsonl(data_dicts, temp_path + output_name)


def generate_event_detection_label(last_output_name: str, output_name: str, temp_path: str, dataset_type: str):
    data_dicts = load_jsonl(temp_path + last_output_name)

    results = []
    for elem in data_dicts:
        results.append(generate_event_detection_label_p(elem, dataset_type))

    dump_jsonl(results, temp_path + output_name)


def event_detection_main():
    logger.info(f'正在处理{dataset_type}数据')
    logger.info(f'数据源文件的存放路径: {initial_dataset_path}')

    logger.info(f'处理train数据中')
    logger.info(f'[Step 1]正在去除过长的句子')
    data_filter(initial_dataset_path, dataset_type, temp_path, 'train', f'train.{dataset_type}.ED.filtered_length.jsonl')

    logger.info(f'[Step 2]正在去除空格以及非法字符')
    remove_illegal_characters(f'train.{dataset_type}.ED.filtered_length.jsonl',
                              f'train.{dataset_type}.ED.removed_illegal.jsonl', temp_path=temp_path,
                              dataset_type=dataset_type)

    logger.info(f'[Step 3]提取句子中所包含的事件')
    extract_event_types(f'train.{dataset_type}.ED.removed_illegal.jsonl',
                        f'train.{dataset_type}.ED.extracted_type.jsonl', temp_path=temp_path,
                        dataset_type=dataset_type)

    logger.info(f'[Step 4]为训练数据生成label')
    generate_event_detection_label(f'train.{dataset_type}.ED.extracted_type.jsonl',
                                   f'train.{dataset_type}.ED.labeled.jsonl', temp_path=temp_path,
                                   dataset_type=dataset_type)

    logger.info(f'[Step 5]tokenize')
    tokenize_content(f'train.{dataset_type}.ED.labeled.jsonl',
                     f'train.{dataset_type}.ED.RoBERTa.tokenized.pk', temp_path=temp_path,
                     dataset_type=dataset_type, plm_path=pretrained_path)


    logger.info(f'处理valid数据中')
    logger.info(f'[Step 1]正在去除过长的句子')
    data_filter(initial_dataset_path, dataset_type, temp_path, 'valid', f'valid.{dataset_type}.ED.filtered_length.jsonl')

    logger.info(f'[Step 2]正在去除空格以及非法字符')
    remove_illegal_characters(f'valid.{dataset_type}.ED.filtered_length.jsonl',
                              f'valid.{dataset_type}.ED.removed_illegal.jsonl', temp_path=temp_path,
                              dataset_type=dataset_type)

    logger.info(f'[Step 3]提取句子中所包含的事件')
    extract_event_types(f'valid.{dataset_type}.ED.removed_illegal.jsonl',
                        f'valid.{dataset_type}.ED.extracted_type.jsonl', temp_path=temp_path,
                        dataset_type=dataset_type)

    logger.info(f'[Step 4]tokenize')
    tokenize_content(f'valid.{dataset_type}.ED.extracted_type.jsonl',
                     f'valid.{dataset_type}.ED.RoBERTa.tokenized.pk', temp_path=temp_path,
                     dataset_type=dataset_type, plm_path=pretrained_path)


def event_detection_full():
    logger.info(f'正在处理{dataset_type}数据')
    logger.info(f'数据源文件的存放路径: {initial_dataset_path}')

    logger.info(f'处理train数据中')
    logger.info(f'[Step 1]正在去除过长的句子')
    data_filter(initial_dataset_path, dataset_type, temp_path, 'train', f'train.full.{dataset_type}.ED.filtered_length.jsonl', full=True)

    logger.info(f'[Step 2]正在去除空格以及非法字符')
    remove_illegal_characters(f'train.full.{dataset_type}.ED.filtered_length.jsonl',
                              f'train.full.{dataset_type}.ED.removed_illegal.jsonl', temp_path=temp_path,
                              dataset_type=dataset_type)

    logger.info(f'[Step 3]提取句子中所包含的事件')
    extract_event_types(f'train.full.{dataset_type}.ED.removed_illegal.jsonl',
                        f'train.full.{dataset_type}.ED.extracted_type.jsonl', temp_path=temp_path,
                        dataset_type=dataset_type)

    logger.info(f'[Step 4]为训练数据生成label')
    generate_event_detection_label(f'train.full.{dataset_type}.ED.extracted_type.jsonl',
                                   f'train.full.{dataset_type}.ED.labeled.jsonl', temp_path=temp_path,
                                   dataset_type=dataset_type)

    logger.info(f'[Step 5]tokenize')
    tokenize_content(f'train.full.{dataset_type}.ED.labeled.jsonl',
                     f'train.full.{dataset_type}.ED.RoBERTa.tokenized.pk', temp_path=temp_path,
                     dataset_type=dataset_type, plm_path=pretrained_path)


    logger.info(f'处理valid数据中')
    logger.info(f'[Step 1]正在去除过长的句子')
    data_filter(initial_dataset_path, dataset_type, temp_path, 'valid', f'valid.full.{dataset_type}.ED.filtered_length.jsonl', full=True)

    logger.info(f'[Step 2]正在去除空格以及非法字符')
    remove_illegal_characters(f'valid.full.{dataset_type}.ED.filtered_length.jsonl',
                              f'valid.full.{dataset_type}.ED.removed_illegal.jsonl', temp_path=temp_path,
                              dataset_type=dataset_type)

    logger.info(f'[Step 3]提取句子中所包含的事件')
    extract_event_types(f'valid.full.{dataset_type}.ED.removed_illegal.jsonl',
                        f'valid.full.{dataset_type}.ED.extracted_type.jsonl', temp_path=temp_path,
                        dataset_type=dataset_type)

    logger.info(f'[Step 4]tokenize')
    tokenize_content(f'valid.full.{dataset_type}.ED.extracted_type.jsonl',
                     f'valid.full.{dataset_type}.ED.RoBERTa.tokenized.pk', temp_path=temp_path,
                     dataset_type=dataset_type, plm_path=pretrained_path)


def event_detection_merge():
    logger.info(f'正在处理{dataset_type}数据')
    logger.info(f'数据源文件的存放路径: {initial_dataset_path}')

    logger.info(f'处理train数据中')
    logger.info(f'[Step 1]正在去除过长的句子')
    data_filter(initial_dataset_path, dataset_type, temp_path, 'train', f'train.merge.{dataset_type}.ED.filtered_length.jsonl', merge=True)

    logger.info(f'[Step 2]正在去除空格以及非法字符')
    remove_illegal_characters(f'train.merge.{dataset_type}.ED.filtered_length.jsonl',
                              f'train.merge.{dataset_type}.ED.removed_illegal.jsonl', temp_path=temp_path,
                              dataset_type=dataset_type)

    logger.info(f'[Step 3]提取句子中所包含的事件')
    extract_event_types(f'train.merge.{dataset_type}.ED.removed_illegal.jsonl',
                        f'train.merge.{dataset_type}.ED.extracted_type.jsonl', temp_path=temp_path,
                        dataset_type=dataset_type)

    logger.info(f'[Step 4]为训练数据生成label')
    generate_event_detection_label(f'train.merge.{dataset_type}.ED.extracted_type.jsonl',
                                   f'train.merge.{dataset_type}.ED.labeled.jsonl', temp_path=temp_path,
                                   dataset_type=dataset_type)

    logger.info(f'[Step 5]tokenize')
    tokenize_content(f'train.merge.{dataset_type}.ED.labeled.jsonl',
                     f'train.merge.{dataset_type}.ED.RoBERTa.tokenized.pk', temp_path=temp_path,
                     dataset_type=dataset_type, plm_path=pretrained_path)


    logger.info(f'处理valid数据中')
    logger.info(f'[Step 1]正在去除过长的句子')
    data_filter(initial_dataset_path, dataset_type, temp_path, 'valid', f'valid.merge.{dataset_type}.ED.filtered_length.jsonl', merge=True)

    logger.info(f'[Step 2]正在去除空格以及非法字符')
    remove_illegal_characters(f'valid.merge.{dataset_type}.ED.filtered_length.jsonl',
                              f'valid.merge.{dataset_type}.ED.removed_illegal.jsonl', temp_path=temp_path,
                              dataset_type=dataset_type)

    logger.info(f'[Step 3]提取句子中所包含的事件')
    extract_event_types(f'valid.merge.{dataset_type}.ED.removed_illegal.jsonl',
                        f'valid.merge.{dataset_type}.ED.extracted_type.jsonl', temp_path=temp_path,
                        dataset_type=dataset_type)

    logger.info(f'[Step 4]tokenize')
    tokenize_content(f'valid.merge.{dataset_type}.ED.extracted_type.jsonl',
                     f'valid.merge.{dataset_type}.ED.RoBERTa.tokenized.pk', temp_path=temp_path,
                     dataset_type=dataset_type, plm_path=pretrained_path)


def event_detection_merge2():
    peroid = 'merge6'

    logger.info(f'正在处理{dataset_type}数据')
    logger.info(f'数据源文件的存放路径: {initial_dataset_path}')

    logger.info(f'处理train数据中')
    logger.info(f'[Step 1]正在去除过长的句子')
    data_filter(initial_dataset_path, dataset_type, temp_path, 'train', f'train.{peroid}.{dataset_type}.ED.filtered_length.jsonl', merge2=True)

    logger.info(f'[Step 2]正在去除空格以及非法字符')
    remove_illegal_characters(f'train.{peroid}.{dataset_type}.ED.filtered_length.jsonl',
                              f'train.{peroid}.{dataset_type}.ED.removed_illegal.jsonl', temp_path=temp_path,
                              dataset_type=dataset_type)

    logger.info(f'[Step 3]提取句子中所包含的事件')
    extract_event_types(f'train.{peroid}.{dataset_type}.ED.removed_illegal.jsonl',
                        f'train.{peroid}.{dataset_type}.ED.extracted_type.jsonl', temp_path=temp_path,
                        dataset_type=dataset_type)

    logger.info(f'[Step 4]为训练数据生成label')
    generate_event_detection_label(f'train.{peroid}.{dataset_type}.ED.extracted_type.jsonl',
                                   f'train.{peroid}.{dataset_type}.ED.labeled.jsonl', temp_path=temp_path,
                                   dataset_type=dataset_type)

    logger.info(f'[Step 5]tokenize')
    tokenize_content(f'train.{peroid}.{dataset_type}.ED.labeled.jsonl',
                     f'train.{peroid}.{dataset_type}.ED.RoBERTa.tokenized.pk', temp_path=temp_path,
                     dataset_type=dataset_type, plm_path=pretrained_path)


    logger.info(f'处理valid数据中')
    logger.info(f'[Step 1]正在去除过长的句子')
    data_filter(initial_dataset_path, dataset_type, temp_path, 'valid', f'valid.{peroid}.{dataset_type}.ED.filtered_length.jsonl', merge2=True)

    logger.info(f'[Step 2]正在去除空格以及非法字符')
    remove_illegal_characters(f'valid.{peroid}.{dataset_type}.ED.filtered_length.jsonl',
                              f'valid.{peroid}.{dataset_type}.ED.removed_illegal.jsonl', temp_path=temp_path,
                              dataset_type=dataset_type)

    logger.info(f'[Step 3]提取句子中所包含的事件')
    extract_event_types(f'valid.{peroid}.{dataset_type}.ED.removed_illegal.jsonl',
                        f'valid.{peroid}.{dataset_type}.ED.extracted_type.jsonl', temp_path=temp_path,
                        dataset_type=dataset_type)

    logger.info(f'[Step 4]tokenize')
    tokenize_content(f'valid.{peroid}.{dataset_type}.ED.extracted_type.jsonl',
                     f'valid.{peroid}.{dataset_type}.ED.RoBERTa.tokenized.pk', temp_path=temp_path,
                     dataset_type=dataset_type, plm_path=pretrained_path)


if __name__ == '__main__':
    event_detection_merge2()
