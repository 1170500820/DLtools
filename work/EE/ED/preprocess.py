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

# initial_dataset_path = '../../../data/NLP/EventExtraction/FewFC-main'
# dataset_type = 'FewFC'

initial_dataset_path = '../../../data/NLP/EventExtraction/duee'
dataset_type = 'Duee'


def remove_illegal_characters_p(data_dict: dict):
    content = data_dict['content']
    content = content.replace(' ', '_')
    data_dict['content'] = content
    return data_dict


def remove_illegal_characters(last_output_name: str, output_name: str, temp_path: str, dataset_type: str):
    data_dicts = load_jsonl(temp_path + last_output_name)

    results = []
    for elem in data_dicts:
        results.append(remove_illegal_characters_p(elem))

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


    logger.info(f'处理valid数据中')
    logger.info(f'[Step 1]正在去除过长的句子')
    data_filter(initial_dataset_path, dataset_type, temp_path, 'valid', f'valid.{dataset_type}.ED.filtered_length.jsonl')

    logger.info(f'[Step 2]正在去除空格以及非法字符')
    remove_illegal_characters(f'valid.{dataset_type}.ED.filtered_length.jsonl', f'valid.{dataset_type}.ED.removed_illegal.jsonl', temp_path=temp_path, dataset_type=dataset_type)


if __name__ == '__main__':
    pass