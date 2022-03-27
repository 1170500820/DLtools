import json
import pickle
import random
from tqdm import tqdm
from loguru import logger

from type_def import *
from dataset.ee_dataset import load_FewFC_ee, load_Duee_ee_formated
from utils import tools, tokenize_tools
from work.EE.EE_utils import data_filter, divide_by_event_type
from work.EE.EE_utils import split_by_content_type_trigger, remove_illegal_length
from work.EE.EE_utils import load_jsonl, dump_jsonl
from work.EE import EE_settings


# 包括存放中间处理结果的临时目录在内的一些信息
temp_path = 'temp_data/'
initial_dataset_path = '../../../data/NLP/EventExtraction/FewFC-main'
dataset_type = 'FewFC'

# 数据处理函数


def generate_input_and_label_p(data_dict: dict, dataset_type: str):
    """
    - input_ids
    - token_type_ids
    - attention_mask
    - offset_mapping
    - tokens
    - content
    - events
    - event_type
    - trigger_info
    - other_mentions
    :param data_dict:
    :return:
    """
    if dataset_type == 'FewFC':
        event_types = EE_settings.event_types_full
        role_types = EE_settings.role_types
    else:
        raise Exception(f'{dataset_type}数据集不存在！')
    event_type_idx = {x: i for (i, x) in enumerate(event_types)}
    role_type_ids = {x: i for (i, x) in enumerate(role_types)}


    trigger_info = data_dict['trigger_info']
    trigger_span = trigger_info['span']
    offset_mapping = data_dict['offset_mapping']
    events = data_dict['events']
    other_mentions = data_dict['other_mentions']

    # trigger gt
    trigger_token_span = tokenize_tools.charSpan_to_tokenSpan(trigger_span, offset_mapping)

    # trigger label
    start_labels = []  # list element: (event type idx, start index)
    end_labels = []  # list element: (event type idx, end index)
    for e_event in events:
        etype = e_event['type']
        etype_idx = event_type_idx[etype]
        cur_trigger_span = (-1, -1)
        for e_mention in e_event['mentions']:
            if e_mention['role'] == 'trigger':
                cur_trigger_span = e_mention['span']
        cur_trigger_token_span = tokenize_tools.charSpan_to_tokenSpan(cur_trigger_span, offset_mapping)
        start_labels.append((etype_idx, cur_trigger_token_span[0]))
        end_labels.append((etype_idx, cur_trigger_token_span[1]))

    # argument label for trigger gt
    arg_start_labels, arg_end_labels = [], []  # list element: (role type idx, start/end idx)
    for e_mention in other_mentions:
        cur_role_type = e_mention['role']
        cur_role_idx = role_type_ids[cur_role_type]
        cur_span = e_mention['span']
        cur_token_span = tokenize_tools.charSpan_to_tokenSpan(cur_span, offset_mapping)
        arg_start_labels.append((cur_role_idx, cur_token_span[0]))
        arg_end_labels.append((cur_role_idx, cur_token_span[1]))

    result = {
        'content': data_dict['content'],
        'input_ids': data_dict['input_ids'],
        'token_type_ids': data_dict['token_type_ids'],
        'attention_mask': data_dict['attention_mask'],
        'trigger_gt': trigger_token_span,
        'trigger_start_label': start_labels,
        'trigger_end_label': end_labels,
        'argument_start_label': arg_start_labels,
        'argument_end_label': arg_end_labels,
        'offset_mapping': offset_mapping
    }
    return result


def generate_gt_p(data_dict: dict, dataset_type: str):
    """
    只需要计算trigger gt即可
    :param data_dict:
    :param dataset_type:
    :return:
    """
    if dataset_type == 'FewFC':
        event_types = EE_settings.event_types_full
    else:
        raise Exception(f'{dataset_type}数据集不存在！')
    event_type_idx = {x: i for (i, x) in enumerate(event_types)}

    events = data_dict['events']
    offset_mapping = data_dict['offset_mapping']
    trigger_start_gts, trigger_end_gts = [], []  # list element: (event type idx, start/end idx)
    for e_event in events:
        mentions = e_event['mentions']
        e_type = e_event['type']
        e_type_idx = event_type_idx[e_type]
        for e_mention in mentions:
            if e_mention['role'] == 'trigger':
                cur_span = e_mention['span']
                cur_token_span = tokenize_tools.charSpan_to_tokenSpan(cur_span, offset_mapping)
                trigger_start_gts.append((e_type_idx, cur_token_span[0]))
                trigger_end_gts.append((e_type_idx, cur_token_span[1]))
    data_dict['trigger_start_gts'] = trigger_start_gts
    data_dict['trigger_end_gts'] = trigger_end_gts
    return data_dict



# 数据处理函数的调用函数


def tokenize_content(last_output_name: str, output_name: str, temp_path: str, dataset_type: str):
    data_dicts = load_jsonl(temp_path + last_output_name)

    # tokenize
    data_dict = tools.transpose_list_of_dict(data_dicts)
    lst_tokenizer = tokenize_tools.bert_tokenizer()
    content_result = lst_tokenizer(data_dict['content'])
    content_result = tools.transpose_list_of_dict(content_result)
    data_dict.update(content_result)
    data_dicts = tools.transpose_dict_of_list(data_dict)

    # dump_jsonl(data_dicts, temp_path + output_name)
    pickle.dump(data_dicts, open(temp_path + output_name, 'wb'))


def generate_input_and_label(last_output_name: str, output_name: str, temp_path: str, dataset_type: str):
    """

    data_dicts:
    - input_ids
    - token_type_ids
    - attention_mask
    - offset_mapping
    - tokens
    - content
    - events
    - event_type
    - trigger_info
    - other_mentions
    :param last_output_name:
    :param output_name:
    :param temp_path:
    :param dataset_type:
    :return:
    """
    data_dicts = pickle.load(open(temp_path + last_output_name, 'rb'))

    results = []
    for elem in data_dicts:
        results.append(generate_input_and_label_p(elem, dataset_type))

    pickle.dump(results, open(temp_path + output_name, 'wb'))


def generate_gt(last_output_name: str, output_name: str, temp_path: str, dataset_type: str):
    """
    只需要为trigger生成gt即可。
    :param last_output_name:
    :param output_name:
    :param temp_path:
    :param dataset_type:
    :return:
    """
    data_dicts = pickle.load(open(temp_path + last_output_name, 'rb'))

    results = []
    for elem in data_dicts:
        results.append(generate_gt_p(elem, dataset_type))

    pickle.dump(results, open(temp_path + output_name, 'wb'))


if __name__ == '__main__':
    logger.info(f'正在处理{dataset_type}数据')
    logger.info(f'数据源文件的存放路径: {initial_dataset_path}')

    # logger.info(f'处理train数据中')
    # logger.info(f'[Step 1]正在去除过长的句子')
    # data_filter(initial_dataset_path, dataset_type, temp_path, 'train', f'train.{dataset_type}.filtered_length.jsonl')
    #
    # logger.info(f'[Step 2]正在按事件类型和触发词拆分数据')
    # divide_by_event_type(f'train.{dataset_type}.filtered_length.jsonl', f'train.{dataset_type}.divided.jsonl', temp_path=temp_path, dataset_type=dataset_type)
    #
    # logger.info(f'[Step 3]正在tokenize')
    # tokenize_content(f'train.{dataset_type}.divided.jsonl', f'train.{dataset_type}.tokenized.pk', temp_path=temp_path, dataset_type=dataset_type)

    # logger.info(f'[Step 4]')
    # generate_input_and_label(f'train.{dataset_type}.tokenized.pk', f'train.{dataset_type}.labeled.pk', temp_path=temp_path, dataset_type=dataset_type)

    logger.info(f'处理valid数据中')
    # logger.info(f'[Step 1]正在去除过长的句子')
    # data_filter(initial_dataset_path, dataset_type, temp_path, 'valid', f'valid.{dataset_type}.filtered_length.jsonl')
    #
    # logger.info(f'[Step 2]正在tokenize')
    # tokenize_content(f'valid.{dataset_type}.filtered_length.jsonl', f'valid.{dataset_type}.tokenized.pk', temp_path, dataset_type)
    #
    logger.info(f'[Step 3]正在生成gt')
    generate_gt(f'valid.{dataset_type}.tokenized.pk', f'valid.{dataset_type}.gt.pk', temp_path, dataset_type)


