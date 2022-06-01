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


def merge_arguments_with_same_trigger_p(data_dict: dict):
    sentence = data_dict['content']
    sid = data_dict['id']
    events = data_dict['events']
    results = []

    # 合并触发词相同的论元，然后存放在event_dict里面
    event_dict = {}
    for elem in events:
        e_type = elem['type']
        e_mentions = elem['mentions']
        trigger_span, trigger_word = (0, 0), ''
        new_mentions = []
        for e_mention in e_mentions:
            if e_mention['role'] == 'trigger':
                trigger_span = tuple(e_mention['span'])
                trigger_word = e_mention['word']
            else:
                new_mentions.append(e_mention)

        type_trigger = (e_type, trigger_span, trigger_word)
        if type_trigger not in event_dict:
            event_dict[type_trigger] = list(map(lambda x: (x['word'], tuple(x['span']), x['role']), new_mentions))
        else:
            new_mentions = list(map(lambda x: (x['word'], tuple(x['span']), x['role']), new_mentions))
            old_mentions = event_dict[type_trigger]
            merged_mentions = old_mentions + new_mentions
            merged_mentions = list(set(merged_mentions))
            event_dict[type_trigger] = merged_mentions

    # 然后重新转换会原来的形式
    for key, value in event_dict.items():
        e_type, trigger_span, trigger_word = key
        mentions = list(map(lambda x: ({
            "word": x[0],
            'span': x[1],
            'role': x[2]
        }), value))
        results.append({
            'id': sid,
            'content': sentence,
            'event_type': e_type,
            'trigger_span': trigger_span,
            'trigger_word': trigger_word,
            'other_mentions': mentions
        })

    return results


def tokenize_content(last_output_name: str, output_name: str, temp_path: str, dataset_type: str, plm_path: str = jointee_settings.plm_path):
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


def generate_label_for_trigger_and_argument_p(data_dict, dataset_type: str):
    """
    trigger label (bsz, seq_l, 1)
    argument label (bsz, seq_l, role_cnt)

    :param data_dict:
    :return:
    """
    if dataset_type == 'FewFC':
        event_types = EE_settings.event_types_full
        role_types = EE_settings.role_types
    elif dataset_type == 'Duee':
        event_types = EE_settings.duee_event_types
        role_types = EE_settings.duee_role_types
    else:
        raise Exception(f'{dataset_type}数据集不存在！')
    event_type_idx = {x: i for (i, x) in enumerate(event_types)}
    role_type_idx = {x: i for (i, x) in enumerate(role_types)}

    offset_mapping = data_dict['offset_mapping']
    # 该span为左闭右开，而预测时改为左闭右闭更加合理
    trigger_span = data_dict['trigger_span']

    event_type = data_dict['event_type']

    other_mentions = data_dict['other_mentions']

    if trigger_span[0] == trigger_span[1]:
        real_span = trigger_span
    else:
        real_span = (trigger_span[0], trigger_span[1] - 1)

    trigger_token_span = tokenize_tools.charSpan_to_tokenSpan(real_span, offset_mapping)

    arguments = []  # List of (role_idx, span0, span1)
    for elem_arg in other_mentions:
        arg_span, arg_role = elem_arg['span'], elem_arg['role']
        if arg_span[0] == arg_span[1]:
            real_arg_span = arg_span
        else:
            real_arg_span = (arg_span[0], arg_span[1] - 1)
        arg_token_span = tokenize_tools.charSpan_to_tokenSpan(real_arg_span, offset_mapping)
        arg_index = role_type_idx[arg_role]
        arguments.append((arg_index, arg_token_span))

    data_dict['trigger_token_span'] = trigger_token_span
    data_dict['argument_token_spans'] = arguments

    return data_dict


def merge_arguments_with_same_trigger(last_output_name: str, output_name: str, temp_path: str, dataset_type: str):
    data_dicts = load_jsonl(temp_path + last_output_name)

    results = []
    for elem in data_dicts:
        results.extend(merge_arguments_with_same_trigger_p(elem))

    dump_jsonl(results, temp_path + output_name)


def generate_label_for_trigger_and_argument(last_output_name: str, output_name: str, temp_path: str, dataset_type: str):
    data_dicts = pickle.load(open(temp_path + last_output_name, 'rb'))

    results = []
    for elem in data_dicts:
        results.append(generate_label_for_trigger_and_argument_p(elem, dataset_type))

    pickle.dump(results, open(temp_path + output_name, 'wb'))


# 完整的数据生成函数

def JointEE_main():
    logger.info(f'正在处理{dataset_type}数据')
    logger.info(f'数据源文件的存放路径: {initial_dataset_path}')

    logger.info(f'处理train数据中')
    logger.info(f'[Step 1]正在去除过长的句子')
    # data_filter(initial_dataset_path, dataset_type, temp_path, 'train', f'train.{dataset_type}.filtered_length.jsonl')

    logger.info(f'[Step 2]正在合并触发词相同的论元')
    # merge_arguments_with_same_trigger(f'train.{dataset_type}.filtered_length.jsonl', f'train.{dataset_type}.merged_arguments.jsonl', temp_path=temp_path, dataset_type=dataset_type)

    logger.info(f'[Step 3]正在tokenize')
    tokenize_content(f'train.{dataset_type}.merged_arguments.jsonl', f'train.{dataset_type}.tokenized.pk', temp_path=temp_path, dataset_type=dataset_type)

    logger.info(f'[Step 4]为trigger与argument生成label')
    generate_label_for_trigger_and_argument(f'train.{dataset_type}.tokenized.pk', f'train.{dataset_type}.labeled.pk', temp_path=temp_path, dataset_type=dataset_type)

    logger.info(f'处理valid数据中')
    logger.info(f'[Step 1]正在去除过长的句子')
    data_filter(initial_dataset_path, dataset_type, temp_path, 'valid', f'valid.{dataset_type}.filtered_length.jsonl')

    logger.info(f'[Step 2]正在tokenize')
    tokenize_content(f'valid.{dataset_type}.filtered_length.jsonl', f'valid.{dataset_type}.tokenized.pk', temp_path=temp_path, dataset_type=dataset_type)



if __name__ == '__main__':
    JointEE_main()
