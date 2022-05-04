"""
部分数据处理在这里完成，生成各种中间数据
"""
import json
import pickle

import jieba
import random
import stanza
from tqdm import tqdm
import numpy as np

from type_def import *
from dataset.ee_dataset import load_FewFC_ee, load_Duee_ee_formated
from utils import tools, tokenize_tools
from work.EE.EE_utils import remove_illegal_length
from work.EE.EE_utils import split_by_content_type_trigger
from work.EE import EE_settings

# 中间处理结果将会被输入到临时目录当中
temp_path = 'temp_data/'
initial_dataset_path = '../../../data/NLP/EventExtraction/FewFC-main'
dataset_type = 'FewFC'


def split_by_content_type_trigger(data_dict: Dict[str, Any]):
    content = data_dict['content']
    events = data_dict['events']

    result_dicts = []

    for elem_event in events:
        event_type = elem_event['type']
        event_mentions = elem_event['mentions']

        # 分离trigger与argument
        triggers = []
        other_mentions = []
        for elem_mention in event_mentions:
            if elem_mention['role'] == 'trigger':
                triggers.append(elem_mention)
            else:
                other_mentions.append(elem_mention)
        if len(triggers) != 1:
            raise Exception(f'[split_by_content_type_trigger]不合法的mentions！包含的trigger个数错误。应当为1，实际为{len(triggers)}')

        cur_sample = {
            "content": content,
            "events": events,
            "event_type": event_type,
            "trigger_info": triggers[0],
            "other_mentions": other_mentions
        }
        result_dicts.append(cur_sample)

    return result_dicts


def construct_EAR_ERR_context(data_dict: Dict[str, Any], dataset_type: str, stanza_nlp):
    """
    同时构建EAR,ERR与context

    :param data_dict: [content, event_type, trigger_info, other_mentions]
    :param dataset_type:
    :param stanza_nlp:
    :return:
    """
    content, event_type, trigger_info, mentions = \
        data_dict['content'], data_dict['event_type'], data_dict['trigger_info'], data_dict['other_mentions']

    trigger_span = trigger_info['span']
    # 首先构建context
    context_sentence = f"{event_type}[SEP]{content[:trigger_span[0]]}[SEP]{content[trigger_span[0]:trigger_span[1]]}[SEP]{content[trigger_span[1]:]}"
    trigger_append_length = len(event_type) + len('[SEP]')
    def index_convert(old_idx: int):
        if old_idx < trigger_span[0]:
            return old_idx + trigger_append_length
        elif trigger_span[0] <= old_idx < trigger_span[1]:
            return old_idx + trigger_append_length + 5
        else:
            return old_idx + trigger_append_length + 10

    for idx in range(len(data_dict['other_mentions'])):
        old_span = data_dict['other_mentions'][idx]['span']
        new_span = (index_convert(old_span[0]), index_convert(old_span[1]))
        data_dict['other_mentions'][idx]['span'] = new_span

    # 先构建EAR
    EAR_questions = []
    # 构建EAR-获取schema
    if dataset_type == 'FewFC':
        schema = EE_settings.event_available_roles
        new_schema = {}
        for key, value in schema.items():
            # breakpoint()
            new_schema[key] = list(EE_settings.role_types_translate[x] for x in value)
        schema = new_schema
        role_types = list(EE_settings.role_types_translate[x] for x in EE_settings.role_types)
        for idx in range(len(mentions)):
            mentions[idx]['role'] = EE_settings.role_types_translate[mentions[idx]['role']]
    elif dataset_type == 'Duee':
        schema = EE_settings.duee_event_available_roles
        role_types = EE_settings.duee_role_types
    else:
        raise Exception(f'数据集{dataset_type}不存在！')

    role_index = {v: i for i, v in enumerate(role_types)}
    # 构建EAR-构建正例
    EAR_results = []
    exist_role = set()  # 所有在该事件中出现过的论元角色
    for elem_mention in mentions:
        exist_role.add(elem_mention['role'])
        role, span, word = elem_mention['role'], tuple(elem_mention['span']), elem_mention['word']
        cur_sample = {
            'content': content,
            'event_type': event_type,
            'role': role,
            'span': span,
            'word': word,
            'neg': False
        }
        EAR_results.append(cur_sample)
    pos_cnt = len(EAR_results)
    # 构建EAR-构建负例
    for elem_role in schema[event_type]:  # 对schema中该事件类型所具有对论元角色进行遍历
        if elem_role not in exist_role:  # 如果某个论元没有出现过
            cur_sample = {
                'content': content,
                'event_type': event_type,
                'role': elem_role,
                'span': (0, 0),  # 该论元对应的span为开头
                'word': None,  # 对应的词语则为None
                'neg': True
            }
            EAR_results.append(cur_sample)
    # 构建EAR-提取必要信息
    for elem_info in EAR_results:
        role, word, span = elem_info['role'], elem_info['word'], elem_info['span']
        is_neg = elem_info['neg']
        # question = f'词语{word}在事件{event_type}中作为什么角色？'
        question = f'在该句子的"{event_type}"事件中作为"{role}"的是哪一个词？'
        EAR_questions.append({
            'question': question,
            'label': span,
            'EAR_gt': word if word is not None else '',  # 如果为负例，把None切换成空字符串
            'neg': is_neg
        })

    # 然后构建ERR
    ERR_questions = []
    words = list(jieba.cut(content))  # 对句子进行分词
    entities = list(x.text for x in stanza_nlp(content).ents)  # 提取句子中所有的命名实体
    ERR_results = []
    exist_words = set(x['word'] for x in mentions)  # 所有出现的抽取词
    for elem in exist_words:
        if elem in entities:
            entities.remove(elem)  # 把所有实体中的抽取词移除
        if elem in words:
            words.remove(elem)  # 把分词中的抽取词移除
    # 这样entities和words当中的词就一定不会与标注重复了
    for elem_mention in mentions:
        # pos
        role, span, word = elem_mention['role'], tuple(elem_mention['span']), elem_mention['word']
        ERR_results.append({
            'content': content,
            'event_type': event_type,
            "word": word,
            'role': role
        })
    # ERR的数量需要与EAR相同，所以构建剩余的
    pos_cnt = len(ERR_results)
    for elem_result in EAR_results[pos_cnt:]:
        # neg
        random_word = random.choice(entities) if len(entities) != 0 else None
        if random_word is None or random_word in exist_words:
            if random_word is not None:
                entities.remove(random_word)
            random_word = random.choice(words)  # words的长度应该不会为0吧
            words.remove(random_word)
        ERR_results.append({
            'content': content,
            'event_type': event_type,
            "word": random_word,
            'role': None   # 该词不对应任何角色
        })
    for elem_info in ERR_results:
        role, word = elem_info['role'], elem_info['word']
        question = f'词语"{word}"在该句子的"{event_type}"事件中作为什么角色？'
        # breakpoint()
        ERR_questions.append({
            'question': question,
            'label': role,
            'ERR_gt': role_index[role] if role is not None else len(role_index)  # 负例在末尾
        })

    results = []
    for idx, (elem_a, elem_r) in enumerate(zip(EAR_questions, ERR_questions)):
        results.append({
            "context": context_sentence,
            "EAR_question": elem_a['question'],
            "EAR_label": elem_a['label'],
            'ERR_question': elem_r['question'],
            'ERR_label': elem_r['label'],
            'content': content,
            'EAR_gt': elem_a['EAR_gt'],
            'ERR_gt': elem_r['ERR_gt'],
            'neg': elem_a['neg']
        })
    return results


def new_generate_EAR_target(data_dict: Dict[str, Any]):
    """

    :param data_dict:
    :return:
    """
    input_ids = data_dict['input_ids']
    input_length = len(input_ids)
    offset_mapping = data_dict['offset_mapping']
    input_ids_length = len(input_ids)
    # argument_start_target, argument_end_target = np.zeros(input_ids_length), np.zeros(input_ids_length)
    span = data_dict['EAR_label']
    if span == (0, 0) or span == [0, 0]:
        token_span = (0, 0)
    else:
        token_span = (-1, -1)
        for idx, elem in enumerate(offset_mapping):
            if elem[0] <= span[0] < elem[1]:
                token_span = (idx, token_span[1])
            if elem[0] <= span[1] - 1 < elem[1]:
                token_span = (token_span[0], idx)
            # 如果token_span中仍有-1，说明原句中某个字为非法字符，而在tokenize时被扔掉了。
    if token_span[0] == -1:
        for idx, elem in enumerate(offset_mapping):
            reach = elem[1]
            if reach > token_span[0]:
                token_span = (reach, token_span[1])
        if token_span[0] == -1:
            token_span = (token_span[1], token_span[1])
    if token_span[1] == -1:
        for idx, elem in enumerate(offset_mapping):
            reach = elem[1]
            if reach > token_span[1]:
                token_span = (token_span[0], reach)
        if token_span[1] == -1:
            token_span = (token_span[0], token_span[0])
    if token_span[0] == -1 or token_span[1] == -1:
        print('糟了')
    if token_span[0] >= input_length or token_span[1] >= input_length:
        print('出现token span越界情况')
        if token_span[0] >= input_length:
            token_span = (input_length - 1, token_span[1])
        if token_span[1] >= input_length:
            token_span = (token_span[0], input_length - 1)
    # assert ''.join(data_dict['token'][token_span[0]: token_span[1] + 1]) == data_dict['EAR_gt'] if span != (0, 0) else True, data_dict
    # argument_start_target[token_span[0]] = 1
    # argument_end_target[token_span[1]] = 1
    argument_start_target = np.array(token_span[0], dtype=np.int)
    argument_end_target = np.array(token_span[1], dtype=np.int)

    data_dict['argument_target_start'] = argument_start_target
    data_dict['argument_target_end'] = argument_end_target
    return [data_dict]


def new_generate_ERR_target(data_dict: Dict[str, Any], dataset_type: str):
    if dataset_type == 'FewFC':
        role_types = list(EE_settings.role_types_translate[x] for x in EE_settings.role_types)
    elif dataset_type == 'Duee':
        role_types = EE_settings.duee_role_types
    else:
        raise Exception(f'不存在{dataset_type}数据集!')
    role_index = {v: i for i, v in enumerate(role_types)}
    role_label = data_dict['ERR_label']
    role_target = np.zeros(len(role_types) + 1)  # 在最后面加上一个负例label
    if role_label is None:
        label_idx = len(role_types)
    else:
        label_idx = role_index[role_label]
    data_dict['role_target'] = label_idx  # (role_cnt)
    return [data_dict]


def construct_T_TWord_context(data_dicts: Dict[str, Any], dataset_type: str, stanza_nlp):
    """
    为DualQA_Trigger构建QT， QTWord， context
    :param data_dicts: [content, event_type, trigger_info, other_mentions]
    :param dataset_type:
    :return:
    """
    content, event_type, trigger_info, mentions = \
        data_dicts['content'], data_dicts['event_type'], data_dicts['trigger_info'], data_dicts['other_mentions']

    trigger_span = trigger_info['span']

    # context
    context_sentence = f'{event_type}[SEP]{content}'
    # 给content添加的前缀改变了index，需要转换一下
    appended_length = len(event_type) + len('[SEP]')
    for idx in range(len(data_dicts['other_mentions'])):
        old_span = data_dicts['other_mentions'][idx]['span']
        new_span = (old_span[0] + appended_length, old_span[1] + appended_length)
        data_dicts['other_mentions'][idx]['span'] = new_span

    # Question T
    T_question = '该句子中作为事件触发词的是哪一个词？'
    T_label = trigger_span
    T_gt = trigger_info['word']

    # Question TWord
    processed = stanza_nlp(content).to_dict()
    #  先决定是正例还是负例:0-负例 1-正例
    choice = random.choice([0, 1])
    # 随机选一个词。如果为空，则使用第一个词
    if choice == 0:
        verbs = []
        for elem in processed[0]:
            if elem['upos'] == 'VERB':
                verbs.append({
                    'text': elem['text'],
                    'span': (elem['start_char'], elem['end_char'])
                })
        if len(verbs) > 0:
            verb = random.choice(verbs)
        else:
            verb = {
                'text': processed[0][0]['text'],
                'span': (processed[0][0]['start_char'], processed[0][0]['end_char'])
            }
        TWord_label = 0
    else:
        verb = {
            'text': trigger_info['word'],
            'span': trigger_span
        }
        TWord_label = 1
    TWord_question = f'该句子中的"{verb["text"]}"能否作为触发词？'

    return {
        'context': context_sentence,
        'T_question': T_question,
        'T_label': T_label,
        'T_gt': T_gt,
        'TWord_question': TWord_question,
        'TWord_label': TWord_label,

    }


def generate_T_label(data_dicts: Dict[str, Any]):
    context_length = len(data_dicts['input_ids'])

    span = data_dicts['T_label']
    token_span = tokenize_tools.charSpan_to_tokenSpan(span, data_dicts['offsets_mapping'])
    start_label, end_label = np.array([0] * context_length), np.array([0] * context_length)
    start_label[token_span[0]] = 1
    end_label[token_span[1]] = 1

    data_dicts.update({
        'start_T_label': start_label,
        'end_T_label': end_label
    })
    return data_dicts

"""
调用part
"""


def data_filter(data_path: str = initial_dataset_path, dataset_type: str = dataset_type, subset_name: str = 'train', output_name: str = 'filtered'):
    """
    该函数对数据进行过滤

    - 去除长度大于某个值的数据
    :param data_path:
    :param dataset_type:
    :return:
    """
    if dataset_type == 'FewFC':
        loaded = load_FewFC_ee(data_path)
    elif dataset_type == 'Duee':
        loaded = load_Duee_ee_formated(data_path)
    else:
        raise Exception(f'[dual_qa:dataset_factory]不存在{dataset_type}数据集！')
    data_dicts = loaded[subset_name]

    # 去除content过长的sample
    data_dicts = tools.map_operation_to_list_elem(remove_illegal_length, data_dicts)
    # [content, events]

    f = open(temp_path + output_name, 'w', encoding='utf-8')
    for elem in data_dicts:
        s = json.dumps(elem, ensure_ascii=False)
        f.write(s + '\n')
    f.close()


def divide_by_event_type(last_output_name: str, output_name: str, dataset_type: str = dataset_type):
    """
    按照事件类型将数据进行划分
    :param last_temp_path:
    :param dataset_type:
    :param output_name:
    :return:
    """
    data_dicts = list(json.loads(x) for x in open(temp_path + last_output_name, 'r', encoding='utf-8').read().strip().split('\n'))

    # 按content-事件类型-触发词-进行划分
    data_dicts = tools.map_operation_to_list_elem(split_by_content_type_trigger, data_dicts)
    # [content, event_type, trigger_info, other_mentions]

    f = open(temp_path + output_name, 'w', encoding='utf-8')
    for elem in data_dicts:
        s = json.dumps(elem, ensure_ascii=False)
        f.write(s + '\n')
    f.close()


def construct_context_and_questions(last_output_name: str, output_name: str, dataset_type: str = dataset_type, from_cache: bool = False):
    """

    :param last_output_name:
    :param output_name:
    :param dataset_type:
    :return:
    """
    if from_cache:
        results = list(json.loads(x) for x in open(temp_path + output_name, 'r', encoding='utf-8').read().strip().split('\n'))
    else:
        data_dicts = list(json.loads(x) for x in open(temp_path + last_output_name, 'r', encoding='utf-8').read().strip().split('\n'))
        data_dicts = data_dicts

        # 同时构造context，EAR问题与ERR问题。
        stanza_nlp = stanza.Pipeline(lang='zh', processors='tokenize,ner')
        results = []
        for elem in tqdm(data_dicts):
            results.extend(construct_EAR_ERR_context(elem, dataset_type, stanza_nlp))
        # [context, content, EAR_question, EAR_label, ERR_question, ERR_label]


    # 平衡results中的正负例个数
    neg_results, pos_results = [], []
    for elem in results:
        if elem['neg']:
            neg_results.append(elem)
        else:
            pos_results.append(elem)
    neg_cnt, pos_cnt = len(neg_results), len(pos_results)
    if neg_cnt > pos_cnt:
        to_delete_cnt = neg_cnt - pos_cnt
        for i in range(to_delete_cnt):
            neg_results.pop(random.randrange(0, i + 1))
    results = pos_results + neg_results

    f = open(temp_path + output_name, 'w', encoding='utf-8')
    for elem in results:
        s = json.dumps(elem, ensure_ascii=False)
        f.write(s + '\n')
    f.close()


def construct_context_and_questions_trigger(last_output_name: str, output_name: str, dataset_type: str = dataset_type):
    data_dicts = list(
        json.loads(x) for x in open(temp_path + last_output_name, 'r', encoding='utf-8').read().strip().split('\n'))
    data_dicts = data_dicts

    # 同时构造context，EAR问题与ERR问题。
    stanza_nlp = stanza.Pipeline(lang='zh', processors='tokenize,lemma,pos')
    results = []
    for elem in tqdm(data_dicts):
        results.append(construct_T_TWord_context(elem, dataset_type, stanza_nlp))

    f = open(temp_path + output_name, 'w', encoding='utf-8')
    for elem in results:
        s = json.dumps(elem, ensure_ascii=False)
        f.write(s + '\n')
    f.close()


def tokenize_context_and_questions(last_output_name: str, output_name: str, dataset_type: str = dataset_type):
    """
    需要分别对context，EAR_question与ERR_quesiton进行tokenize
    :param last_output_name:
    :param output_name:
    :param dataset_type:
    :return:
    """
    data_dicts = list(json.loads(x) for x in open(temp_path + last_output_name, 'r', encoding='utf-8').read().strip().split('\n'))

    # tokenize
    data_dict = tools.transpose_list_of_dict(data_dicts)
    lst_tokenizer = tokenize_tools.bert_tokenizer()
    context_result = lst_tokenizer(data_dict['context'])
    context_result = tools.transpose_list_of_dict(context_result)
    context_result = tools.modify_key_of_dict(context_result, lambda x: x)
    EAR_result = lst_tokenizer(data_dict['EAR_question'])
    EAR_result = tools.transpose_list_of_dict(EAR_result)
    EAR_result = tools.modify_key_of_dict(EAR_result, lambda x: 'EAR_' + x)
    ERR_result = lst_tokenizer(data_dict['ERR_question'])
    ERR_result = tools.transpose_list_of_dict(ERR_result)
    ERR_result = tools.modify_key_of_dict(ERR_result, lambda x: 'ERR_' + x)
    data_dict.update(context_result)
    data_dict.update(EAR_result)
    data_dict.update(ERR_result)
    data_dicts = tools.transpose_dict_of_list(data_dict)

    f = open(temp_path + output_name, 'w', encoding='utf-8')
    for elem in data_dicts:
        s = json.dumps(elem, ensure_ascii=False)
        f.write(s + '\n')
    f.close()


def tokenize_context_and_questions_trigger(last_output_name: str, output_name: str, dataset_type: str = dataset_type):
    data_dicts = list(json.loads(x) for x in open(temp_path + last_output_name, 'r', encoding='utf-8').read().strip().split('\n'))

    # tokenize
    data_dict = tools.transpose_list_of_dict(data_dicts)
    lst_tokenizer = tokenize_tools.bert_tokenizer()

    context_result = lst_tokenizer(data_dict['context'])
    context_result = tools.transpose_list_of_dict(context_result)
    context_result = tools.modify_key_of_dict(context_result, lambda x: x)

    T_result = lst_tokenizer(data_dict['T_question'])
    T_result = tools.transpose_list_of_dict(T_result)
    T_result = tools.modify_key_of_dict(T_result, lambda x: 'T_' + x)

    TWord_result = lst_tokenizer(data_dict['TWord_question'])
    TWord_result = tools.transpose_list_of_dict(TWord_result)
    TWord_result = tools.modify_key_of_dict(TWord_result, lambda x: 'TWord_' + x)

    data_dict.update(context_result)
    data_dict.update(T_result)
    data_dict.update(TWord_result)
    data_dicts = tools.transpose_dict_of_list(data_dict)

    pickle.dump(data_dicts, open(temp_path + output_name, 'wb'))


def generate_label(last_output_name: str, output_name: str, dataset_type: str = dataset_type):
    """
    为训练集生成label
    :param last_output_name:
    :param output_name:
    :param dataset_type:
    :return:
    """
    data_dicts = list(json.loads(x) for x in open(temp_path + last_output_name, 'r', encoding='utf-8').read().strip().split('\n'))

    data_dicts = tools.map_operation_to_list_elem(new_generate_EAR_target, data_dicts)
    results = []
    for elem in data_dicts:
        results.append(new_generate_ERR_target(elem, dataset_type))
    data_dicts = results
    # data_dicts = tools.map_operation_to_list_elem(new_generate_ERR_target, data_dicts)

    pickle.dump(data_dicts, open(temp_path + output_name, 'wb'))


def generate_label_trigger(last_output_name: str, output_name: str, dataset_type: str = dataset_type):

    data_dicts = list(json.loads(x) for x in open(temp_path + last_output_name, 'r', encoding='utf-8').read().strip().split('\n'))

    # data_dicts = tools.map_operation_to_list_elem(new_generate_EAR_target, data_dicts)
    results = []
    for elem in data_dicts:
        results.append(generate_T_label(elem))
    data_dicts = results
    # data_dicts = tools.map_operation_to_list_elem(new_generate_ERR_target, data_dicts)

    pickle.dump(data_dicts, open(temp_path + output_name, 'wb'))


def construct_context_and_question_dualqa_trigger(last_output_name: str, output_name: str, dataset_type: str = dataset_type):
    pass

"""
任务对应的pipeline
"""


def generate_data_for_dualqa():
    print(f'正在预处理{dataset_type}数据')
    print(f'初始路径：{initial_dataset_path}')
    # 首先对train和val进行筛选
    print('Step 1 - 正在去除过长句子')
    data_filter(initial_dataset_path, dataset_type, 'train', f'train.{dataset_type}.filtered_length.balanced.jsonl')
    data_filter(initial_dataset_path, dataset_type, 'valid', f'valid.{dataset_type}.filtered_length.balanced.jsonl')

    # 然后按照事件类型进行切分
    print('Step 2 - 正在按事件类型拆分数据')
    divide_by_event_type(f'train.{dataset_type}.filtered_length.balanced.jsonl', f'train.{dataset_type}.divided.balanced.jsonl')
    divide_by_event_type(f'valid.{dataset_type}.filtered_length.balanced.jsonl', f'valid.{dataset_type}.divided.balanced.jsonl')

    # 然后构造EAR question, ERR question, context
    print('正在生成context与question')
    construct_context_and_questions(f'train.{dataset_type}.divided.balanced.jsonl', f'train.{dataset_type}.questioned.balanced.test.jsonl')
    construct_context_and_questions(f'valid.{dataset_type}.divided.balanced.jsonl', f'valid.{dataset_type}.questioned.balanced.test.jsonl')

    # 对context和question进行tokenize
    print('正在tokenize')
    # tokenize_context_and_questions(f'train.{dataset_type}.questioned.balanced.jsonl', f'train.{dataset_type}.tokenized.balanced.jsonl')
    # tokenize_context_and_questions(f'valid.{dataset_type}.questioned.balanced.jsonl', f'valid.{dataset_type}.tokenized.balanced.jsonl')

    # 然后为训练集生成label
    print('正在生成label')
    # generate_label(f'train.{dataset_type}.tokenized.balanced.jsonl', f'train.{dataset_type}.labeled.balanced.pk')

    # 为评价集生成gt
    print('正在生成gt')


    # 得到的train.labeled.jsonl与valid.questioned.jsonl就是最终输入给模型的数据


def generate_data_for_dualqa_trigger():
    print(f'为DualQA_Trigger生成训练、评价数据')
    print(f'正在预处理{dataset_type}数据')
    print(f'初始路径：{initial_dataset_path}')
    # 首先对train和val进行筛选
    print('Step 1 - 正在去除过长句子')
    data_filter(initial_dataset_path, dataset_type, 'train', f'train.DualQA_Trigger.{dataset_type}.filtered_length.jsonl')
    data_filter(initial_dataset_path, dataset_type, 'valid', f'valid.DualQA_Trigger.{dataset_type}.filtered_length.jsonl')

    # 然后按照事件类型进行切分
    print('Step 2 - 正在按事件类型拆分数据')
    divide_by_event_type(f'train.DualQA_Trigger.{dataset_type}.filtered_length.jsonl', f'train.DualQA_Trigger.{dataset_type}.divided.jsonl')
    divide_by_event_type(f'valid.DualQA_Trigger.{dataset_type}.filtered_length.jsonl', f'valid.DualQA_Trigger.{dataset_type}.divided.jsonl')

    # 然后构造EAR question, ERR question, context
    print('Step 3 - 正在生成context与question')
    construct_context_and_questions_trigger(f'train.DualQA_Trigger.{dataset_type}.divided.jsonl', f'train.DualQA_Trigger.{dataset_type}.questioned.jsonl')
    construct_context_and_questions_trigger(f'valid.DualQA_Trigger.{dataset_type}.divided.jsonl', f'valid.DualQA_Trigger.{dataset_type}.questioned.jsonl')

    # 对context和question进行tokenize
    print('Step 4 - 正在tokenize')
    tokenize_context_and_questions_trigger(f'train.DualQA_Trigger.{dataset_type}.questioned.jsonl', f'train.DualQA_Trigger.{dataset_type}.tokenized.pk')
    tokenize_context_and_questions_trigger(f'valid.DualQA_Trigger.{dataset_type}.questioned.jsonl', f'valid.DualQA_Trigger.{dataset_type}.tokenized.pk')

    print('Step 5 - 为train生成label')
    generate_label_trigger(f'train.DualQA_Trigger.{dataset_type}.tokenized.pk', f'train.DualQA_Trigger.{dataset_type}.labeled.pk')

def main():
    # generate_data_for_dualqa()
    generate_data_for_dualqa_trigger()

if __name__ == '__main__':
    main()
