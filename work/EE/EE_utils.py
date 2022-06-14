from utils import tools
import functools
import json

from type_def import *
from dataset.ee_dataset import load_FewFC_ee, load_Duee_ee_formated, load_FewFC_ee_full
from work.EE import EE_settings


# 未改造的函数，完全通用，不针对特定结构

# 替换句子中的非法字符
# *tools.replace_chars(content: str, replace_dict: Dict[str, str] = None) -> str:
# 过滤掉长度过长的句子
# *无


# 改造成能够传入list_of_dict格式的函数
func_remove_illegal = functools.partial(tools.replace_chars_in_dict, replace_dict={' ': '_'})


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



# remove illegal chars
def remove_function(data_dict: Dict[str, Any]):
    """
    改变dict中的某个值
    :param data_dict:
    :return:
    """
    return tools.replace_chars_in_dict(data_dict, 'content', {' ': '_'})


def remove_illegal_length(data_dict: Dict[str, Any], max_length: int = EE_settings.max_sentence_length):
    if len(data_dict['content']) >= max_length:
        return []
    else:
        return [data_dict]


def extract_event_types(data_dict: Dict[str, Any]):
    events = data_dict['events']
    event_types = [x['type'] for x in events]
    data_dict['event_types'] = list(set(event_types))
    return [data_dict]


"""
事件抽取数据处理常用流程
"""


def get_event_extraction_essentials(filepath: str):
    """
    获得事件抽取模型所常用的数据以及属性

    所有配置都按默认进行

    这只是一个模版，直接调用并不方便
    :param filepath:
    :return:
    """
    # read file
    json_lines = tools.read_json_lines(filepath)
    json_dict_lines = [{'content': x['content'], 'events': x['events']} for x in json_lines]
    # [content, events]

    # remove illegal characters
    tools.list_of_dict_modify()


def check_ee_dataset(dataset_type: str):
    """
    检查ee数据集的一些性质
    :param dataset_type:
    :return:
    """
    if dataset_type == 'FewFC':
        loaded = load_FewFC_ee('../../data/NLP/EventExtraction/FewFC-main')
    elif dataset_type == 'Duee':
        loaded = load_Duee_ee_formated('../../data/NLP/EventExtraction/duee')
    else:
        raise Exception(f'不存在{dataset_type}数据集！')

    overlap_cnt = 0
    for elem in loaded['train'] + loaded['valid']:
        for elem_event in elem['events']:
            role_set = set()
            overlap = False
            for elem_mention in elem_event['mentions']:
                if elem_mention['role'] not in role_set:
                    role_set.add(elem_mention['role'])
                else:
                    overlap = True
                    break
            if overlap:
                overlap_cnt += 1
    print(f'{len(loaded["train"] + loaded["valid"])} samples in total, {overlap_cnt} overlap samples')


"""
事件抽取数据集
数据处理函数
"""


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




"""
事件抽取数据集
数据处理函数的调用函数
"""


def data_filter(data_path: str, dataset_type: str, temp_path: str, subset_name: str = 'train', output_name: str = 'filtered', full: bool = False):
    """
    该函数对数据进行过滤

    - 去除长度大于某个值的数据
    :param data_path:
    :param dataset_type:
    :return:
    """
    if dataset_type == 'FewFC' and full:
        loaded = load_FewFC_ee_full(data_path)
    elif dataset_type == 'FewFC' and not full:
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


def divide_by_event_type(last_output_name: str, output_name: str, temp_path: str, dataset_type: str):
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






if __name__ == '__main__':
    # fewfc = load_FewFC_ee('../../data/NLP/EventExtraction/FewFC-main')
    duee = load_Duee_ee_formated('../../data/NLP/EventExtraction/duee')
    # check_ee_dataset('FewFC')
    # check_ee_dataset('Duee')