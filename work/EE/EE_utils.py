from type_def import *
from utils import tools
import functools
from dataset.ee_dataset import load_FewFC_ee, load_Duee_ee_formated
from work.EE import EE_settings


# 未改造的函数，完全通用，不针对特定结构

# 替换句子中的非法字符
# *tools.replace_chars(content: str, replace_dict: Dict[str, str] = None) -> str:
# 过滤掉长度过长的句子
# *无



# 改造成能够传入list_of_dict格式的函数
func_remove_illegal = functools.partial(tools.replace_chars_in_dict, replace_dict={' ': '_'})




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
这下面是所有事件抽取数据集的读取函数
"""


if __name__ == '__main__':
    # fewfc = load_FewFC_ee('../../data/NLP/EventExtraction/FewFC-main')
    duee = load_Duee_ee_formated('../../data/NLP/EventExtraction/duee')
    # check_ee_dataset('FewFC')
    # check_ee_dataset('Duee')