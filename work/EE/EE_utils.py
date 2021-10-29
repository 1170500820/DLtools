from type_def import *
from utils import tools
import functools


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


def remove_illegal_length(data_dict: Dict[str, Any]):
    if len(data_dict['content']) >= 254:
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


