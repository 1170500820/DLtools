"""
管理record文件的读/写

以及一些方便的处理工具
"""
import pickle

from type_def import *


"""
首先是Record的格式类型

Record具体的格式说明将后续整理成文档，现在先凑活看把
"""


# record的格式
class MyRecordType(TypedDict):
    meta: Any
    data: Dict[str, Any]


def write_record(record: MyRecordType, save_path: str, record_attr_v: Dict[str, Any] = None, record_attr_s: Set[str] = None):
    """

    :param record: record数据本体
    :param save_path: 保存该文件的路径
    :param record_attr_v: 应当写入文件名的属性的键值对
    :param record_attr_s: 应当写入文件名的属性的值
    :return:
    """
    if save_path[-1] != '/':
        save_path += '/'

    filename = 'record'
    for key, value in record_attr_v.items():
        filename += f'.{key}-{str(value)}'
    for elem in record_attr_s:
        filename += f'.{str(elem)}'

    save_name = save_path + filename
    pickle.dump(record, open(save_name, 'wb'))


def build_record(data_lst: list, names: List[str], descriptions: List[List[str]] = None) -> MyRecordType:
    """

    :param data_lst: 需要写入的data
    :param names: 需要写入的data的名字
    :param descriptions: 每一个data的维度描述
    :return:
    """
    record_dict = {
        "meta": None,
        "data": {}
    }
    for (elem_data, elem_name, elem_des) in zip(data_lst, names, descriptions):
        record_dict['data'][elem_name] = elem_data
        record_dict['data']['_' + elem_name] = elem_des
    return record_dict


def simple_save_record(anything):
    write_record()
