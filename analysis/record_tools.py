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


def build_record(data_lst: list, names: List[str], descriptions: List[List[Union[str, Dict[str, List[str]]]]] = None) -> MyRecordType:
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


def parser_record_name(record_name: str) -> dict:
    """
    输入一个record的文件名，返回该record的文件名中所包含的属性
    :param record_name: record的文件名
    :return:
    """
    parts = record_name.split('.')
    if parts[0] != 'record':
        raise Exception('[parse_record_name]record文件名应当以record开头')
    record_attrs = {"key": {}, 'attrs': set()}
    for elem in parts[1:]:
        if '-' in elem:
            if elem.count('-') > 1:
                raise Exception(f'[parse_record_name]record文件属性错误-包含多个连字符->{elem}')
            key, value = elem.split('-')
            record_attrs['key'][key] = value
        else:
            record_attrs['attrs'].add(elem)
    return record_attrs


# 选择器的早期设计，通过一个简单dict实现
# 四个key值分别代表
class MySelectConditionDict(TypedDict):
    include_key: dict
    include_attr: set
    exclude_key: dict
    exclude_attr: set


def condition_judge(filename: str, condition: MySelectConditionDict) -> bool:
    """
    判断一个文件名是否符合条件
    :param filename:
    :param condition:
    :return:
    """
    record_attrs = parser_record_name(filename)
    if 'include_key' in condition:
        for key, value in condition['include_key']:
            if key not in record_attrs['key']:
                return False
            elif value != record_attrs['key'][key]:
                return False
    if 'include_attr' in condition:
        for attrs in condition['include_attr']:
            if attrs not in record_attrs['attrs']:
                return False
    if 'exclude_key' in condition:
        for key, value in condition['exclude_key']:
            if key not in record_attrs['key']:
                continue
            elif value == record_attrs['key'][key]:
                return False
    if 'exclude_attr' in condition:
        for attrs in condition['exclude_attr']:
            if attrs in record_attrs['attrs']:
                return False
    return True


def select_files(filenames: List[str], condition: MySelectConditionDict):
    """
    返回符合条件的文件名
    :param filenames:
    :param condition: 条件dict
    :return:
    """
    return list(filter(lambda x: condition_judge(x, condition), filenames))
