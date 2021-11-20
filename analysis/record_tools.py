"""
管理record文件的读/写

record类型是recorder从模型中抽取信息之后保存为pickle文件的格式。
而jupyter-lab上又会将record解析为ndarray与dimension。不过ndarray属于jupyter-lab-analyze-ui的内部状态，因此只需要为jupyter-lab
提供转换的接口。从外部文件读取，或者转换数据格式时，仍然以record为中心

any    ->    record    ->    ndarray
    everywhere       ui only

"""
import pickle
import numpy as np

from type_def import *
from settings import *
from utils import format_convert, tools
from pathlib import Path


"""
record与RecordDataWrap的定义

Record具体的格式说明将后续整理成文档，现在先凑活看把
"""


# record的格式
class MyRecordType(TypedDict):
    meta: Any
    data: Dict[str, Any]


class RecordDataWrap:
    """
    通过numpy创建数组的时候，会自动将sequential数据转化为矩阵，但是recorder获取的数据有很多是序列形式作为单个元素的。
    为了使numpy不将这些数据拆分转换，需要用RecordDataWrap包装起来。

    为了统一处理，所有的数据都用该方法包装。

    后续的开发，可以加入钩子，替换一些内置方法比如说__add__
    """
    shape = ()

    def __init__(self, data):
        self.data = data
        if hasattr(self.data, 'shape'):
            self.shape = getattr(self.data, 'shape')

    def get(self):
        return self.data

    def __call__(self):
        return self.get()

    def __repr__(self):
        return self.get().__repr__()

    # 内置方法-索引
    def __getitem__(self, item):
        if hasattr(self, 'data'):
            if hasattr(self.data, '__getitem__'):
                return self.data.__getitem__(item)
            raise Exception('[RecordDataWrap]内部data没有__getitem__属性！')
        raise Exception('[RecordDataWrap]未找到data属性！')

    def __setitem__(self, key, value):
        if hasattr(self, 'data'):
            if hasattr(self.data, '__setitem__'):
                return self.data.__setitem__(key, value)
            raise Exception('[RecordDataWrap]内部data没有__setitem__属性！')
        raise Exception('[RecordDataWrap]未找到data属性！')

    def __delitem__(self, key):
        if hasattr(self, 'data'):
            if hasattr(self.data, '__delitem__'):
                return self.data.__delitem__(key)
            raise Exception('[RecordDataWrap]内部data没有__delitem__属性！')
        raise Exception('[RecordDataWrap]未找到data属性！')

    # 内置方法-迭代
    def __iter__(self):
        if hasattr(self, 'data'):
            if hasattr(self.data, '__iter__'):
                return self.data.__iter__()
            raise Exception('[RecordDataWrap]内部data没有__iter__属性！')
        raise Exception('[RecordDataWrap]未找到data属性！')

    def __next__(self):
        if hasattr(self, 'data'):
            if hasattr(self.data, '__next__'):
                return self.data.__next__()
            raise Exception('[RecordDataWrap]内部data没有__next__属性！')
        raise Exception('[RecordDataWrap]未找到data属性！')


"""
convert to record

将各种形式的数据转化为record的形式。
any -> record
最需要做的就是分析维度。

- build_record
    将一个序列数据转化为record的形式。names作为key必须提供，而维度信息也就是descriptions是可选的
"""


def build_record(data_lst: list, names: List[str], descriptions: List[List[Union[str, Dict[str, List[str]]]]] = None) -> MyRecordType:
    """
    将一个序列的data数据包装成record的形式

    data_lst        d1,     d2,     ...
    names           loss,   scores, ...
    descriptions    [b, n], [b, float], ...

    :param data_lst: 需要写入的data
    :param names: 需要写入的data的名字
    :param descriptions: 每一个data的维度描述
    :return:
    """
    record_dict = {
        "meta": None,
        "data": {}
    }
    if names is None:
        names = [None] * len(data_lst)
    if descriptions is None:
        descriptions = [None] * len(data_lst)
    for (elem_data, elem_name, elem_des) in zip(data_lst, names, descriptions):
        record_dict['data'][elem_name] = elem_data
        record_dict['data']['_' + elem_name] = elem_des
    return record_dict


def merge_record(record1: MyRecordType, record2: MyRecordType):
    """
    合并两个record中的数据

    目前没有meta的定义，因此直接合并
    后续引入了meta，可能需要先根据meta判断meta和data是否能够合并，然后再执行
    :param record1:
    :param record2:
    :return:
    """
    #
    record1['data'].update(record2['data'])
    return record1


def build_record_with_infer(seq_data, seq_data_name: str = 'default', elem_name: str = 'seq_elem') -> MyRecordType:
    """
    使用infer_sequence_shape推断seq_data的shape，然后给每一个维度一个默认的名字，并构建record
    :param seq_data: 任意类型数据均可接受，但是对序列数据处理才有意义
    :param seq_data_name: 整个seq_data的名字
    :param elem_name: sequence中每一个最小元素的名字
    :return:
    """
    shape = tools.infer_sequence_shape(seq_data)
    desc = [elem_name]
    default_dim_names = list(f'dim-{x + 1}' for x in range(len(shape)))
    descriptions = default_dim_names + desc
    result_record = build_record([seq_data], [seq_data_name], [descriptions])
    return result_record


"""
record IO

record文件的读与写会包含很多东西
读
    读record
    读其他常见格式并转为record
    读任意格式、推理其维度、转为record
写
    写record
    将常见格式转为record并写
    将任意格式、推断其维度、转为record并写
    cache
更多功能待补充

- read_record
"""


def parse_record_data(record_dict: dict):
    """
    record文件中用于存放数据的dict，默认是record['data']
    将record转换为ndarray与dimension。

    准确来说是转换为ndarray，并抽取dimension信息（如果存在的话）
    :param record_dict:
    :return:
    """
    meta = record_dict['meta']
    d = record_dict['data']
    d_attr, d_data = {}, {}
    for key, value in d.items():
        if key[0] == '_':
            d_attr[key] = value
        else:
            d_data[key] = value

    matrix_dict = {}
    for key, value in d_data.items():
        if '_' + key in d_attr:
            dimension_lst = d_attr['_' + key]
            value_array = wrap_record_item_to_ndarray(value, len(dimension_lst) - 1)
            matrix_dict[key] = {
                'value': value_array,
                'dimension': dimension_lst
            }
        else:
            value_array = wrap_record_item_to_ndarray(value, 0)
            matrix_dict[key] = {
                'value': value_array,
                'dimension': ['Unknown']
            }
    return matrix_dict


def read_record(filepath: str) -> Dict[str, Any]:
    """
    主入口函数。

    本函数的目的是读取一个目录下的所有记录文件，解析出其中的record数据并转化为np.ndarray格式，然后以dict形式返回

    todo 后续会加入文件配置功能，可以动态指定多文件，以及只选取部分文件。
    todo 探索自由度更高的配置，能够根据记录文件的元信息决定组合方式
    :param filepath:
    :return:
    """
    record = pickle.load(open(filepath, 'rb'))
    m = parse_record_data(record)
    return m


def universal_load(filepath: str) -> Tuple[np.ndarray, list]:
    """
    尝试根据文件名和文件内容，对文件的类型进行推断，先转换为record格式，然后转换为ndarray
    :param filepath:
    :return:
    """
    p = Path(filepath)
    suffix = p.suffix
    if p.stem.split('.')[0] == 'record':
        m = read_record(filepath)
        ndary, dims = m['value'], m['dimension']
        return ndary, dims
    elif suffix == '.conll':
        conll_lst = format_convert.conllner_to_lst(filepath)
        conll_record = build_record([conll_lst], ['conll_samples'], [['samples', 'Dict[id, chars and tags] conll sample dict']])
        conll_matrix = parse_record_data(conll_record)
        return conll_matrix
    else:
        raise NotImplementedError


"""
将record文件转化为ndarray与dimension
用RecordDataWrap包装以保证转换为ndarray后不会有歧义

- wrap_record_item_to_ndarray
    把record中的数据包装为ndarray。需要指定包装的层数。其中ndarray维度部分需要长度相同，warp部分会被包装。
"""


def wrap_record_item_to_ndarray(record_data, array_dimension_cnt: int):
    """
    根据提供的维度，将record中的数据部分包装为ndarray
    :param record_data: 序列型数据
    :param array_dimension_cnt: 合法的维度数目
        record的序列型数据中，其合法的维度数目代表按这个深度转换为ndarray，不存在list长度不一致的情况
    :return:
    """
    if array_dimension_cnt == 0:
        return RecordDataWrap(record_data)
    if array_dimension_cnt == 1:
        record_lst = list(RecordDataWrap(x) for x in record_data)
        return np.array(record_lst)
    else:
        return np.array([wrap_record_item_to_ndarray(x, array_dimension_cnt - 1) for x in record_data])


"""
ndarray of RecordDataWrap的处理函数

这里提供一些处理ndarray包装的RecordDataWrap的工具

- release_RecordDataWrap
    将RecordDataWrap内部的序列释放出来
"""


def release_RecordDataWrap(matrix: np.ndarray, level=-1):
    """
    将以RecordDataWrap为元素的matrix，每个RecordDataWrap里面的序列转到np.ndarray的维度

    level是转移多少维，level默认值为-1，默认情形是全部拆开，
    但是如果level非-1，那么只会从RecordDataWrap中释放level层，剩余的仍然会被包装为RecordDataWrap


    :param matrix: 以RecordDataWrap为元素的np.ndarray
    :param level: release的执行层数。当RecordDataWrap实例也为类似矩阵（比如整齐list）的数据时，level才有意义
    :return:
    """
    if level == 0:  # level=0意味着根本不需要改变
        return matrix
    elif level == -1:  # 处理全部展开的情形
        matrix_lst = matrix.tolist()
        matrix_lst = tools.convert_list_with_operation(lambda x: x(), matrix_lst)
        released_matrix = np.array(matrix_lst)
        return released_matrix

    # 对于level!=-1的情形，一层一层执行
    matrix_lst = matrix.tolist()
    matrix_lst = tools.convert_list_with_operation(lambda x: [RecordDataWrap(v) for v in x()], matrix_lst)
    released_matrix = np.array(matrix_lst)
    full_released_matrix = release_RecordDataWrap(released_matrix, level-1)
    return full_released_matrix


def classic_record_to_data_matrix(filepath: str):
    """
    classic格式，分别记录了三类信息以及其step
    - model
    - model_step
    - loss
    - loss_step
    - evaluator
    - evaluate_step
    这是六个key对应的value都是list，且长度相等。
    model、loss与evaluator的每个元素又都是dict，key是数据的类型，value就是数据的本体了
    本体可能有以下集中
    - 维度() 比如loss
    - 维度(n) 比如ner_loss与sentiment_loss，即loss的组成部分
    - 更高维度 比如模型返回的tensor之类的数据
    - str 比如预测目标
    - list 比如预测结果

    数据分析过程并不需要很强的类型保证与运行顺畅的保证，这个过程本身可能就是各种报错的，所以解耦合做的一般，也没有严格的保存格式限制，因此
    record的类型很难保证，因此数据分析时请谨慎确认数据类型与顺序，确保自己知道某一个操作的行为
    :param filepath:
    :return:
    """
    record = pickle.load(open(filepath, 'rb'))

    matrix_dict = {}
    dimension_dict = {}

    model_data = record['model']
    if len(model_data) != 0 and len(model_data[0].keys()):
        outer_key = list(model_data[0].keys())[0]
        data_names = list(list(model_data[0].values())[0].keys())
        for elem_name in data_names:
            data_lst = []
            for elem_data in model_data:
                data_lst.append(RecordDataWrap(elem_data[outer_key][elem_name]))
            matrix_dict[elem_name] = np.array(data_lst, dtype=object)
            dimension_dict[elem_name] = ['train_step']

    loss_data = record['loss']
    if len(loss_data) != 0 and len(loss_data[0].keys()):
        outer_key = list(loss_data[0].keys())[0]
        data_names = list(list(loss_data[0].values())[0].keys())
        for elem_name in data_names:
            data_lst = []
            for elem_data in loss_data:
                data_lst.append(RecordDataWrap(elem_data[outer_key][elem_name]))
            matrix_dict[elem_name] = np.array(data_lst, dtype=object)
            dimension_dict[elem_name] = ['train_step']

    evaluator_data = record['evaluator']
    if len(evaluator_data) != 0 and len(evaluator_data[0].keys()):
        outer_key = list(evaluator_data[0].keys())[0]
        data_names = list(list(evaluator_data[0].values())[0].keys())
        for elem_name in data_names:
            data_lst = []
            for elem_data in evaluator_data:
                data_lst.append(RecordDataWrap(elem_data[outer_key][elem_name]))
            matrix_dict[elem_name] = np.array(data_lst, dtype=object)
            dimension_dict[elem_name] = ['train_step']

    return matrix_dict, dimension_dict


def record_to_data_matrix(record_path: str):
    record = pickle.load(open(record_path, 'rb'))

    matrix_dict = {}

    model_data = record['model']
    if len(model_data) != 0:
        data_names = list(model_data[0].keys())
        for elem_name in data_names:
            data_lst = []
            for elem_data in model_data:
                data_lst.append(RecordDataWrap(elem_data[elem_name]))
            matrix_dict[elem_name] = np.array(data_lst, dtype=object)

    loss_data = record['loss']
    if len(loss_data) != 0:
        data_names = list(loss_data[0].keys())
        for elem_name in data_names:
            data_lst = []
            for elem_data in loss_data:
                data_lst.append(RecordDataWrap(elem_data[elem_name]))
            matrix_dict[elem_name] = np.array(data_lst, dtype=object)

    evaluator_data = record['evaluator']
    if len(evaluator_data) != 0:
        data_names = list(evaluator_data[0].keys())
        for elem_name in data_names:
            data_lst = []
            for elem_data in evaluator_data:
                data_lst.append(RecordDataWrap(elem_data[elem_name]))
            matrix_dict[elem_name] = np.array(data_lst, dtype=object)

    return matrix_dict


def write_record(record: MyRecordType, save_path: str, record_attr_v: Dict[str, Any] = None, record_attr_s: Set[str] = None):
    """
    将record数据写入到文件当中，按照属性来生成文件名，以点'.'做分隔
    :param record: record数据本体
    :param save_path: 保存该文件的路径
    :param record_attr_v: 应当写入文件名的属性的键值对
    :param record_attr_s: 应当写入文件名的属性的值
    :return:
    """
    if save_path[-1] != '/':
        save_path += '/'
    if record_attr_v is None:
        record_attr_v = {}
    if record_attr_s is None:
        record_attr_s = set()

    filename = 'record'
    for key, value in record_attr_v.items():
        filename += f'.{key}-{str(value)}'
    for elem in record_attr_s:
        filename += f'.{str(elem)}'

    save_name = save_path + filename
    pickle.dump(record, open(save_name, 'wb'))




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


"""
选择器部分

record的文件名包含了该record的相关属性
这部分的代码实现了一些非常简易的对属性进行筛选对函数
"""


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


if __name__ == '__main__':
    d = format_convert.conllner_to_lst('../data/NLP/SemEval/MultiCoNER/training_data/MIX_Code_mixed/mix_train.conll')
    d_record = build_record([d], ['ner_samples', 'sample'], [['samples', 'ner_sample']])
    write_record(d_record, '../data/analyze_cache/', record_attr_s={'MultiCoNER'})
