from type_def import *
from settings import *
import numpy as np
import pickle
from utils import tools


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


def wrap_record_item_to_ndarray(record, array_dimension_cnt: int):
    if array_dimension_cnt == 0:
        return RecordDataWrap(record)
    if array_dimension_cnt == 1:
        record_lst = list(RecordDataWrap(x) for x in record)
        return np.array(record_lst)
    else:
        return np.array([wrap_record_item_to_ndarray(x, array_dimension_cnt - 1) for x in record])


def parse_record_data(record_dict: dict):
    """
    record文件中用于存放数据的dict，默认是record['data']
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


def read_record(filepath: str) -> Dict[str, dict]:
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


if __name__ == '__main__':
    record = pickle.load(open('../work/NER/record-2021_10_23_23_25_8', 'rb'))
    m = parse_record_data(record)
