"""
Infer Tools

todolist:
infer structure
    multi layer list的完整形状
    dict结构的推断
    dict与list的复合结构的推断
    包含其他类型的dict与list的复合结构的推断
需要解决的问题
    复合结构的表示，通过typing吗？
"""
import typing

from type_def import *
import typing_utils


def infer_sequence_shape(data):
    """
    推断data作为list的形状
    一个合法的维度，其每个元素的长度必须相同。
    比如[[1, 2], [3, 4]]是(2, 2)
    而[[1, 2], [3, 4, 5]]只能是(2)
    保证长度一致，是为了能够包装为ndarray之类的类型，进行更复杂的矩阵操作
    :param data:
    :return:
    """
    if typing_utils.issubtype(type(data), typing.Sequence):
        outer_len = len(data)
        if outer_len == 0:
            return tuple([outer_len])
        inner_lens = list(infer_sequence_shape(d) for d in data)  # list of Tuple[int], len > 0
        if len(set(inner_lens)) == 1:
            return tuple([outer_len] + list(inner_lens[0]))
        else:
            return tuple([outer_len])
    else:
        return ()


# infer会需要返回一个容器的结构特点信息
# 用dict
#  {
#   "type": list, set, tuple, dict, List[int], ...
#   "feature": 序列的结构特点
#   "value": 能反映结构特点的一些值
#  }

def infer_sequence_of_int(int_seq: Sequence[int], set_limit=50):
    """
    分析一个int类型的序列的类型，可能是如下类型：
        类型      值
        相同      该值
       某几个值   set
       某个范围    Tuple[int, int]
    :param int_seq:
    :return:
    """
    # 可以直接使用与float相同的函数
    d = infer_sequence_of_float(int_seq, set_limit)
    d['type'] = 'List[int]'
    return d


def infer_sequence_of_float(float_seq: Sequence[int], set_limit=50):
    """
    分析一个float类型的序列的类型，可能是如下类型：
        类型      值
        相同      该值
       某几个值   set
       某个范围    Tuple[float, float]
    :param float_seq:
    :return:
    """
    float_set = set(float_seq)
    if len(float_set) == 1:
        return {
            "type": "List[float]",
            "feature": 'single',
            "value": list(float_seq)[0]
        }
    elif len(float_seq) <= set_limit:
        return {
            "type": "List[float]",
            "feature": 'set',
            "value": float_set
        }
    else:
        return {
            "type": "List[float]",
            "feature": 'range',
            "value": tuple([min(float_seq), max(float_seq)])
        }


def infer_sequence_of_str(str_seq: Sequence[str], set_limit=50):
    """
    分析一个str类型的序列的类型，可能是如下类型：
        类型      值
        相同      该值
       某几个值   set
    长度属于某个范围    Tuple[int, int]
    :param str_seq:
    :param set_limit:
    :return:
    """
    str_set = set(str_seq)
    if len(str_set) == 1:
        return {
            "type": "List[str]",
            "feature": 'single',
            "value": list(str_seq)[0]
        }
    elif len(str_seq) <= set_limit:
        return {
            "type": "List[str]",
            "feature": 'set',
            "value": str_set
        }
    else:
        lengths = list(len(x) for x in str_set)
        return {
            "type": "List[str]",
            "feature": 'range',
            "value": tuple([min(lengths), max(lengths)])
        }


def check_list_type(lst: list) -> str:
    """
    判断一个list所含有的所有元素的类型，如果一致则返回该类型，否则返回mix:
    - int
    - float
    - str
    - tuple
    - list
    - set
    - object 包含除上述几个类型之外的类型
    - mix 包含多种类型
    :param lst:
    :return:
    """
    type_set = set(type(x) for x in lst)
    if len(type_set) > 1:
        return 'mix'
    elif list(type_set)[0] in {int, float, str, tuple, list, set}:
        return str(list(type_set)[0]).split('\'')[1]
    else:
        return 'object'


def infer_sequence_of_list(lst_seq: Sequence[list], set_limit=50):
    """
    sequence of list的数据推测，很难直接来，只能通过一些指标描述
    - 长度
    - 类型 （int, float, str, tuple, list, set, object, mix）
    由于list的指标还是过多了，先采用简化的设计：
    长度分为以下三种
    - 1       value
    - 某几个值  set
    - 多个    最长、最短
    而类型由于数量不多，直接用set即可

    :param lst_seq:
    :param set_limit:
    :return:
    """
    list_lengths = list(len(x) for x in lst_seq)
    list_types = list(check_list_type(x) for x in lst_seq)
    length_set, type_set = set(list_lengths), set(list_types)

    if len(length_set) == 1:
        return {
            "type": "List[list]",
            "feature": 'single',
            "value": (list_lengths[0], type_set)
        }
    elif len(length_set) <= set_limit:
        return {
            "type": "List[list]",
            "feature": 'set',
            "value": (length_set, type_set)
        }
    else:
        return {
            "type": "List[list]",
            "feature": 'range',
            "value": (tuple([min(length_set), max(length_set)]), type_set)
        }


def infer_sequence_of_tuple(tuple_seq: Sequence[Tuple], set_limit=50):
    """
    原理上tuple是和list一样的，所以直接调用
    :param tuple_seq:
    :param set_limit:
    :return:
    """
    d = infer_sequence_of_list(tuple_seq, set_limit)
    d['type'] = 'List[tuple]'
    return d


def infer_sequence_of_set(set_seq: Sequence[set], set_limit=50):
    """
    远离上set也和list一样
    :param set_seq:
    :param set_limit:
    :return:
    """
    d = infer_sequence_of_set(set_seq, set_limit)
    d['type'] = 'List[set]'
    return d


def infer_sequence(seq: Sequence[Any], set_limit=50):
    """
    分析一个序列的类型
    :param seq:
    :param set_limit:
    :return:
    """
    seq_types = set(type(x) for x in seq)

    if len(seq_types) == 1:  # 单类型序列
        seq_type = list(seq_types)[0]
        if seq_type == int:
            return infer_sequence_of_int(seq, set_limit)
        elif seq_type == float:
            return infer_sequence_of_float(seq, set_limit)
        elif seq_type == str:
            return infer_sequence_of_str(seq, set_limit)
        elif seq_type == list:
            return infer_sequence_of_list(seq, set_limit)
        elif seq_type == set:
            return infer_sequence_of_set(seq, set_limit)
        elif seq_type == tuple:
            return infer_sequence_of_tuple(seq, set_limit)
        else:  # 其他python object。object无法分析，因此不提供feature与value
            return {
                "type": 'List[obj]',
                "feature": None,
                "value": None
            }
    else:  # 对于包含不止一种类型的list，也无法分析，因此不提供value和feature
        return {
            "type": 'List[multi]',
            "feature": None,
            "value": None
        }


def is_sequence_structure_similar(seq1: Sequence[Any], seq2: Sequence[Any]) -> bool:
    """
    判断两个sequence是否在结构上相似
    :param seq1:
    :param seq2:
    :return:
    """
    return False


def infer_dicts_pattern(dicts: List[dict]):
    """
    分析一组dict的模式
    :param dicts:
    :return:
    """
