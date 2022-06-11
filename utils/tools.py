# todo 找找还有没有能够写成tools的
import inspect
import typing

from type_def import *
import typing_utils

import json
import numpy as np

from transformers import BertTokenizerFast
import pandas as pd
import torch
import functools



"""
Dict Tools
"""


class dotdict(dict):
    """
    这个奇妙的设计来自
    https://github.com/sonack/PatternRecognitionExps/blob/65b191b5409223ba317a39e28b4664846e916973/%E5%AE%9E%E9%AA%8C4-MNIST%E5%88%86%E7%B1%BB%E8%80%83%E8%AF%95/main.py#L28
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def transpose_dict_of_list(dict_of_list: Dict[str, list]):
    """
    将一个包含多个长度相同的list为value，拆解为list，list中的每个dict只包含对应key的list的一个元素
    :param dict_of_list: key值为str，value为长度相同的list。key值为任意hashable时，也能正常运转，但是尽量保持dict的key值类型单一
    :return:
    """
    keys = list(dict_of_list.keys())
    result_lst = []
    value_l = len(dict_of_list[keys[0]])

    for idx in range(value_l):
        cur_dict = {}
        for elem_key in keys:
            cur_dict[elem_key] = dict_of_list[elem_key][idx]
        result_lst.append(cur_dict)
    return result_lst


def transpose_list_of_dict(list_of_dict: List[Dict[str, Any]]):
    """
    将一个包含相同key值的list的dict，转化为一个dict，其每个key所对应的value是之前list中每一个dict对应key的元素所组合成的list
    :param list_of_dict:
    :return:
    """
    keys = list(list_of_dict[0].keys())
    result_dict = {x: [] for x in keys}
    for elem_dict in list_of_dict:
        for key, value in elem_dict.items():
            result_dict[key].append(value)
    return result_dict


def split_dict(d: dict, key_lst: StrList, keep_origin=False):
    """
    将一个dict分成两个，一个是key_lst中的key组成的，另一个是不包含在key_lst中的
    :param d:
    :param key_lst:
    :param keep_origin: 是否保留原dict，如果False，那么会真的把原dict拆开
    :return:
    """

    new_dict = {}
    for elem_key in key_lst:
        if elem_key not in d:
            raise Exception(f'[split_dict]{elem_key}不在输入的dict当中！')
        if keep_origin:
            new_dict[elem_key] = d[elem_key]
        else:
            new_dict[elem_key] = d.pop(elem_key)
    return new_dict, d


def flattern_dict(d: dict, order=True):
    """
    把一个dict的value展开为list

    :param d:
    :param order: 是否保持展开的顺序一致
        若为是，则会将value按照key词典序排序
    :return:
    """
    if not order:
        return list(d.values())
    else:
        keys_and_values = list(d.items())
        keys_and_values.sort()
        sorted_values = list(x[1] for x in keys_and_values)
        return sorted_values


def modify_key_of_dict(d: Dict[str, Any], modify_func: Callable[[str], str]):
    """
    对于一个dict，将其所有key修改
    :param d: 被修改的词典。返回的是新建的dict
    :param modify_func: 修改key的function，对每一个key值遍历调用
    :return:
    """
    new_dict = {}
    for key, value in d.items():
        new_dict[modify_func(key)] = value
    return new_dict


"""
List Tools
"""


def transpose_double_list(double_list: List[List[Any]]):
    """
    将一个双层list转置
    要求内层list具有相同的len
    :param double_list: len(double_list[*])应当相同
    :return:
    """
    if len(set(map(len, double_list))) != 1:
        raise Exception('[transpose_double_list]内层list的长度应当相同')
    inner_length = len(double_list[0])
    outer_length = len(double_list)
    transposed = []
    for elem in range(inner_length):
        transposed.append([])
    for elem_outer in double_list:
        for inner_idx, elem_inner in enumerate(elem_outer):
            transposed[inner_idx].append(elem_inner)
    return transposed


def split_list_with_ratio(lst: list, ratio: float):
    """
    按照ratio返回两个列表
    :param lst:
    :param ratio: 0 <= ratio <= 1
    :return:
    """
    if ratio < 0 or ratio > 1:
        raise Exception('[split_list_with_ratio]ratio中值不在[0, 1]范围内！')
    if ratio == 0:  # ratio输入0和1但，真让人摸不着头脑
        return [], lst
    elif ratio == 1:
        return lst, []
    else:
        l = len(lst)
        left = int(l * ratio)
        left_lst = lst[:left]
        right_lst = lst[left:]
        return left_lst, right_lst


def convert_list_with_operation(operation: Callable, lst: list):
    """
    对于lst但每个元素，调用operation，使用得到但返回值代替该元素
    这个函数会返回新的list，但是形状会保持一致
    当遇到list[not list]时，就会对该list内的每个元素执行操作。因此如果operation的输入是list，需要
    转换为tuple，或者自己

    不要瞎玩。
    :param operation:
    :param lst: 可以是任意形状，任意多层list多包装,但是
        1，每个list中含有多元素要么全是list，要么不包含list
    :return:
    """
    if len(lst) == 0:
        return lst
    elif isinstance(lst, list) and not isinstance(lst[0], list):
        new_lst = list(map(operation, lst))
        return new_lst
    else:
        result_lst = []
        for elem in lst:
            converted_lst = convert_list_with_operation(operation, elem)
            result_lst.append(converted_lst)
        return result_lst


"""
Read Tools
"""


def read_json_lines(filename: str):
    return list(map(json.loads, open(filename, 'r', encoding='utf-8').read().strip().split('\n')))


def read_tsv(filename: str):
    lines = open(filename, 'r').read().strip().split('\n')
    content = []
    for elem_line in lines:
        content.append(elem_line.split('\t'))
    return content


def read_csv_as_datadict(filename: str) -> List[Dict[str, Any]]:
    """
    读取一个csv文件，转化为data_dict格式
    即，List[dict], dict中key为csv但tag，value为对应值
    :param filename:
    :return:
    """
    d = pd.read_csv(filename).to_dict()
    _tags = list(d.keys())
    tags = []
    for elem in _tags:
        if 'Unnamed' not in elem:
            tags.append(elem)
    data_dict = {}
    for k in tags:
        df_k_data = list(d[k].items())
        df_k_data.sort()
        if df_k_data[-1][0] + 1 != len(df_k_data):
            raise Exception(f'[read_csv_as_datadict]DataFrame的{k}字段序号不连续！')
        data_dict[k] = [x[1] for x in df_k_data]
    data_dicts = transpose_dict_of_list(data_dict)
    return data_dicts


"""
Collate and Compose
"""


def batchify_ndarray_1d(array_lst: Sequence[np.ndarray], padding=0):
    """
    将一组一维的ndarray组合成一个矩阵。对于不同长度的元素，用padding代替
    :param array_lst: 一组需要batchify的numpy数组
    :param padding: 用于补全的值
    :return:
    """
    array_lst = list(array_lst)
    array_lst_size = len(array_lst)
    lengths = np.array(list(map(lambda x: x.shape[0], array_lst)))
    max_length = max(lengths)
    pad_lengths = max_length - lengths
    for i in range(array_lst_size):
        array_lst[i] = np.pad(array_lst[i], (0, pad_lengths[i]), mode='constant', constant_values=padding)
    result = np.stack(array_lst)  # (bsz, max_length)

    # check result
    if result.shape[0] != array_lst_size or result.shape[1] != max_length:
        raise Exception(f'[batchify_ndarray_1d]result array shape error, should be '
                        f'[{array_lst_size}, {max_length}], get [{result.shape[0]}, {result.shape[1]}]')
    return result


def batchify_ndarray(array_lst: Sequence[np.ndarray], padding=0):
    """
    将一组一维的ndarray组合成一个矩阵。对于不同长度的元素，用padding代替
    array: (*, l1, l2, ...)
    :param array_lst: 多个具有相同维数的ndarray。除了第一维之外，其他维度必须相同
    :param padding: pad所用的值
    :return:
    """
    # 一些转化，获得一些方便的变量
    array_lst = list(array_lst)
    array_lst_size = len(array_lst)
    lengths = np.array(list(map(lambda x: x.shape[0], array_lst)))
    max_length = max(lengths)
    pad_lengths = max_length - lengths

    # pre check list
    #   - 维数相同
    #   - 除第一维之外的维度相同
    shape_lst = list(map(lambda x: tuple(x.shape[1:]), array_lst))
    shape_len_set = set(map(lambda x: len(x), shape_lst))
    if len(shape_len_set) != 1:
        raise Exception(f'[batchify_ndarray]ndarray应当具有相同的维数！shape_lst:{shape_lst}')
    shape_set = set(shape_lst)
    if len(shape_set) != 1:
        raise Exception(f'[batchify_ndarray]ndarray每一维应当具有相同的维度！shape_lst:{shape_lst}')

    # pad and stack
    res_shape = array_lst[0].shape[1:]  # [l1, l2, ...] 需要补全的0的形状。stack上去的
    pad_array = np.full(res_shape, padding)  # (l1, l2, ...) of full zero
    for i in range(array_lst_size):
        pad_len = pad_lengths[i]
        to_pad_array = array_lst[i]  # (*, l1, l2, ...)
        if pad_len == 0:
            padded_array = to_pad_array
        else:
            cur_pad_array = np.stack([pad_array] * pad_len)  # (pad_len, l1, l2, ...)
            padded_array = np.concatenate([to_pad_array, cur_pad_array])  # (max(*), l1, l2, ...)
        array_lst[i] = padded_array
    pad_result = np.stack(array_lst)  # (bsz, max(*), l1, l2, ...)

    # post check list
    #   - res shape correct
    #   - bsz correct
    result_shape = pad_result.shape
    if result_shape[0] != array_lst_size:
        raise Exception(f'[batchify_ndarray]产生的结果的第一维的值不对！expect:{array_lst_size}, actually:{res_shape[0]}')
    if tuple(result_shape[2:]) != tuple(res_shape) or result_shape[1] != max_length:
        raise Exception(f'[batchify_ndarray]补全结果不对！expect:{res_shape}, actually:{result_shape}')

    return pad_result  # (bsz, max(*), l1, l2, ...)


def batchify_list_1d(lst_of_lst: List[list], padding=0):
    lst_of_array = [np.array(x) for x in lst_of_lst]
    result = batchify_ndarray_1d(lst_of_array, padding=padding)
    result_lst = result.tolist()
    return result_lst


def batchify_dict(lst_of_dict: List[Dict[str, Any]], padding=0):
    """
    对于一组dict进行batch化。
    分别对每一个key值进行batch化，返回的dict的key值仍然不变
    :param lst_of_dict:
    :param padding:
    :return:
    """
    keys = list(lst_of_dict[0].keys())
    result_dict = {}
    for elem_key in keys:
        value_lst = [np.array(x[elem_key]) for x in lst_of_dict]
        result_dict[elem_key] = batchify_ndarray_1d(np.array(value_lst), padding=padding)
    return result_dict


def convert_dict_to_cuda(dict_of_tensor: Dict[str, Any]):
    """
    将一个含有tensor的dict中的tensor转换为cuda
    :param dict_of_tensor:
    :return:
    """
    for key, value in dict_of_tensor.items():
        if isinstance(value, torch.Tensor):
            dict_of_tensor[key] = value.cuda()
    return dict_of_tensor


def padded_stack(list_of_tensor: List[torch.Tensor], pad_value: float = 0):
    """
    在长度较短的tensor后面pad，然后stack
    :param list_of_tensor: (*, x, y, ...) 第一维度可以不一样，其余维度必须一样
    :param pad_value:
    :return:
    """
    shapes = list(list(x.shape) for x in list_of_tensor)
    shape_lengths = set(len(x) for x in shapes)
    if len(shape_lengths) > 1:
        raise Exception(f'[padded_stack]list_of_tensor中所有tensor的维度数应该相同。得到的维度数：{shape_lengths}')
    # todo


"""
Task Oriented - Modified Data in Dict
此处的部分function是通用的，也有一些function是针对由Dict组织的数据进行处理的。
    - data clean
        -- replace chars
        -- replace chars in dict
    - tokenize tools
        -- naive bert tokenizer
"""


def replace_chars(content: str, replace_dict: Dict[str, str] = None):
    """
    根据replace_dict，将content中的某些字符替换为其他字符
    :param content: 将被处理的字符串
    :param replace_dict: key为需要替换的字符，value为对应的替代字符。默认是将空格替换为下划线
    :return:
    """
    if replace_dict is None:
        replace_dict = {' ': '_'}
    for key, value in replace_dict.items():
        content = content.replace(key, value)
    return content


def replace_chars_in_dict(data_dict: Dict[str, Any], key: str = 'content', replace_dict: Dict[str, str] = None):
    """
    replace_chars的in-dict版本。
    :param data_dict: 需要进行修改的dict
    :param key: 需要修改的字段的key
    :param replace_dict: 替换字符与代替字符
    :return:
    """
    data_dict[key] = replace_chars(data_dict[key], replace_dict)
    return [data_dict]


def find_matches(content: str, tokenized_seq: [str, ]) -> Tuple[dict, dict]:
    """
    为了简化处理，默认已经将空格替换为了下划线。
    因为有空格的情况实在太难处理了。太难了。太难了。
    todo 实现有空格情况
    :param content:
    :param tokenized_seq:
    :return:
    """
    assert ' ' not in content

    token2origin = {}
    origin2token = {}
    # 先给每一个位置开辟
    for i in range(len(tokenized_seq)):
        token2origin.update({i: []})
    for i in range(len(content)):
        origin2token.update({i: -1})

    reach = 0
    unk_last = False
    unk_count = 0
    for i, token in enumerate(tokenized_seq):
        ltoken = len(token)
        first_token = token[0]
        if token == '[UNK]':
            unk_count += 1
            if not unk_last:
                unk_last = True
            continue
        elif token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
        elif token[:2] == '##':
            ltoken = len(token) - 2
            first_token = token[2]
        if unk_last:
            unk_last = False
            current_reach = content.find(token[0], reach)
            token2origin[i - 1] = list(range(reach, current_reach))
            for k in range(unk_count):
                pass
            reach = current_reach
            unk_count = 0

        position = content.find(first_token, reach)
        token2origin[i] = list(range(position, position + ltoken))
        reach = position + ltoken

        assert content[position + ltoken - 1] == token[-1], str([content, position, ltoken, token, i])

    for key, value in token2origin.items():
        for v in value:
            origin2token[v] = key

    return token2origin, origin2token


class bert_tokenizer:
    def __init__(self, max_len=256, plm_path='bert-base-chinese'):
        print('this version of bert_tokenizer is about to deprecated')
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
        self.max_len = max_len

    def __call__(self, input_lst: List[Union[str,List[str]]]):
        """
        每个tokenize结果包括
        - token
        - input_ids
        - token_type_ids
        - attention_mask
        - offset_mapping
        :param input_lst:
        :return:
        """
        results = []
        for elem_input in input_lst:
            tokenized = self.tokenizer(
                elem_input,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_offsets_mapping=True)
            token_seq = self.tokenizer.convert_ids_to_tokens(tokenized['input_ids'])
            tokenized['token'] = token_seq
            results.append(dict(tokenized))
        return results


def get_word_occurrences_in_sentence(sentence: str, word: str):
    """
    计算一个word在一个sentence中的每一个出现的span:(第一个char的index，最后一个char的index)
    如果word只有一个char，那么span[0] == span[1]
    本函数要求word至少在sentence中出现在一次，否则会抛出异常
    :param sentence:
    :param word:
    :return:
    """
    if word == '':
        raise Exception(f'[get_word_occurrences_in_sentence]出现了空word！')
    word_len = len(word)
    starts = [i for i in range(len(sentence)) if sentence.startswith(word, i)]
    spans = list((x, x + word_len - 1) for x in starts)
    if len(spans) == 0:
        raise Exception(f'[get_word_occurrences_in_sentence]No occurrences of word:[{word}] found in sentence:[{sentence}]')

    # check span
    for elem_span in spans:
        if elem_span[0] < 0 or elem_span[1] < 0 or elem_span[1] >= len(sentence) or elem_span[0] > len(sentence):
            raise Exception(f'[get_word_occurrences_in_sentence]')
    return spans


def argument_span_determination(binary_start: IntList, binary_end: IntList, prob_start: ProbList, prob_end: ProbList) -> SpanList:
    """
    来自paper: Exploring Pre-trained Language Models for Event Extraction and Generation
    Algorithm 1
    :param binary_start:
    :param binary_end:
    :param prob_start:
    :param prob_end:
    :return: List[Tuple[int, int]]
    """
    a_s, a_e = -1, -1
    state = 1
    # state
    #   1 - 在外面
    #   2 - 遇到了一个start
    #   3 - 在start之后遇到了一个end，看看还有没有更长的end
    spans = []
    seq_l = len(binary_start)
    for i in range(seq_l):
        if state == 1 and binary_start[i] == 1:
            a_s = i
            state = 2
        elif state == 2:
            # 什么叫new start?
            if binary_start[i] == 1:
                if prob_start[i] > prob_start[a_s]:
                    a_s = i
            if binary_end[i] == 1:
                a_e = i
                state = 3
        elif state == 3:
            if binary_end[i] == 1:
                if prob_end[i] > prob_end[a_e]:
                    a_e = i
            if binary_start[i] == 1:
                spans.append((a_s, a_e))
                a_s, a_e = i, -1
                state = 2
    if state == 3:  # todo 这个debug有问题吗？
        spans.append((a_s, a_e))
    return spans


# todo 从模型预测出的tensor转化到易读的字符串或者json格式

class FocalWeight:
    """
    把所有weight都放在一个class里面是很垃圾的结构
    总之先把focal weight拆出来

    用于计算focal weight
    """
    def __init__(self, alpha=0.3, gamma=2):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, logits: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor = None):
        """
        输出logits对于ground_truth的focal loss权值矩阵
        实际上logits和ground_truth的shape一样就可以计算，
        :param logits: (~)
        :param ground_truth: (~)
        :param mask: (~)
        :return: (~)
        """
        logits_clone = logits.detach().clone()
        ground_truth_clone = ground_truth.detach().clone()
        pt = (1 - logits_clone) * ground_truth_clone + logits_clone * (1 - ground_truth_clone)
        focal_weight = (self.alpha * ground_truth_clone + (1 - self.alpha) * (1 - ground_truth_clone)) * torch.pow(pt, self.gamma)
        return focal_weight


class PosPrefWeight:
    def __init__(self, pref=2):
        self.pref = pref

    def __call__(self, ground_truth: torch.Tensor):
        """

        :param ground_truth: (bsz, n_label)
        :return:
        """
        gt_copy = ground_truth.clone().detach().cpu()
        gt_mask = torch.ones(gt_copy.shape) - gt_copy
        gt_weight = gt_copy * self.pref + gt_mask
        return gt_weight.clone().detach()

"""
In-Dict and Sequence Execute functions

将对应某个任务的多个输入数据用一个dict保存

类型定义
    - InDict
        约定为只包含str作为key的dict。一个dict可以保存对应某个任务输入的多个数据
    - InDictList
        多个InDict
    - ElemOperator
        对InDict的处理函数。
        输出为[]则表示删除；输出为[InDict]表示修改，返回一个修改后的InDict；输出为[InDict1, InDict2, ...]代表生成多个
"""

InDict = Dict[str, Any]
InDictList = List[InDict]
ElemOperator = Callable[[InDict], InDictList]


def map_operation_to_list_elem(operation: ElemOperator, lst: InDictList) -> InDictList:
    """

    :param operation:
    :param lst:
    :return:
    """
    result_lst = []
    for elem in lst:
        result_lst += operation(elem)
    return result_lst


# 下面是由map_operation_to_list_elem衍生出的function

"""
InDict粒度
只要一个操作不需要知道InDictList的全局信息，对每一个InDict同等的进行，那么这个操作就是InDict粒度的。可以很容易和InDictList粒度的操作区分。
- 对key的改、删、增操作
    - list_of_dict_modify
    - list_of_dict_add
    - list_of_dict_delete
"""


def list_of_dict_modify(operation: Callable[[Any], Any], lst: InDictList, keyname: str = None) -> InDictList:
    """
    对于每一个dict，使用operation接收其中的某个key所对应的value，返回一个修改好的新value替换旧value
    :param keyname: 如果为None，则通过inspect.getfullargspec获得opeartion所需要的输入参数（排除self后应当只有一个，否则报错）
    :param operation: 接收value，返回修改后的value。不会对输入和输出的类型做约束，但要求输入和输出数量都为1
    :param lst:
    :return:
    """
    if keyname is None:
        operation_params = list(inspect.getfullargspec(operation)[0])
        if 'self' in operation_params:
            operation_params.remove('self')
        if len(operation_params) != 1:
            raise Exception('[list_of_dict_modify]operation函数的参数错误！')
        keyname = operation_params[0]

    def wrap_operation(data_dict: Dict[str, Any]):
        v_input = data_dict[keyname]
        v_result = operation(v_input)
        data_dict[keyname] = v_result
        return [data_dict]
    result = map_operation_to_list_elem(wrap_operation, lst)
    return result


def list_of_dict_add(add_operation: Callable, lst: InDictList, return_name: StrList, keynames: str = None) -> InDictList:
    """
    取data_dict中的几个value作为参数，输出一些新的value
    根据return_name给定的key值，这些新的value将被插入到data_dict当中
    :param add_operation: 需要谨慎选择返回值的方式
        首先返回值不能是None
        如果返回的是除tuple以外的任何值，哪怕是list，都会被看作一个value
        如果返回的是tuple而return_name只包含一个str，那么这个tuple会被看作一个value，否则
        tuple会被解包，当作多个value，与return_name的key值对于
    :param lst:
    :param return_name: 必须要明确指定返回值所要使用的key值
    :param keynames:
    :return:
    """
    if keynames is None:
        operation_params = list(inspect.getfullargspec(add_operation)[0])
        if 'self' in operation_params:
            operation_params.remove('self')
        if len(operation_params) == 0:
            raise Exception('[list_of_dict_filter]bool_operation不包含可用参数！')
        keynames = operation_params

    def wrap_add_operation(data_dict: Dict[str, Any]):
        _, input_dict = split_dict(data_dict, keynames, keep_origin=True)
        result = add_operation(**input_dict)
        # 对输出进行分情况
        if len(return_name) == 1:
            data_dict[return_name[0]] = result
        elif isinstance(result, Tuple) and len(return_name) == len(result):
            for (n, r) in zip(return_name, result):
                data_dict[n] = r
        else:
            raise Exception('[list_of_dict_add]返回值与给定签名匹配失败！')
        return [data_dict]

    res = map_operation_to_list_elem(wrap_add_operation, lst)

    return res


def list_of_dict_delete(lst: InDictList, delete_names: StrList) -> InDictList:
    """
    删除某些key，
    :param lst:
    :param delete_names:
    :return:
    """
    new_lst = []
    for elem in lst:
        new_lst.append(split_dict(elem, delete_names)[0])
    return new_lst


"""
InDictList粒度
"""


def list_of_dict_filter(bool_operation: Callable, lst: InDictList, keynames: str = None) -> InDictList:
    """
    这里不要求bool_operation只包含单个参数，因为有可能要根据不同key之间的关系来决定是否删除
    所以inspect.getfullargspec之后只需要检查参数个数是否为0即可
    :param keynames:
    :param bool_operation: 返回值为布尔值的函数。如果返回False，则删除这一个data_dict。否则
    :param lst:
    :return:
    """
    if keynames is None:
        operation_params = list(inspect.getfullargspec(bool_operation)[0])
        if 'self' in operation_params:
            operation_params.remove('self')
        if len(operation_params) == 0:
            raise Exception('[list_of_dict_filter]bool_operation不包含可用参数！')
        keynames = operation_params

    def wrap_bool_operation(data_dict: Dict[str, Any]):
        _, input_dict = split_dict(data_dict, keynames, keep_origin=True)
        if bool_operation(**input_dict):
            return [data_dict]
        else:
            return []
    result = map_operation_to_list_elem(wrap_bool_operation, lst)

    return result


def list_of_dict_groupby(groupby_operation: Callable[[InDict], Hashable], lst: InDictList) -> Dict[Hashable, InDictList]:
    """
    groupby_operation接收一个InDict，输出一个Hashable。这个Hashable将被用于作为group的键值

    :param groupby_operation:
    :param lst:
    :return:
        {
        Hashable1: InDictList,
        Hashable2: InDictList,
        ...
        }
    """
    group_dict = {}
    for elem_indict in lst:
        h = groupby_operation(elem_indict)
        if h not in group_dict:
            group_dict[h] = [elem_indict]
        else:
            group_dict[h].append(elem_indict)
    return group_dict


"""
In-Dict wrapped functions

"""


"""
todo function list

1, 划分一个list
    按返回值，划分为两个不同的list。可以是bool，true第一个list，false第二个list
    也可以是实值，直接按照结果的不同划分list（无序或者按实值序）
    应用点：将mentions分为trigger与其他
2，infertools
    见本py文件的infer部分
"""