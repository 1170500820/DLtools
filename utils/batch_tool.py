"""
utils.py 编写了一些方便数据处理的函数。
使用paddlepaddle
"""
import numpy as np
from type_def import *
import random
import torch


"""数据的划分相关
- randomize 将一系列列表同步随机化
- train_val_split 将一个列表按比例划分
- multi_train_val_split 将多个列表按比例同步划分
"""


def randomize(*lsts):
    """
    randomize([1,2,3], [4,5,6])
    打乱顺序
    :param lsts: iterables of iterables,
    :return:
    """
    zipped = list(zip(*lsts))
    random.shuffle(zipped)
    return list(zip(*zipped))


def train_val_split(lst, split_ratio=None):
    cnt = len(lst)
    bound = int(cnt * split_ratio)
    return lst[:bound], lst[bound: ]


def multi_train_val_split(*lst, split_ratio=None):
    """

    :param lst: [lst1, lst2, ...]
    :param split_ratio:
    :return: train_lst1
    """
    trains, vals = [], []
    for l in lst:
        train_l, val_l = train_val_split(l, split_ratio=split_ratio)
        trains.append(train_l)
        vals.append(val_l)
    return trains + vals


"""
对不同数据进行batch化的函数
- batchify_tensor 将tensor batch化，按照tensor中最大长度padding
- batchify_iterable 对于iterable的batch化
- batchify_dict_of_tensors
"""

def batchify_tensor(tensors_lst: [], bsz=None, keep_tail=False):
    """

    :param tensors_lst: [tensor1, tensor2, ...]. Tensor shape: (L x *)， 其中第一维必须相同
    :param bsz:
    :param keep_tail: if True, keep the last batch whose batch size might be lower than bsz
    :return: [tensor_batch1, tensor_batch2, ...]. Tensor batch shape: (bsz, max(L), *)
    """
    temp_tensors, tensor_batches = [], []
    for i, current_tensor in enumerate(tensors_lst):
        temp_tensors.append(tensors_lst[i])
        if len(temp_tensors) == bsz:
            batched_tensor = torch.stack(temp_tensors)
            tensor_batches.append(batched_tensor)
            temp_tensors = []
    if keep_tail and len(temp_tensors) != 0:
        tensor_batches.append(torch.stack(temp_tensors))
    return tensor_batches


def batchify_iterable(lst: [], bsz=None, keep_tail=False):
    """

    :param lst: [v1, v2, ...]
    :param bsz:
    :param keep_tail:
    :return: [[v1, v2, ..., v_bsz], ...]
    """
    temp_lst, lst_batches = [], []
    for i, v in enumerate(lst):
        temp_lst.append(v)
        if len(temp_lst) == bsz:
            lst_batches.append(temp_lst)
            temp_lst = []
    if keep_tail and len(temp_lst) != 0:
        lst_batches.append(temp_lst)
    return lst_batches


def batchify_ndarray1d(lst: List[np.ndarray]):
    """
    将一个均为一维的np.ndarray的list batch化
    :param lst:
    :param bsz:
    :param keep_tail:
    :return:
    """
    max_length = max(list(len(x) for x in lst))
    new_lst = []
    for elem in lst:
        add_length = max_length - len(elem)
        if add_length != 0:
            add_array = np.zeros(add_length)
            new_lst.append(np.concatenate([elem, add_array]))
        else:
            new_lst.append(elem)
    final_array = np.stack(new_lst)  # (bsz, max_length)
    return final_array


def batchify_dict_of_tensors(lst: [dict, ], bsz=None, keep_tail=False):
    """
    返回的仍然是dict的list
    每个dict的tensor现在是输入的bsz个dict中tensor的batch化
    :param lst: [dict1, dict2, ...]. dict : {key1: tensor1, key2, tensor2, ...}, every dict should have same keys
    :param bsz:
    :param keep_tail:
    :return: [batched_dict1, batched_dict2, ...]. batched_dict: {key1: batched_tensor1, key2: batched_tensor2, ...}
    """
    dict_tensors = {}
    for d in lst:
        for key, value in d.items():
            if key not in dict_tensors:
                dict_tensors[key] = []
            dict_tensors[key].append(value.squeeze(dim=0))
    dict_batched_tensors = {}
    batch_cnt = 0
    for key, value in dict_tensors.items():
        dict_batched_tensors[key] = batchify_tensor(value, bsz=bsz, keep_tail=keep_tail)
        batch_cnt = len(dict_batched_tensors[key])
    result_dicts = []
    for i in range(batch_cnt):
        cur_dict = {}
        for key, value in dict_batched_tensors.items():
            cur_dict[key] = value[i]
        result_dicts.append(cur_dict)
    return result_dicts


def batchify(*lsts, bsz=None, lst_types=None):
    """

    :param bsz:
    :param lsts: list of list.
    :param lst_types: list of batchify types, in {'tensor', 'iterable'， ‘dict_tensor}.
        tensors need paddings, iterables do not
        if not provided, all iterables in default
    :return: list of batchified list
    """
    if lst_types is None:
        lst_types = ['iterable'] * len(lsts)
    function_map = {
        'iterable': batchify_iterable,
        'tensor': batchify_tensor,
        'dict_tensor': batchify_dict_of_tensors,

        # in case i messed up
        'iterables': batchify_iterable,
        'tensors': batchify_tensor,
        'dict_tensors': batchify_dict_of_tensors
    }
    results = list(map(lambda x: function_map[x[1]](lsts[x[0]], bsz=bsz), enumerate(lst_types)))
    return results


def find_matches(content: str, tokenized_seq: [str, ]) -> {}:
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


def as_dicts(lsts, names):
    """
    lst lengths must be equal
    :param lsts: [lst1, lst2, ..., lst_n]
    :param names: [name1, name2, ..., name_n]
    :return: [{name1: lst1[0], ..., name_n: lst_n[0]}, ...]
    """
    results = []
    for i in range(len(lsts[0])):
        cur_dict = {}
        for idx, idx_name in enumerate(names):
            cur_dict.update({idx_name: lsts[idx][i]})
        results.append(cur_dict)
    return results


def simple_collate_fn(lst: List[Dict[str, Any]]):
    """
    对于tensor，直接stack, 默认是对axis=0进行stack，所以tensor的size(0)必须相等
    对于np.ndarray，也是直接stack。注意不会把ndarray转化为tensor，需要手动转化
    对于其他的，都堆叠为list
    :param lst:
    :return:
    """
    lst_keys = list(lst[0].keys())
    result_dict = {}
    for k in lst_keys:
        # 如果是tensor，就stack
        if type(lst[0][k]) == torch.Tensor:
            tensor_lst = []
            for l in lst:
                tensor_lst.append(l[k])
            result_dict[k] = torch.stack(tensor_lst)
        elif type(lst[0][k]) == np.ndarray:
            array_lst = []
            for l in lst:
                array_lst.append(l[k])
            result_dict[k] = np.stack(array_lst)
        else:
            object_lst = []
            for l in lst:
                object_lst.append(l[k])
            result_dict[k] = object_lst
    return result_dict


def argument_span_determination(binary_start: IntList, binary_end: IntList, prob_start: ProbList, prob_end: ProbList) -> SpanList:
    """
    来自paper: Exploring Pre-trained Language Models for Event Extraction and Generation
    Algorithm 1
    :param binary_start:
    :param binary_end:
    :param prob_start:
    :param prob_end:
    :return:
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


def regular_collate_fn(lst: List[Any]):
    """
    以通用为目的的collate function。
    只支持pytorch tensor，没有加入对paddle tensor的支持，因为不想引用过多的包。
    todo 希望做成插件式的

    - 以lst的长度为batch size
    - tensor
        tensor除了第一维之外，其他维度必须相同。batch化时，使用pad_sequence对第一维进行padding
    - list
    :param lst: Any可以包含list，dict，str等。每一个元素必须具有相同的类型，
    :return:
    """

