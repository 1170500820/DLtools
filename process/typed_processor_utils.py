from process.data_processor_type import *
import json
from transformers import BertTokenizer
import numpy as np


"""
File Reader

    - tsv
    - json
    - csv not implemented!
"""


class TSV_Reader(TypedDataProcessor):
    def __init__(self):
        super(TSV_Reader, self).__init__('str|filepath', 'List[List[str]]|tsv_content')

    def process(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        filepath = data_dict['str|filepath']
        lines = open(filepath, 'r').read().strip().split('\n')
        content = []
        for elem_line in lines:
            content.append(elem_line.split('\t'))
        return {
            "List[List[str]]|tsv_content": content
        }


class Json_Reader(TypedDataProcessor):
    def __init__(self):
        super(Json_Reader, self).__init__(['str|json_path'], ['List[dict]|json_content'])

    def process(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        filepath = data_dict['str|json_path']
        lines = open(filepath, 'r', encoding='utf-8').read().strip().split('\n')
        dicts = list(map(json.loads, lines))
        return {
            "List[dict]|json_content": dicts
        }


# todo csv reader?


"""
Param Management

    - KeyFilter 过滤不需要的参数
"""


class KeyFilter(TypedDataProcessor):
    def __init__(self, key_lst: List[str], all_input: bool = False):
        super(KeyFilter, self).__init__(key_lst, all_input=all_input)

    def process(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        return data_dict


class KeyEater(TypedDataProcessor):
    """
    顾名思义，就是把这些key吃掉

    输入key，不产生任何输出
    """
    def __init__(self, key_lst: List[str]):
        super(KeyEater, self).__init__(key_lst, keep=True)

    def process(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {}


class KeyNameChanger(TypedDataProcessor):
    def __init__(self, output_name):
        super(KeyNameChanger, self).__init__(['temp'], output_name)

    def process(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        d = list(data_dict.values())[0]
        return {
            self.get_output_keys()[0]: d
        }

# todo 更多层的transpose


"""
Iterable Tools

"""


class DoubleListTranspose(TypedDataProcessor):
    """
    将一个双层list转置
    """
    def __init__(self):
        super(DoubleListTranspose, self).__init__('List[List[Any]]|double_list', "List[List[Any]]|transposed")

    def process(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        double_list = data_dict['List[List[Any]]|double_list']
        inner_length = len(double_list[0])
        outer_length = len(double_list)
        transposed = []
        for elem in range(inner_length):
            transposed.append([])
        for elem_outer in double_list:
            for inner_idx, elem_inner in enumerate(elem_outer):
                transposed[inner_idx].append(elem_inner)
        return {
            "List[List[Any]]|transposed": transposed
        }


# converted
class ListOfDictTranspose(TypedDataProcessor):
    def __init__(self):
        super(ListOfDictTranspose, self).__init__('List[Dict[str,Any]]|list_of_dict', "Dict[str,List[Any]]|dict_of_list")

    def process(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        list_of_dict = data_dict['List[Dict[str,Any]]|list_of_dict']
        keys = list(list_of_dict[0].keys())
        result_dict = {x: [] for x in keys}
        for elem_dict in list_of_dict:
            for key, value in elem_dict.items():
                result_dict[key].append(value)
        return {
            self.get_output_keys()[0]: result_dict
        }


# converted
class DictOfListTranspose(TypedDataProcessor):
    def __init__(self):
        super(DictOfListTranspose, self).__init__("Dict[str,list]|dict_of_list", "List[Dict[str,Any]]|list_of_dict")

    def process(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        dict_of_list = data_dict["Dict[str,list]|dict_of_list"]
        keys = list(dict_of_list.keys())
        result_lst = []
        value_l = len(dict_of_list[keys[0]])

        for idx in range(value_l):
            cur_dict = {}
            for elem_key in keys:
                cur_dict[elem_key] = dict_of_list[elem_key][idx]
            result_lst.append(cur_dict)
        return {
            self.get_output_keys()[0]: result_lst
        }


class List2Dict(TypedDataProcessor):
    """
    对于一个List，将里面的内容用dict_keys对应上，然后存储成dict
    """
    def __init__(self, dict_keys: List[str]):
        super(List2Dict, self).__init__("List[Any]|list", "Dict[str,Any]|dict")
        self.dict_keys = dict_keys

    def process(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        lst = data_dict['List[Any]|list']
        if len(lst) != len(self.dict_keys):
            raise Exception('分配的dict_keys与list无法对应！')

        result_dict = {}
        for (elem_key, elem_lst) in zip(self.dict_keys, lst):
            result_dict[elem_key] = elem_lst
        return {
            "Dict[str,Any]|dict": result_dict
        }


class Dict2List(TypedDataProcessor):
    def __init__(self, key_lst: List[str]):
        self.key_lst = key_lst
        super(Dict2List, self).__init__(key_lst, 'List[Dict[str,Any]]|list_of_dict')

    def process(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        lsts = []
        for elem_value in data_dict.values():
            lsts.append(elem_value)
        length = len(lsts[0])
        list_of_dict = []
        for i in range(length):
            new_dict = {}
            for idx, key in enumerate(data_dict.keys()):
                new_dict[key] = lsts[idx][i]
            list_of_dict.append(new_dict)
        return {
            self.get_output_keys()[0]: list_of_dict
        }


class GroupBy(TypedDataProcessor):
    """
    对于一个list中的每个element，输入给group_function
    group_function输入一个hashable
    最后输出一个dict，key为hashable object，value为list
    """
    def __init__(self, group_function):
        self.group_func = group_function
        super(GroupBy, self).__init__("List[Any]|lst")

    def process(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        grouped_dict = {}
        lst = data_dict['List[Any]|lst']
        for elem in lst:
            group_key = self.group_func(elem)
            if group_key in grouped_dict:
                grouped_dict[group_key].append(elem)
            else:
                grouped_dict[group_key] = [elem]
        return {
            "Dict[Any,list]|grouped": grouped_dict
        }


class ReleaseDict(TypedDataProcessor):
    """
    把一个dict内的参数直接释放出来
    """
    def __init__(self):
        super(ReleaseDict, self).__init__("Dict[str,Any]|dict")

    def process(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        dct = data_dict['Dict[str,Any]|dict']
        return dct


class SqueezeDict(TypedDataProcessor):
    def __init__(self, key_lst: List[str]):
        self.key_lst = key_lst
        super(SqueezeDict, self).__init__(key_lst, 'Dict[str,Dict[str,Any]]|squeezed_dict')

    def process(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'Dict[str,Dict[str,Any]]|squeezed_dict': data_dict
        }


class ListMerger(TypedDataProcessor):
    def __init__(self, merge_keys: List[str], output_key: str = 'merged'):
        super(ListMerger, self).__init__(merge_keys, 'List[Any]|' + output_key)

    def process(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        result_lst = []
        for elem in data_dict.values():
            result_lst += elem
        output_k = self.get_output_keys()[0]
        return {
            output_k: result_lst
        }


class ListMapper(TypedDataProcessor):
    def __init__(self, map_func):
        super(ListMapper, self).__init__('List[Any]|list', "List[Any]|mapped")
        self.map_func = map_func

    def process(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        resul = list(map(self.map_func, data_dict['List[Any]|list']))
        return {
            'List[Any]|mapped': resul
        }


class ListModifier(TypedDataProcessor):
    """
    处理每一个list，产生一个新的list,[0, inf)
    最后的结果是这些list的拼接
    """
    def __init__(self, modify_func):
        super(ListModifier, self).__init__("List[Any]|lst")
        self.modify_func = modify_func

    def process(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        lst = data_dict['List[Any]|lst']
        result_lst = []
        for elem in lst:
            result_lst += self.modify_func(elem)
        return {
            "List[Any]|modified_lst": result_lst
        }


def single_key_observer(key_name: str, key_processor: ProcUnit, keep: bool = False):
    """
    创建一个ProcUnit，可以给key_processor一个只包含key_name的data_dict，防止出现参数匹配错误
    :param key_name:
    :param key_processor:
    :param keep:
    :return:
    """
    if keep:
        two_way = KeyFilter(None, all_input=True) * (KeyFilter([key_name]) + key_processor)
    else:
        two_way = (KeyFilter(None, all_input=True) + KeyEater(key_name)) * (KeyFilter([key_name]) + key_processor)
    return two_way


class ListExtractor(TypedDataProcessor):
    def __init__(self, extract_func: Callable):
        self.extract_func = extract_func
        super(ListExtractor, self).__init__('list|lst', 'list|extracted')

    def process(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        lst = data_dict['list|lst']
        extracted = list(map(self.extract_func, lst))
        return {
            self.get_output_keys()[0]: extracted
        }


class ListInjector(TypedDataProcessor):
    def __init__(self, inject_func: Callable[[Any, Any], Any]):
        self.inject_func = inject_func
        super(ListInjector, self).__init__(['RESULT', 'list|original'], 'list|result')

    def process(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        result = data_dict[self.find_input_key('RESULT')]
        original = data_dict['list|original']
        injected = []
        for (elem_res, elem_org) in zip(result, original):
            injected.append(self.inject_func(elem_org, elem_res))
        return {
            self.get_output_keys()[0]: injected
        }


def _simple_inject(origin, inject):
    origin['processed'] = inject
    return origin


def extract_and_inject(inner_proc: ProcUnit, extract_func: Callable = lambda x: x, inject_func: Callable = _simple_inject):
    """
    为了将'对一个序列执行某种操作'和'对一个序列中每个元素取其某种属性，处理完后再注入到原序列中'这两种操作分离，采用
    这种Extract-Inject的模式
    extract, 即对初始序列对每个元素，按某种规则取其中某个属性，然后用属性构成等长的新序列
    inject, 将属性的序列按序注入到初始序列当中
    由inner_proc执行具体的操作。这样分离，inner_proc就可以专门设计，不需要考虑如何从可能结构比较复杂的容器中取值
    :param inner_proc: 该ProcUnit，对序列数据执行某种操作
    :param extract_func: 用来构建ListExtractor的function，将对每一个元素单独操作，返回一个抽取的属性
    :param inject_func: 用来构建ListInjector的function，第一个参数为原序列中的原始元素，第二个参数为需要注入的属性
    :return:
    """
    p = KeyFilter(None, all_input=True) \
        * (KeyFilter(None, all_input=True) + ListExtractor(extract_func) + inner_proc + KeyNameChanger('RESULT')) + ListInjector(inject_func)
    return p

"""
Processors depending on outer package
"""


class BERT_Tokenizer(TypedDataProcessor):
    """
    input_lst是一个list，每一个元素要么是str，要么是List[str]，总之是能够被tokenize的
    当前TDP的行为是对每一个元素调用tokenize
    """
    def __init__(self, max_len=128):
        super(BERT_Tokenizer, self).__init__('List[Union[str,List[str]]]|input_lst', "List[Dict[str,list]]|tokenized")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.max_len = max_len

    def process(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        input_lst = data_dict['List[Union[str,List[str]]]|input_lst']
        results = []
        for elem_input in input_lst:
            tokenized = self.tokenizer(elem_input, padding=True, truncation=True, max_length=self.max_len)
            token_seq = self.tokenizer.convert_ids_to_tokens(tokenized['input_ids'])
            tokenized['token'] = token_seq
            results.append(dict(tokenized))
        return {
            list(self.get_output_keys())[0]: results
        }


class BERT_Tokenizer_double(TypedDataProcessor):
    """
    与BERT_Tokenizer的区别是，double的句子输入将作为上下句，因此也需要保证input_lst的每一个元素都是长为2的list
    """
    def __init__(self, max_len=128):
        super(BERT_Tokenizer_double, self).__init__('List[List[str]]|input_lst', "List[Dict[str,list]]|tokenized")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.max_len = max_len

    def process(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        input_lst = data_dict['List[List[str]]|input_lst']
        results = []
        for elem_input in input_lst:
            if len(elem_input) != 2:
                raise Exception('list of str的长度不为2，无法正确进行上下句tokenize！')
            tokenized = self.tokenizer(*elem_input, padding=True, truncation=True, max_length=self.max_len)
            results.append(dict(tokenized))
        return {
            list(self.get_output_keys())[0]: results
        }


"""
batchify and collate tools ?
"""

"""
对不同数据进行batch化的函数
- batchify_tensor 将tensor batch化，按照tensor中最大长度padding
- batchify_iterable 对于iterable的batch化
- batchify_dict_of_tensors
"""


def batchify_ndarray(tensors_lst: [], bsz=None, keep_tail=False):
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
            batched_tensor = np.stack(temp_tensors)
            tensor_batches.append(batched_tensor)
            temp_tensors = []
    if keep_tail and len(temp_tensors) != 0:
        tensor_batches.append(np.stack(temp_tensors))
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
        dict_batched_tensors[key] = batchify_ndarray(value, bsz=bsz, keep_tail=keep_tail)
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
        'ndarray': batchify_ndarray,
        'dict_tensor': batchify_dict_of_tensors,

        # in case i messed up
        'iterables': batchify_iterable,
        'ndarrayss': batchify_ndarray,
        'dict_tensors': batchify_dict_of_tensors
    }
    results = list(map(lambda x: function_map[x[1]](lsts[x[0]], bsz=bsz), enumerate(lst_types)))
    return results


def simple_collate_fn(lst: List[Dict[str, Any]]):
    """
    现在不接收tensor了。对于tensor，直接stack, 默认是对axis=0进行stack，所以tensor的size(0)必须相等
    对于np.ndarray，也是直接stack。注意不会把ndarray转化为tensor，需要手动转化
    对于其他的，都堆叠为list
    :param lst:
    :return:
    """
    lst_keys = list(lst[0].keys())
    result_dict = {}
    for k in lst_keys:
        # 如果是tensor，就stack
        # if type(lst[0][k]) == torch.Tensor:
        #     tensor_lst = []
        #     for l in lst:
        #         tensor_lst.append(l[k])
        #     result_dict[k] = torch.stack(tensor_lst)
        if type(lst[0][k]) == np.ndarray:
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


"""
中文处理相关
"""
