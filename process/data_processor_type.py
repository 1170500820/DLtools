from type_def import *
import typing_utils

DataUnit = Tuple[str, str]
DataUnitStr = str
Units = List[Union[DataUnit, DataUnitStr]]
UnitLike = Union[Units, Union[DataUnitStr, DataUnit]]


"""
data_unit的对齐与转换函数

对齐
    - key_match 对两组data_unit计算对齐关系
转换
    - key_cast 把data_dict按照data_unit，计算对齐后对key，并更新key值
    - infer_unit_type 推断data的typing类型，在未指定输出的情况下使用
    - signature_to_data_unit 将字符串转化为data_unit对tuple
    - data_unit_to_signature 将data_unit转换为字符串你
"""


def key_match(pre_output: List[DataUnit], post_input: List[DataUnit], mode='strict') -> Dict[DataUnit, DataUnit]:
    """
    todo key_match内部模块化
    接收上一个TDP的输出的data_unit list，以及下一个TDP所能够接受的输入的data_unit list
    输出data_unit之间的对应关系.
    输出的对应关系不一定覆盖pre_output或post_input中的所有data_unit，但是可以保证：
        1，每个data_unit只会match一次，不会重复match
        2，（strict）每个match一定是唯一解，不会有多个可行解/ （moderate）每个match不一定是唯一解，可能存在多个平行的可行解

    spec:
        name
            name不能为空。如果name的输入是一个空字符串，可以勉强假设这个data_unit的name就是空字符串，但是原则上不允许这种情况出现，在TDP的
            设计中也会杜绝这种情况。
            name与type中不能包含"｜"
            在pre_output与post_input中，不能出现重复的name。

        type
            type必须为合法的typing表达式


    implementation detail：

        首先，找出所有name相同的作为match，然后从list中移除。

        然后，对于每一个post，找出所有与其类型相同或者是其子类型的type，分别存入equal和candidate。首先检查equal，若equal不为空，则最终
        会采用equal的match。

        type
            如果存在多个符合条件的type，算法的行为？
            不考虑cnt，在name也没法提供有用信息的前提下，如果pre中存在相同的type，或者无法进行排序的type，本来就没法将他们
            正确地对应到post中的type。因为信息不足。
            因此这种情况要么报错，扔出一个exception作为提示。要么直接忽略，随机match。
        cnt
    :param pre_output:
    :param post_input:
    :param mode: {'moderate', 'strict'}，如果为moderate，则对于冲突或无法排序的type，随机选择，对顺序不保证
        如果为strict，则凡遇到无法唯一确定的type，扔出exception
    :return:
    """
    def check_units_legality(data_unit_lst: List[DataUnit]):
        """
        检查合法性
            - name不能为空
            - name不能重复
            - name与type不能包含"｜"
            - type必须是合法的typing表达式
        :param data_unit_lst:
        :return:
        """
        name_set = set(x[1] for x in data_unit_lst)
        if len(name_set) < len(data_unit_lst):
            raise Exception('存在重复name！')
        if '' in name_set:
            raise Exception('存在空name！')
        for elem in data_unit_lst:
            try:
                eval(elem[0])
            except:
                raise Exception(f'{elem[0]}不是合法的typing表达式！')
            for part in elem:
                if '|' in part:
                    raise Exception(f'{elem} 的 {part} 部分包含\"｜\"！')

    def find_name_match(pre: List[DataUnit], post: List[DataUnit]) -> (List[DataUnit], List[DataUnit], Dict[DataUnit, DataUnit]):
        """
        接收两个data unit list
        找出name匹配的结果，然后删去name匹配后的data_unit，返回处理后的data unit list和match dict
        :param pre:
        :param post:
        :return:
        """
        name_match_dict: Dict[DataUnit, DataUnit] = {}
        pre_name_dict = {x[1]: x for x in pre}
        for elem_post in post:
            if elem_post[1] in pre_name_dict:
                name_match_dict[pre_name_dict[elem_post[1]]] = elem_post
        for k, v in name_match_dict.items():
            pre.remove(k)
            post.remove(v)
        return pre, post, name_match_dict

    def find_equal_type_match(pre: List[DataUnit], post: List[DataUnit], mode='strict') -> (List[DataUnit], List[DataUnit], Dict[DataUnit, DataUnit]):
        """
        找到所有具有相等类型的匹配。

        与find_name_match的实现略有不同，因为type要考虑存在相等的，而存在多种选择时，需要抛出异常
        :param pre:
        :param post:
        :return:
        """
        equal_type_match_dict: Dict[DataUnit, DataUnit] = {}
        matched_pre_set, matched_post_set = set(), set()
        for elem_post in post:
            if elem_post in matched_post_set:
                continue
            target_type = eval(elem_post[0])
            equal_type_dict, equal_type_lst = {}, []
            for elem_pre in pre:
                if elem_pre in matched_pre_set:
                    continue
                pre_type = eval(elem_pre[0])

                if pre_type == target_type:
                    equal_type_dict[elem_pre[0]] = elem_pre
                    equal_type_lst.append(elem_pre[0])
            if len(equal_type_lst) >= 2 and mode == 'strict':
                raise Exception(f'类型{equal_type_lst[0]}存在重复！')
            if len(equal_type_lst) == 1:
                match_pre = equal_type_dict[equal_type_lst[0]]
                equal_type_match_dict[match_pre] = elem_post
                matched_pre_set.add(match_pre)
                matched_post_set.add(elem_post)

        for k, v in equal_type_match_dict.items():
            pre.remove(k)
            post.remove(v)
        return pre, post, equal_type_match_dict

    def find_subtype_match(pre: List[DataUnit], post: List[DataUnit], mode='strict') -> (List[DataUnit], List[DataUnit], Dict[DataUnit, DataUnit]):
        """
        找出所有子类型的匹配
        :param pre:
        :param post:
        :param mode:
        :return:
        """
        type_match_dict: Dict[DataUnit, DataUnit] = {}
        matched_post_set, matched_pre_set = set(), set()
        for elem_post in post:
            target_type = eval(elem_post[0])

            # 没有建立过连接的一对dataunit，进行类型判断
            if elem_post in matched_post_set:
                continue
            candidate_dict, candidate_type_lst = {}, []  # pre中是post子类的
            for elem_pre in pre:
                if elem_pre in matched_pre_set:
                    continue

                if eval(elem_post[0]) == eval(elem_pre[0]):  # 若类型相等，直接建立match
                    raise Exception('遇到类型相同的！')
                elif not typing_utils.issubtype(eval(elem_pre[0]), eval(elem_post[0])):  # 不同且不为子类，不合法
                    continue
                else:  # 不同，但是构成子类
                    # todo 重复的类型？
                    candidate_dict[elem_pre[0]] = elem_pre
                    candidate_type_lst.append(elem_pre[0])
                    continue

            if len(candidate_type_lst) > 0:
                discard_set = set()
                for elem1 in candidate_type_lst:
                    for elem2 in candidate_type_lst:
                        result = type_compare(eval(elem1), eval(elem2), target_type)
                        if result == 0:  # 二者同级
                            continue
                        elif result == -1:  # 无法比较
                            if mode == 'strict':
                                raise Exception(f'{elem1}与{elem2}无法比较！')
                        else:
                            if result == elem1:
                                discard_set.add(elem2)
                            else:
                                discard_set.add(elem1)
                for elem_discard in discard_set:
                    candidate_type_lst.remove(elem_discard)
                if len(candidate_type_lst) >= 2 and mode == 'strict':
                    raise Exception('超过2个可选类型！')
                random_pick = list(candidate_dict.values())[0]
                type_match_dict[random_pick] = elem_post
                matched_post_set.add(elem_post)
                matched_pre_set.add(random_pick)
        #  remove matched dataunit
        for elem_p in matched_post_set:
            post.remove(elem_p)
        for elem_pre in matched_pre_set:
            pre.remove(elem_pre)

        return pre, post, type_match_dict

    def find_type_match(pre: List[DataUnit], post: List[DataUnit]) -> (List[DataUnit], List[DataUnit], Dict[DataUnit, DataUnit]):
        """
        寻找所有的类型匹配的data unit
        pre与post已经不包含可以name匹配的了

        返回去除了匹配unit的pre与post，以及type匹配产生的match
        :param pre:
        :param post:
        :return:
        """
        type_match_dict: Dict[DataUnit, DataUnit] = {}
        matched_post_set, matched_pre_set = set(), set()
        for elem_post in post:
            target_type = eval(elem_post[0])

            # 没有建立过连接的一对dataunit，进行类型判断
            if elem_post in matched_post_set:
                continue
            candidate_dict, candidate_type_lst = {}, []  # pre中是post子类的
            equal_dict, equal_type_lst = {}, []  # pre中与post类型相同的
            for elem_pre in pre:
                if elem_pre in matched_pre_set:
                    continue

                if eval(elem_post[0]) == eval(elem_pre[0]):  # 若类型相等，直接建立match
                    equal_dict[elem_pre[0]] = elem_pre
                    equal_type_lst.append(elem_pre[0])
                    # match_dict[elem_pre] = elem_post
                    # matched_post_set.add(elem_post)
                    # matched_pre_set.add(elem_pre)
                    continue
                elif not typing_utils.issubtype(eval(elem_pre[0]), eval(elem_post[0])):  # 不同且不为子类，不合法
                    continue
                else:  # 不同，但是构成子类
                    # todo 重复的类型？
                    candidate_dict[elem_pre[0]] = elem_pre
                    candidate_type_lst.append(elem_pre[0])
                    continue

            # 先检查equal的
            if len(equal_type_lst) > 0:
                if len(equal_type_lst) > 1 and mode == 'strict':
                    raise Exception(f'{elem_post[0]}在pre中存在多个相同的类型！')
                random_pick = list(equal_dict.values())[0]
                type_match_dict[random_pick] = elem_post
                matched_post_set.add(elem_post)
                matched_pre_set.add(random_pick)
                continue
            elif len(candidate_type_lst) > 0:
                discard_set = set()
                for elem1 in candidate_type_lst:
                    for elem2 in candidate_type_lst:
                        result = type_compare(eval(elem1), eval(elem2), target_type)
                        if result == 0:  # 二者同级
                            continue
                        elif result == -1:  # 无法比较
                            if mode == 'strict':
                                raise Exception(f'{elem1}与{elem2}无法比较！')
                        else:
                            if result == elem1:
                                discard_set.add(elem2)
                            else:
                                discard_set.add(elem1)
                for elem_discard in discard_set:
                    candidate_type_lst.remove(elem_discard)
                if len(candidate_type_lst) >= 2 and mode == 'strict':
                    raise Exception('超过2个可选类型！')
                random_pick = list(candidate_dict.values())[0]
                type_match_dict[random_pick] = elem_post
                matched_post_set.add(elem_post)
                matched_pre_set.add(random_pick)
        #  remove matched dataunit
        for elem_p in matched_post_set:
            post.remove(elem_p)
        for elem_pre in matched_pre_set:
            pre.remove(elem_pre)

        return pre, post, type_match_dict

    def type_compare(type1, type2, type_target):
        """
        比较type1和type2谁离type_target更近，也就是更适合与其匹配

        :param type1:
        :param type2:
        :param type_target:
        :return:
            superior type
            0 二者相等
            -1 无法比较
        """
        if not typing_utils.issubtype(type1, type_target):
            raise Exception(f'{type1}不是{type_target}的子类！')
        if not typing_utils.issubtype(type2, type_target):
            raise Exception(f'{type2}不是{type_target}的子类！')
        if type1 == type2:
            return 0
        elif typing_utils.issubtype(type1, type2):
            return type2
        elif typing_utils.issubtype(type2, type1):
            return type1
        else:
            return -1

    match_dict: Dict[DataUnit, DataUnit] = {}

    # 复制，不要对引用进行修改
    pre_output = pre_output.copy()
    post_input = post_input.copy()

    # 检查一下输入的合法性
    check_units_legality(pre_output)
    check_units_legality(post_input)

    # check name
    pre_output, post_input, name_match = find_name_match(pre_output, post_input)

    # check type
    pre_output, post_input, type_match = find_type_match(pre_output,  post_input)

    # check cnt， 只有在刚好只有一个输入和输出的时候，对齐
    cnt_match: Dict[DataUnit, DataUnit] = {}
    if len(post_input) == len(pre_output) == 1 and len(type_match) == 0 and len(name_match) == 0:
        if len(post_input) == len(pre_output):
            for (elem_pre, elem_post) in zip(pre_output, post_input):
                cnt_match[elem_pre] = elem_post

    match_dict.update(name_match)
    match_dict.update(type_match)
    match_dict.update(cnt_match)

    return match_dict


def key_cast(data_dict: Dict[str, Any], input_keys: List[DataUnit]):
    """
    根据input_keys中的data_unit，计算match关系，并将data_dict中符合match关系的data_unit的key值更正，使得数据能够正常传入
    :param data_dict:
    :param input_keys:
    :return:
    """
    pass


def remove_type(data_dict: Dict[str, Any]):
    """
    去除key值中的类型信息

    直接在data_dict上修改，返回的也是输入的data_dict
    :param data_dict:
    :return:
    """
    key_lst = list(data_dict.keys())
    for elem_key in key_lst:
        if "|" in elem_key:
            new_key = elem_key[elem_key.index('|') + 1:]
            data_dict[new_key] = data_dict[elem_key]
            data_dict.pop(elem_key)
    return data_dict


def infer_unit_type(data: Any) -> str:
    """
    推断data的类型。返回符合typing格式的字符串
    :param data:
    :return:
    """
    typestr = str(type(data)).split('\'')[1]
    return typestr


def signature_to_data_unit(signature: str) -> DataUnit:
    """
    将data_dict的键值转化为data_unit
    :param signature:
    :return:
    """
    if '|' in signature:
        type_str, name_str = signature.split('|')
        try:
            eval(type_str)
        except Exception as e:
            raise Exception(f'非法的type_str:{type_str}')
        return type_str, name_str
    else:  # 默认type为Any
        return 'Any', signature


def data_unit_to_signature(data_unit: DataUnit) -> str:
    """
    将data_unit转化为signature
    :param data_unit:
    :return:
    """
    return data_unit[0] + '|' + data_unit[1]


"""
核心Class：TypedDataProcessor与TypedProcessorManager的设计
"""


class TypedDataProcessor:
    def __init__(self, input_keys: UnitLike, output_keys: UnitLike = None, all_input: bool = False, keep: bool = False):
        """

        :param input_keys:
        :param output_keys:
        :param all_input: 若为True，则接收所有的input_keys，而不进行输入检查。设为True需要保证输入的为合法的data_dict，因为不会进行输入检查了
        :param keep: 若为True，则保留没有作为输入的部分。会在process之前保存，然后update到process的输出中
        """
        self.all_input = all_input
        self.keep = keep
        if all_input:
            self.input_keys = None
            self.output_keys = None
            self.func = None
            return
        if not isinstance(input_keys, list):
            input_keys = [input_keys]
        input_keys = [signature_to_data_unit(x) if isinstance(x, str) else x for x in input_keys]
        self.input_keys = input_keys

        if output_keys is None:
            self.output_keys = output_keys
        else:
            if not isinstance(output_keys, list):
                output_keys = [output_keys]
            output_keys = [signature_to_data_unit(x) if isinstance(x, str) else x for x in output_keys]
            self.output_keys = output_keys
        self.func = None

    def __call__(self, data: Union[Dict[str, Any], Any] = None, datas: List[Any] = None) -> Dict[str, Any]:
        """
        __call__对输入进行格式转换，键值对齐，以及其他各种可能的处理，以保证传入process的是一个能够正常运行的data_dict
        虽然现在只有dict的输入，但是后续可以不断扩展，提高易用性

        TDP不会依赖于别的TDP来确定输入的unit之间的对应关系，而是每次都通过重新计算match（或者从cache读取）来确定对应。
        对于输入的data_dict，TDP会根据对应算法计算对应关系，将data_dict中的key转化为TDP所需要的key值。
        :param data_dict:
        :return:
        """
        # 现在就做个简单的转换吧
        # 首先是把乱七八糟的输入转化为dict todo 写文档
        if data is not None:  # data_dict是最高优先级
            if isinstance(data, dict):  # data_dict是最高优先级
                data_dict = data
            else:  # 单独的data是第二优先级
                typestr = infer_unit_type(data)
                data_dict = {f'{typestr}:data': data}
        elif datas is not None:  # 最后取datas
            if len(datas) == 0:
                raise Exception('空输入！')
            data_dict = {}
            for idx, elem_data in enumerate(datas):
                typestr = infer_unit_type(elem_data)
                data_dict[f'{typestr}|data-{idx + 1}'] = elem_data
        else:
            raise Exception('至少要有一个输入！')

        # 然后把dict的key值转化为process能够接受的
        input_data_unit_lst = list(map(signature_to_data_unit, list(data_dict.keys())))
        input_index = {}
        for elem_input in list(data_dict.keys()):
            input_index[signature_to_data_unit(elem_input)] = elem_input

        # 若all_input设置为True，直接接收所有输入
        if self.all_input:  # 只需要将match_dict改为input_data_unit_lst的本体映射
            match_dict = {a: a for a in input_data_unit_lst}
        else:  # 否则通过key_match如实计算
            self_data_unit_lst = list(self.input_keys)
            match_dict = key_match(input_data_unit_lst, self_data_unit_lst)
        converted_input = {}
        extra_input_keys = set(data_dict.keys())
        extra_input = {}
        for key, value in match_dict.items():
            converted_input[data_unit_to_signature(value)] = data_dict[input_index[key]]
            extra_input_keys.remove(input_index[key])
        for elem_extra in extra_input_keys:
            extra_input[elem_extra] = data_dict[elem_extra]
        if self.func:
            result_dict = self.func(converted_input)
        else:
            result_dict = self.process(converted_input)
        if self.keep:
            result_dict.update(extra_input)
        return result_dict

    def process(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def register_process(self, func):
        self.func = func

    def get_input_keys(self):
        return list(map(data_unit_to_signature, self.input_keys))

    def get_output_keys(self):
        return list(map(data_unit_to_signature, self.output_keys))

    def find_input_key(self, name: str):
        for elem in self.input_keys:
            if elem[1] == name:
                return elem[0] + '|' + elem[1]
        raise Exception(f'[{name}]not found in input keys!')

    def __add__(self, other: 'ProcUnit') -> 'TypedProcessorManager':
        if isinstance(other, TypedDataProcessor):
            return TypedProcessorManager([self, other])
        elif isinstance(other, TypedProcessorManager):
            other.add_before(self)
            return other
        elif isinstance(other, list):
            for elem in other:
                if not (isinstance(elem, TypedProcessorManager) or isinstance(elem, TypedDataProcessor)):
                    raise Exception('TypedDataProcessor只能与List[ProcUnit]进行加法运算')
            return TypedProcessorManager([self, other])
        else:
            raise Exception('TypedDataProcessor只能够与ProcUnit进行加法运算！')

    def __mul__(self, other: 'ProcUnit') -> 'TypedProcessorManager':
        if isinstance(other, TypedDataProcessor):
            return TypedProcessorManager([[self, other]])
        elif isinstance(other, TypedProcessorManager):
            other.add_parallel(self)
            return other
        else:
            raise Exception('TypedDataProcessor只能够与ProcUnit进行加法运算！')


class TypedProcessorManager:
    """
    包装TypedDataProcessor以及自己
    依然是processor的两种处理关系：串联，并联


    """
    def __init__(self, processors: 'ProcSequence'):
        """
        接收一个ProcSequence
        :param processors:
        """
        self.processors = processors

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        TPM
        数据的传入，key的转换，都在TDP中执行了。因此TPM的任务就简单了。
        :param data_dict:
        :return:
        """
        input_dict = data_dict
        for elem_proc in self.processors:
            result_dict = {}
            if isinstance(elem_proc, list):
                for elem_parallel_proc in elem_proc:
                    parallel_output = elem_parallel_proc(input_dict)
                    result_dict.update(parallel_output)
                input_dict = result_dict
            else:
                single_output = elem_proc(input_dict)
                input_dict = single_output
        # output = TypedProcessorManager.run(data_dict, self.processors)
        return input_dict

    @staticmethod
    def run(data_dict: Dict[str, Any], proc) -> Dict[str, Any]:
        input_dict = data_dict
        if isinstance(proc, TypedProcessorManager) or isinstance(proc, TypedDataProcessor):
            output_data = proc(input_dict)
        elif isinstance(proc, list):
            output_data = {}
            for elem in proc:
                output_data.update(TypedProcessorManager.run(data_dict, elem))
        else:
            raise Exception(f'未知的输入类型{type(proc)}')
        return output_data

    def add_before(self, proc: ProcUnit):
        """
        在前面加入一段proc
        :param proc:
        :return:
        """
        if isinstance(proc, TypedDataProcessor):
            self.processors.insert(0, proc)
        elif isinstance(proc, TypedProcessorManager):
            self.processors = proc.processors + self.processors
        elif isinstance(proc, list):
            for elem in proc:
                if not (isinstance(elem, TypedProcessorManager) or isinstance(elem, TypedDataProcessor)):
                    raise Exception('TypedProcessorManager只能与List[ProcUnit]合并')
            self.processors.insert(0, proc)
        else:
            raise Exception('TypedProcessorManager只能与ProcUnit合并')

    def add_after(self, proc: Union[ProcUnit, List[ProcUnit]]):
        if isinstance(proc, TypedDataProcessor):
            self.processors.append(proc)
        elif isinstance(proc, TypedProcessorManager):
            self.processors = self.processors + proc.processors
        elif isinstance(proc, list):
            for elem in proc:
                if not (isinstance(elem, TypedProcessorManager) or isinstance(elem, TypedDataProcessor)):
                    raise Exception('TypedProcessorManager只能与List[ProcUnit]合并')
            self.processors.append(proc)
        else:
            raise Exception('TypedProcessorManager只能与ProcUnit合并')

    def add_parallel(self, proc: Union[ProcUnit, List[ProcUnit]]):
        if isinstance(proc, TypedDataProcessor):
            cur_TPM = TypedProcessorManager(self.processors)
            self.processors = [[proc, cur_TPM]]
        elif isinstance(proc, TypedProcessorManager):
            self.processors = [[proc, TypedProcessorManager(self.processors)]]
        else:
            raise Exception('TypedProcessorManager只能与ProcUnit合并')

    def __add__(self, other: ProcUnit) -> 'TypedProcessorManager':
        self.add_after(other)
        return self

    def __mul__(self, other) -> 'TypedProcessorManager':
        self.add_parallel(other)
        return self

# ProcUnit是能够处理data_dict的一个组件
ProcUnit = Union[TypedDataProcessor, TypedProcessorManager]
# ProcSequence是一系列可执行data_dict的单元。其中[ProcUnit]表示并联结构
ProcSequence = List[Union[ProcUnit, List[ProcUnit]]]
