from type_def import *
import rich
from rich.table import Table


class DataProcessor:
    """
    DataProcessor是建入DLtools数据处理的代码中，一个最小数据处理单元。

    概念：
        DataProcessor应当实现一个数据处理的步骤。通过多个DP的组合来实现复杂的数据处理流程

    ADT
        DP为Callable[[dict], dict]
        DP可以对一个满足条件的function简单包装，也可以继承DataProcessor实现
        DP需要包含input_keys与output_keys，都是tuple of str，分别存储了DP的输入与输出的dict的键值

    运算
        DP本身是最小单元，DP之间的组合只会产生ProcessorManager，按对应顺序封装了多个DP。一个DP专注于实现一个功能而不是多个功能的组合。
        对于DP与PM，将实现加法与乘法两种运算。
        加法代表顺序执行，乘法代表并联（平行）执行，
        - DP之间的加法
            DP的加法产生一个顺序执行的ProcessorManager
            DP1 + DP2 = PM， 调用PM将先后执行DP1，与DP2，DP1的输出作为DP2的输入
        - DP之间的乘法
            DP的乘法将产生并联执行的PM
            DP1 x DP2 = PM， 调用PM，会将PM的输入同时输入到DP1与DP2，然后将二者输出合并，作为最终的输出
        - DP与PM的加法
            代表将DP顺序地加入到PM的执行图中。
            DP + PM = PM1， 调用PM1，会按照[DP, PM_dp1, PM_dp2, ...]的顺序执行
        - DP与PM的乘法
            代表当前的DP将会与PM的运算并联进行
            DP + PM = PM1， 调用PM1， 输入会同时给DP和PM，然后将输出合并
    """
    def __init__(self, input_keys: Union[str, Iterable[str]], output_keys: Union[str, Iterable[str]]):
        if isinstance(input_keys, str):
            self.input_keys = tuple([input_keys])
        else:
            input_key_lst = []
            for elem_input in input_keys:
                input_key_lst.append(elem_input)
            self.input_keys = tuple(input_key_lst)
        if isinstance(output_keys, str):
            self.output_keys = tuple([output_keys])
        else:
            output_key_lst = []
            for elem_output in output_keys:
                output_key_lst.append(elem_output)
            self.output_keys = tuple(output_key_lst)
        self.process_func = None

    def __call__(self, data_dicts: Dict[str, Any]) -> Dict[str, Any]:
        if self.process_func:
            result = self.process_func(data_dicts)
        else:
            result = self.process(data_dicts)
        return result

    def process(self, data_dicts: dict) -> Union[dict, List[dict]]:
        raise NotImplementedError

    def register_precess(self, func):
        self.process_func = func

    def __add__(self, other):
        """
        如果other是DP，则产生顺序执行的PM
        若是PM，则直接接入 todo
        :param other:
        :return:
        """
        if isinstance(other, DataProcessor):
            return ProcessorManager([self, other])
        elif isinstance(other, ProcessorManager):
            other.add_before(self)
            return other
        else:
            raise Exception('unsupported operation!')

    def __mul__(self, other):
        """
        todo
        :param other:
        :return:
        """
        if isinstance(other, DataProcessor):
            return ProcessorManager([[self, other]])
        elif isinstance(other, ProcessorManager):
            other.add_parallel(self)
            return other
        else:
            raise Exception

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__repr__()


class ProcessorManager:
    """
    ProcessorManager用于包装DataProcessor，与PM本身的序列

    概念：
        DP是最基本的操作，而PM就是由DP作为基础垒成的一个复杂的操作
        PM的执行序列中可以包含DP，也可以包含PM，这种递归的结构使得复杂的执行顺序得以定义
        （这并不是本意，PM与DP就不应该用于定义复杂的、图结构的处理流程，复杂流程应当用DataEnv划清界限，充分解耦合。但是实现这个会带来很大的方便）

    ADT：
        PM包含
        - 一个执行序列的list
            List中包含的元素是Union[Callable[[dict], dict], List[Callable[[dict], dict]]]
            Callable[[dict], dict]可以是ProcessorManager与DataProcessor
            list包装的callable代表并联执行的结构
        - 输入与输出的key的tuple
            其中，input_keys与output_keys代表整个执行序列的输入和输出。
            input_keys是第一个proc的输入，以及后面的proc中，无法通过前面的proc得到的输入的key
            output_keys是最后一个proc的输出，以及前面的proc中，未被其后proc收集的key

    实现：
        在调用时，ProcessorManager将首先对DataProcessors的输入与输出进行检查，已确定能否正确传值

        ProcessorManager的输入与输出都是dict，key值为对应数据的名字，而value则是数据本体
        输入的数据将依次经过每一个DataProcessor
        - DataProcessor的输入中包含的dict将被送入，而不包含的dict则会被保留，与当前阶段的输出合并，进行传入下一个DP
        - 如果是单个DP，则直接传入
        - 如果是List[DP]，则key值会同时传入，输出将会被合并
            -- 如果输出有重复值，会被合并，具体保留哪个则未定义
            -- 传入的值是引用，因此若两个DP包含相同的输入，上一个DP对输入的修改会作用到下一个DP的输入
               PM不会管理这些情况，也不保证输入的传入顺序
            -- 未被传入的key会保留至下一步的处理；只要被并联结构中的一个proc调用了的key，都被算作传入，不会保留

    PM的运算
        把DP看作具体的操作，PM就是操作的组织与调度。PM的整个实现不包含任何具体的处理操作，只包含对DP的调度。
        PM的每一个运算都将产生合法的PM，计算时会进行合法性检查
        PM的运算包括加法与乘法，与PM和DP都有定义
        - PM之间的加法
            PM1 + PM2 = PM3，PM3是PM1与PM2的简单顺序拼接，
        - PM之间的乘法
            PM1 x PM2 = PM3，PM3的行为是将PM1与PM2并联执行。
            实现上，若PM1与PM2中有一个的ProcList长度为1，那么会将该proc提取出来，合并到另一个PM中。这样可以减少PM包装的层数。
            如果ProcList长度都大于1，则通过直接在外面包装一层PM来实现
        - PM与DP的加法
            简单的顺序拼接。但是与上面相同，如果PM的ProcList长度为1，则DP会被并入PM，而不是在外面包装一个新的PM
        - PM与DP的乘法
            产生的新PM的行为是并联执行参与乘法的PM与DP。同理，若ProcList长度为1，会在PM内处理。
    """
    def __init__(self, processors: Union[DataProcessor, List[Union[DataProcessor, List[DataProcessor]]]]):
        if isinstance(processors, DataProcessor):
            self.processors = [processors]
        else:
            self.processors = processors
        self._check()
        self.input_keys, self.output_keys = self._get_keys()
        self._check()

    def __call__(self, data: dict) -> dict:
        """
        对于processors的每一个元素，输入data，然后将输出更新到data中
        - 不属于当前processor的输入的key，保留
        - 输入当前processor的key，不保留（processor可自行选择保留）
        - 对于并联processors，输入时，data保留二者的不包含key的交集（也就是凡用过一次的key，都不会被保）
            输出直接覆盖，覆盖顺序不一定与并联顺序一致
        :param data:
        :return:
        """
        input_data = data
        for elem_proc in self.processors:
            if isinstance(elem_proc, list):
                proc_outputs: List[dict] = []  # 存放每一个elem_proc的输出结果
                left_keys = set(input_data.keys())  # 存放没有被任何elem_proc使用到的key
                left_dict = {}  # 存放没有被elem_proc用到的key-value
                for elem_sub_proc in elem_proc:
                    sub_input_dict = {}  # 当前sub proc的输入dict
                    for key in elem_sub_proc.input_keys:
                        sub_input_dict[key] = input_data[key]
                        left_keys.discard(key)
                    proc_outputs.append(elem_sub_proc(sub_input_dict))
                for elem_key in left_keys:
                    left_dict[elem_key] = input_data[elem_key]
                # 把输出结果update到left_dict里面
                for elem_output in proc_outputs:
                    left_dict.update(elem_output)
                input_data = left_dict
            else:  # proc
                temp_input = {}
                for elem_key in elem_proc.input_keys:
                    temp_input[elem_key] = input_data.pop(elem_key)
                input_data.update(elem_proc(temp_input))
        return input_data

    @staticmethod
    def get_output_set(proc):
        """
        获取proc的输出
        :param proc:
        :return:
        """
        if isinstance(proc, DataProcessor):
            return set(proc.output_keys)
        elif isinstance(proc, ProcessorManager):
            return set(proc.output_keys)
        else:  # list
            output_set = set()
            for elem_proc in proc:
                output_set = output_set.union(set(ProcessorManager.get_output_set(elem_proc)))
            return output_set

    @staticmethod
    def get_input_set(proc):
        """
        获取proc的输入
        :param proc:
        :return:
        """
        if isinstance(proc, DataProcessor):
            return set(proc.input_keys)
        elif isinstance(proc, ProcessorManager):
            return set(proc.input_keys)
        else:  # list
            input_set = set()
            for elem_proc in proc:
                input_set = input_set.union(set(ProcessorManager.get_input_set(elem_proc)))
            return input_set

    def _check(self):
        """
        检查processors的输出与输出能否匹配
        若一个并联processors的输出存在重复，发出warning，但并不会抛出异常
        若上一个processors的输出存在不处于下一个processors的输出的key值，则抛出异常

        每一个processor的交接位置。

        发现了一个设计上的漏洞
        既然允许输入的dict跨proc传参，那么理论上一个PM的所有processor之间的输入输出可以完全不对应，形成一个完全展开的并联结构
        :return:
        """
        if len(self.processors) == 0:
            raise Exception('processors不能为空！')
        elif len(self.processors) == 1:
            return

        # inputs = set()
        # 每一个交
        # for idx in range(len(self.processors) - 1):
        #     last_outputs = self.get_output_set(self.processors[idx])
        #     next_inputs = self.get_input_set(self.processors[idx + 1])
        #     middle_data = inputs.union(last_outputs)  # 上一proc的输入，以及前几个proc的保留key
        #     if len(next_inputs - middle_data) != 0:
        #         raise Exception(f'idx-{idx}与idx-{idx + 1}的输入输出无法匹配！')
        #     inputs = middle_data - next_inputs

    def _get_keys(self) -> Tuple[tuple, tuple]:
        """
        解析当前的processors，获取输入与输出的keys
        :return:
        """
        if len(self.processors) == 1:
            return tuple(ProcessorManager.get_input_set(self.processors[0])), tuple(ProcessorManager.get_output_set(self.processors[-1]))
        inputs = self.get_input_set(self.processors[0])
        env_keys = set()  # 积累的key
        for idx in range(len(self.processors) - 1):
            last_output = ProcessorManager.get_output_set(self.processors[idx])  # 上一个proc产生的输出
            next_input = ProcessorManager.get_input_set(self.processors[idx + 1])  # 下一个proc需要的key
            env_keys = env_keys.union(last_output)  # 中间状态时的累积key
            extra_input = next_input - env_keys  # 积累的key所无法提供的，通过输入提供
            env_keys = env_keys - next_input  # 剔除掉下一个proc的输入，余下的key
            inputs = inputs.union(extra_input)
        outputs = env_keys.union(self.get_output_set(self.processors[-1]))
        return tuple(inputs), tuple(outputs)

    def add_after(self, processor: Union[DataProcessor, 'ProcessorManager', List[Union[DataProcessor, 'ProcessorManager']]]):
        """
        从前面加入一段DP或者PM
        :param processor:
        :return:
        """
        if isinstance(processor, DataProcessor):
            self.processors.append(processor)
        elif isinstance(processor, ProcessorManager):
            for elem_proc in processor.processors:
                self.processors.append(elem_proc)
        else:
            self.processors.append(processor)
        self.input_keys, self.output_keys = self._get_keys()
        self._check()

    def add_before(self, processor: Union[DataProcessor, 'ProcessorManager', List[Union[DataProcessor, 'ProcessorManager']]]):
        """
        从后面加入一段DP或者PM
        :param processor:
        :return:
        """
        if isinstance(processor, DataProcessor):
            self.processors.insert(0, processor)
        elif isinstance(processor, ProcessorManager):
            for idx, elem_proc in processor.processors:
                self.processors.insert(idx, elem_proc)
        else:
            self.processors.insert(0, processor)
        self.input_keys, self.output_keys = self._get_keys()
        self._check()

    def add_parallel(self, processor: Union[DataProcessor, List[DataProcessor], 'ProcessorManager', List['ProcessorManager']]):
        """
        将一个processor并行的并入自己

        原则上的考虑，在PM内部不能再出现长度为1的PM，应当将其内部的东西解包出来
        功能上，二者应该没有区别。但是为了可维护与可扩展的考虑，PM内部状态单一还是健康一些。
        不过实际上很难递归的对PM进行内部检查，需要改变很多内部状态，犯错的风险太大。
        因此这里只实现了一个妥协的版本

        :param processor:
        :return:
        """
        if len(self.processors) == 1 and not isinstance(self.processors[0], list):  # 当自己的长度为1时，将processor拿出来。但是不会继续对processor检查
            inner_proc = self.processors[0]
            if isinstance(processor, DataProcessor):
                self.processors.append([inner_proc, processor])
            elif isinstance(processor, ProcessorManager):
                self.processors.append([inner_proc, processor])
            else:  # list
                processor.append(inner_proc)
                self.processors[0] = processor
        elif len(self.processors) == 1:
            inner_proc = self.processors[0]
            if isinstance(processor, DataProcessor):
                self.processors[0].append(processor)
            elif isinstance(processor, ProcessorManager):
                self.processors[0].append(processor)
            else:
                self.processors[0].extend(processor)
        else:
            selfcopy = ProcessorManager(self.processors)
            self.processors = [[processor, selfcopy]]
        self.input_keys, self.output_keys = self._get_keys()
        self._check()

    def __add__(self, other):
        self.add_after(other)
        return self

    def __mul__(self, other):
        self.add_parallel(other)
        return self

    def __repr__(self):
        repr_str = ''
        for elem in self.processors:
            if isinstance(elem, list):
                repr_str += '['
                for elem_elem in elem:
                    repr_str += str(elem_elem) + '|'
                repr_str += ']'
            else:
                repr_str += '-' + str(elem) + '-'
        return repr_str

    def __str__(self):
        return self.__repr__()
