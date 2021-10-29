from data_env import *
from data_processor import *
import rich
from rich.table import Table
from rich.console import Console
import networkx as nx
import matplotlib.pyplot as plt


class DataFlow:
    """
    DataFlow
    是对DataProcessor, ProcessorManager, DataEnv的整体进行包装。
    DataFlow管理着整个项目的每一个数据处理的方法
    通过DataFlow获取/存储/写入/可视化每一个节点的数据
    DataFlow可以动态加入各种数据依赖以及数据

    DataFlow的属性可以分为三种
        1，flow的定义
        定义抽象的数据处理过程，这个过程与文件路径，数据实体都无关，只是一个纯粹的抽象过程，
        包括DP，PM，DataEnv已经他们之间的依赖关系的定义

            - DP+PM
            对DataFlow来说，他们是一个封装。DataFlow只要求输入正确的dict，然后得到正确的输出dict，不会对内部实现做任何假设。
            在分支节点下，DataFlow直接将dict合并，而不会进行任何多余操作，对于重复key值的情况，DF保留哪个也是未定义行为。
            PM的设计应该考虑到这一点，在PM当中解决特化类型数据等问题。

            - DataEnv
            用于节点数据的IO，展示，或者缓存管理。
            IO与展示都可以通过输入新的函数来自定义其行为。也可以直接继承DataEnv来实现更复杂的操作，需注意

                a, 过于复杂的操作应该放入PM中处理，DataEnv应当被视为"数据处理的终点"

                b, flow目前只会调用I，O，inspect，

        2，数据源的定义
        数据总得从一个地方来，绝大多数情况下，就是文件路径，顶多是代码中的一些活数据。
        在获取数据时，会从数据源寻找对应的路径，或者数据。
        如果有缓存，则也会从缓存读取。只是默认情况不应该对数据源进行任何修改，缓存写入只会写到数据结果，也就是缓存路径。
        数据源不涉及复杂的结构也不涉及大量的数据（只是路径而已），因此

        3，数据结果的定义
        数据最终要输出到一个地方，大多数情况是写入文件，因此数据结果（目的地？）也是一批路径，
        其他性质与数据源类似

    ADT spec
        data_dependency, 存储数据的依赖关系
        data_getter, 存储用于获得数据的ProcessorManager
        data_envs, 存储用于管理数据节点的DataEnv
        *directories and simple data structure* 数据源与数据目的地的相关属性


        DataFlow始终处于"搭建中"的状态，也就是在插入/删除/修改新的getter/dependency/envs/directory的过程中，DF不会对内部结构做任何检查。
        只有在调用get/save之类的获取数据的方法的时候，DF才会先对路径上的对齐、env的合法性等进行检查。检查的范围仅限于数据流上的结构，也就是只会
        保证get/save能够正常运行的程度。
        todo 对合法性检查这块，还需要细一些。光和PM对齐不行，还得考虑一些特殊情况，做一些错误处理，留一些钩子函数
        DataEnv的缓存机制是一种持久化的方法，可以将某个中间节点的数据缓存，然后在数据流通过的时候，读取缓存以节约时间或者保持随机数据。因此
        DataEnv也可以被视为checkpoint


    todo 加入分支功能
        todo 更进一步，试着添加更灵活的数据流控制功能
    todo 加入缓存/读取管理功能
    todo 序列化
    todo 更方便的输入依赖信息
    todo 更方便的修改数据流的方法
    todo 增加ADT稳定性
    """
    def __init__(self, data_dependency: dict = None, data_getter: dict = None, cache_dir: str = '', cache_name: str = ''):
        if data_dependency is None:
            self.data_dependency = {}
        else:
            self.data_dependency = data_dependency
        if data_getter is None:
            self.data_getter = {}
        else:
            self.data_getter = data_getter
        self.data_envs = {}
        self.base_input = {}
        self.cache_dir = cache_dir
        self.cache_name = cache_name

    def add_base(self, key: str, value: Any):
        self.base_input[key] = value

    def get(self, name: str):
        self.check_key(name)
        inputs = {}
        if name in self.data_dependency.keys():
            for elem_sub in self.data_dependency[name]:
                inputs.update(self.get(elem_sub))
        else:
            for elem_input_key in self.data_getter[name].input_keys:
                inputs[elem_input_key] = self.base_input[elem_input_key]
        output = self.data_getter[name](inputs)
        return output

    def save(self, name: str):
        """
        获取一个数据，然后调用DataEnv的保存方法
        :param name:
        :return:
        """
        getdata = self.get(name)

    def check_key(self, name: str):
        """
        检查一个key的定义是否合法
        沿着dep链递归地往回找，要求所有env的签名必须完全一致
        :param name:
        :return:
        """
        # 先检查子节点
        if name not in self.data_getter.keys():
            raise Exception(f'要检查的key:[{name}]不存在！')
        if name in self.data_dependency:
            expect_input = set(self.data_getter[name].input_keys)
            actual_output = set()
            for elem_sub in self.data_dependency[name]:
                self.check_key(elem_sub)
                actual_output = actual_output.union(set(self.data_getter[elem_sub].output_keys))
            if actual_output != expect_input:
                raise Exception(f'[{name}]签名无法对齐！expected:{expect_input}, actual:{actual_output}')

    def add_dependency(self, target: str, source: Union[str, Iterable[str]]):
        """
        添加一条数据的依赖，并在同时将不存在data_getter中的项加入，并设为None
        :param target:
        :param source:
        :return:
        """
        # 将依赖存入data_dependency
        if isinstance(source, str):
            self.data_dependency[target] = source
        else:
            self.data_dependency[target] = source
        # 在data getter中添加对应的key，如果不存在，设置为None
        if target not in self.data_getter:
            self.data_getter[target] = None
            self.data_envs[target] = DataEnv(self.cache_dir, self.cache_name)
        for elem_src in source:
            if elem_src not in self.data_getter:
                self.data_getter[elem_src] = None
                self.data_envs[elem_src] = DataEnv(self.cache_dir, self.cache_name)

    def add_dependency_dict(self, dep_dict: dict):
        self.data_dependency.update(dep_dict)
        for key, value in dep_dict.items():
            if key not in self.data_getter:
                self.data_getter[key] = None
                self.data_envs[key] = DataEnv(self.cache_dir, self.cache_name)
            if isinstance(value, list):
                for elem_src in value:
                    self.data_getter[elem_src] = None
                    self.data_envs[elem_src] = DataEnv(self.cache_dir, self.cache_name)
            else:
                self.data_envs[value] = DataEnv(self.cache_dir, self.cache_name)
                self.data_getter[value] = None


    def add_getter(self, name: str, getter: Union[DataProcessor, ProcessorManager]):
        """
        添加一个数据的处理函数
        如果已经存在于data_getter中，则报错
        :param name:
        :param getter:
        :return:
        """
        if isinstance(getter, DataProcessor):
            getter = ProcessorManager([getter])
        if name in self.data_getter and self.data_getter[name] is not None:
            raise Exception('重复添加！')
        self.data_getter[name] = getter
        self.data_envs[name] = DataEnv()

    def show(self):
        """
        打印自己的详细信息
        :return:
        """
        console = Console()
        print('dependency info')
        rich.inspect(self.data_dependency)
        print('\nPM info')
        tb = Table(show_header=True, header_style='bold magenta')
        tb.add_column('Process', style='dim')
        tb.add_column("Input")
        tb.add_column("Output")
        for key, value in self.data_getter.items():
            proc_name = str(key)
            if value is None:
                input_v = 'None'
                output_v = 'None'
            else:
                input_v = str(value.input_keys)
                output_v = str(value.output_keys)
            tb.add_row(proc_name, input_v, output_v)
        console.print(tb)
        del console

    def show_structure(self):
        """
        展示图结构
        :return:
        """
        G = nx.DiGraph()
        for key, value in self.data_dependency.items():
            for elem_v in value:
                G.add_edge(elem_v, key)
        nx.draw(G, with_labels=True)
        plt.show()