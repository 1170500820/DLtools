import pickle
from type_def import *
from data_processor import *

"""
DataEnv
"""


class DataEnv:
    """
    管理一些数据的承载，可视化，保存
    可以视为"数据中转站"
    """
    def __init__(self, cache_dir: str = '', cache_name: str = ''):
        self.cache_dir = cache_dir
        self.cache_name = cache_name if cache_name != '' else 'default'
        self.inspect_funcs: List[Callable[[dict], str]] = []
        self.data = {}
        self.write_func = None
        self.read_func = None

    def load_data(self, data_dict: dict):
        self.data.update(data_dict)

    def inspect(self):
        selfstr = self.__str__()
        added_str = []
        for elem_func in self.inspect_funcs:
            added_str.append(elem_func(self.data))
        result_str = '\n'.join([selfstr] + added_str)
        print(result_str)
        return result_str

    def write(self):
        if self.write_func:
            self.write_func(self.data, self.cache_dir + self.cache_name)
        else:
            pickle.dump(self.data, open(self.cache_dir + self.cache_name, 'wb'))

    def read(self):
        if self.read_func:
            self.read_func(self.cache_dir + self.cache_name)
        else:
            self.data = pickle.load(open(self.cache_dir + self.cache_name, 'rb'))

    def register_write_func(self, write_func: Callable[[dict, str], None]):
        self.write_func = write_func

    def register_read_func(self, read_func: Callable[[str], None]):
        self.read_func = read_func
