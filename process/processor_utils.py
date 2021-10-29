"""
一些常用的DataProcessor的包装，以及方便的函数接口
"""
from data_processor import *


class ListMerger(DataProcessor):
    """
    init params:
        - merge_keys 将被合并的数据名字
        - output_key 合并后输出的数据名字

    对于输入中的merge_keys数据，将其合并
    假定了数据皆为list，使用list的加法合并
    todo 实现更多合并方法，比如set用union，dict用update
    """
    def __init__(self, merge_keys: Iterable[str], output_key: str):
        super(ListMerger, self).__init__(merge_keys, output_key)

    def process(self, data_dicts: dict) -> Union[dict, List[dict]]:
        merged = []
        for value in data_dicts.values():
            merged = merged + value
        return {
            list(self.output_keys)[0]: merged
        }


class Alias(DataProcessor):
    """
    init params:
        - before 修改前的数据的key
        - after 修改后的数据的key
    """
    def __init__(self, before: str, after: str):
        super(Alias, self).__init__(before, after)

    def process(self, data_dicts: dict) -> Union[dict, List[dict]]:
        return {
            list(self.output_keys)[0]: data_dicts[self.input_keys[0]]
        }

