import json
import pickle

from type_def import *

"""
基础定义部分
"""

# 存放中间数据的位置
temp_path = 'temp_data/'

# 源数据路径
initial_dataset_path = ''

# 数据类型
dataset_type = 'duie'


"""
对数据进行一些简单分析
"""


"""
数据处理的一些工具
"""
def load_jsonl(filename: str) -> List[dict]:
    data = list(json.loads(x) for x in open(filename, 'r', encoding='utf-8').read().strip().split('\n'))
    return data



"""
处理数据的函数
"""


"""
对数据处理的包装函数
"""

def tokenize_data(input_data_filename: str, output_data_filename, subset_name: str = 'train', dataset_type: str = dataset_type):
    pass


if __name__ == '__main__':
    pass

# cache
# simple load/dump
#
