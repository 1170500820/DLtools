from type_def import *
import json

"""
与文件读写相关的接口
"""


def load_jsonl(filename: str) -> List[dict]:
    """
    读取一个jsonl格式的文件。该文件的每一行都是一个合法的json

    :param filename 要读取的文件名
    """
    data = list(json.loads(x) for x in open(filename, 'r', encoding='utf-8').read().strip().split('\n'))
    return data


def dump_jsonl(data_dicts: List[dict], filename: str):
    """
    将dict的list输出为jsonl文件

    :param data_dicts: dict的list
    :param filename: 输出的文件名
    """
    f = open(filename, 'w', encoding='utf-8')
    for elem in data_dicts:
        f.write(json.dumps(elem, ensure_ascii=False) + '\n')
    f.close()

