"""
目录下多文件处理相关的函数
"""
from type_def import *
from pathlib import Path


def dir_deepsearch(root_dir, r_pattern):
    """
    对于root_dir进行深度搜索，找出所有符合特征的文件名
    :param root_dir: 开始deepsearch的根目录
    :param r_pattern: 需要寻找的文件名的特征
    :return:
    """
    root_path = Path(root_dir)
    filenames = list(root_path.rglob(r_pattern))
    return filenames



