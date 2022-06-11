"""
写一些jupyterlab ui需要用到的函数
"""
import os


def list_files(path: str, prefix: str = ''):
    """
    返回该目录下前缀为prefix的文件名的list
    :param path:
    :param prefix:
    :return:
    """
    dir_lst = []
    dirs = os.listdir(path)
    for elem_dir in dirs:
        if not os.path.isdir(elem_dir) and elem_dir[:len(prefix)] == prefix:
            dir_lst.append(elem_dir)
    return dir_lst

