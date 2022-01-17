"""
目录下多文件处理相关的函数
"""
from type_def import *
import os
import re


def dir_deepsearch(root_dir):
    """
    来自https://thispointer.com/python-how-to-get-list-of-files-in-directory-and-sub-directories/
    对一个路径进行遍历搜索，获取其中的所有文件，而不是文件夹
    :param root_dir:
    :return:
    """
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(root_dir)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(root_dir, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + dir_deepsearch(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


def dir_deepsearch_with_pattern(root_dir, pattern):
    """
    对一个路径进行深度遍历搜索，获取所有文件名，不包括文件夹
    然后对文件名进行判断，筛选出符合pattern的
    todo 这个pattern该如何定义呢？使用re模块的话，就需要考虑是否适配r字符串。不过路径似乎不需要太复杂的匹配？绝大多数情况都是看后缀和前缀
    :param root_dir:
    :param pattern:
    :return:
    """
    correct_path = []
    path_found = dir_deepsearch(root_dir)
    for elem_path in path_found:
        path_components = os.path.split(elem_path)
        if re.match(pattern, path_components[-1]) is not None:
            correct_path.append(elem_path)
    return correct_path