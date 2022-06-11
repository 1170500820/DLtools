"""
实现一系列方便调用的可视化方法
"""
from type_def import *
import matplotlib.pyplot as plt


"""
简单图

- 画出值的分布范围
"""


def plot_value_distribution_1d(value_seq: Iterable[float]):
    """
    画出value_seq中的值的分布图

    真的要写起来，发现这个东西还不是很容易啊。要通用的对所有数据都能画，不容易。

    :param value_seq:
    :return:
    """
    # value_cnt = {}
    # max_value, min_value = -9999999, 999999999
    # for elem_value in value_seq:
    #     if elem_value > max_value:
    #         max_value = elem_value
    #     if elem_value < min_value:
    #         min_value = elem_value
    #     if elem_value in value_cnt:
    #         value_cnt[elem_value] = value_cnt[elem_value] + 1
    #     else:
    #         value_cnt[elem_value] = 1
    #
    # value_range = max_value - min_value
    #
    value_lst = list(value_seq)
    plt.hist(value_lst, bins=list(range(60)))
    plt.show()


def raw_text_print(text: str):
    """
    最简单的文本输入，直接输出
    :param text:
    :return:
    """
    print(text)