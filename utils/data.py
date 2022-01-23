from torch.utils.data import Dataset
from type_def import *
import random


class SimpleDataset(Dataset):
    def __init__(self, iterable_data):
        self.data = iterable_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class RandomPairDataset(Dataset):
    """
    从数据集中随机选取一对数据

    第一个数据为顺序地从data中选取，作为key
    第二个数据则随机选取，与第一个数据组成pair
    数据集将返回Tuple[element1, element2]
    """
    def __init__(self, iterable_data: Sequence):
        self.data = iterable_data
        self.cnt = 0
        self.size = self.__len__()
        self.random_pick = list(range(self.size))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        """
        item应当是下标，不支持通过str读取

        每当调用__getitem__次数超过self.size时，会重制随机数列表
        :param item:
        :return:
        """
        if self.cnt >= self.size:
            self.cnt = 0
            random.shuffle(self.random_pick)
        self.cnt += 1
        return self.data[item], self.data[self.random_pick[item]]
