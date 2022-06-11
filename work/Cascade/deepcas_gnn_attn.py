import torch
import torch.nn as nn
import torch.nn.functional as F

from type_def import *
from work.Cascade import cascade_settings
from work.Cascade.cascade_utils import *



def train_dataset_factory(train_file: str, bsz: int = 8):
    """

    :param train_file:
    :param bsz:
    :return:
    """