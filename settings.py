"""
定义一些最基本的项目配置参数
"""
from train.trainer import Trainer, ExpandedTrainer
from train.caller import BaseCaller


task_registry = {
    "train": Trainer,
    "ex_train": ExpandedTrainer,
    'call': BaseCaller
}
