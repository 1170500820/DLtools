"""
用于记录模型信息的一些回调函数
"""


def default_train_callback(model, loss) -> dict:
    """
    在每次完成loss计算之后调用
    """
    return {}


def default_eval_callback(model, evaluator):
    """
    在每次对整个数据集完成evaluate之后调用
    """
    return {}