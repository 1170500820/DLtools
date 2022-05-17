"""
用于记录模型信息的一些回调函数
"""


class default_train_callback:
    def __call__(model, model_output, loss, loss_output, others) -> dict:
        """
        在每次完成loss计算之后调用
        """
        return {}


def default_eval_callback(model, model_output, evaluator, eval_result, others):
    """
    在每次对整个数据集完成evaluate之后调用
    """
    return {}