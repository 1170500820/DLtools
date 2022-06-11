"""
生成训练报告的相关工具
"""
from type_def import *


# 只记录这些基本类型的参数；类似function、object的东西不会记录。
available_param_type = {str, bool, int, float}


def arrange_train_info(param_dict: dict, instance_return):
    """
    将参数信息与训练信息进行整理，输出一个训练总结
    :param param_dict: 训练中所涉及的所有参数
    :param instance_return: 训练实例的返回信息
    :return:
    """

    # 首先过滤掉不需要收集的参数信息，比如容器、函数、对象实例
    filtered_param_dict = {}
    for key, value in param_dict.items():
        if type(value) in available_param_type:
            filtered_param_dict[key] = value



def arrange_info(param_dict: dict, instance_return, task_type: str):
    pass


# 生成报告
def generate_train_html_report():
    pass


if __name__ == '__main__':
    pass