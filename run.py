import sys
import argparse
from type_def import *
import funcy as fc
import sys
import importlib
import inspect

from utils import run_tools
import settings
from settings import task_registry
from utils.run_tools import *


def readCommand(argv) -> Dict[str, Any]:
    """
    利用dir函数自动解析parser的options的属性
    自动将属性转化为dict并返回
    :param argv: 命令行参数，即sys.argv[1: ]
    :return: 返回命令行所获取的属性的dict
    """
    usageStr = """
    展示
    """
    parser = argparse.ArgumentParser(usageStr)

    # 首先确定所要执行的任务
    parser.add_argument('--task', dest='task', type=str, help='所要执行的任务类型（train, predict, ...）', default='train')

    # 然后指定模块的目录
    parser.add_argument('--model', dest='working_dir', type=str, help='挂载的模型的路径', default='work/model.py')

    # 获取参数
    options = parser.parse_args(argv)
    #   options的属性中，不含下划线的都是读取到的
    opt_args = list(fc.remove(lambda x: x[0] == '_', options.__dir__()))
    # options.__dir__()
    return_dict = {}
    for ags in opt_args:
        return_dict[ags] = getattr(options, ags)
    return return_dict


def runCommand(param_dict: Dict[str, Any], model_args: StrList = None):
    """

    :param param_dict:
    :param model_args: 与模型相关的参数
    :return:
    """
    if model_args is None:
        model_args = []

    # 获取要执行的任务类型
    print('constructing task and model from command line...', end=' ... ')
    task = param_dict['task']  # ['train', 'ex_train', 'predict', ...]
    task_template = task_registry[task]  # Trainer class, ex_Trainer class, ...

    # 接下来制定具体使用的模型文件，import该文件，该文件要求一定含有model_registry
    working_dir = param_dict['working_dir']
    working_model = importlib.import_module(working_dir)

    # 获取所有可能用到的参数，生成一个argparser，参数来源包括
    #   - task_template
    #       -- class定义
    #       -- __init__与__call__中的参数（会递归对子模块进行相同的遍历过程，得到的参数全部视为该template的参数）
    #   - 模型文件registry中的args配置列表
    #   - 模型文件所有注册的模块，都按照与task_template相同的方法进行解析

    if 'args' in working_model.model_registry:
        working_args = working_model.model_registry['args']  # 模型文件中的args参数配置列表
    else:
        working_args = []
    template_args, init_submodules, call_submodules = run_tools.get_template_params_recursive(task_template, working_model.model_registry)  # runner template的所有参数配置，以及子模块
    working_args.extend(template_args)
    working_params = run_tools.parse_extra_command(working_args, model_args)  # Dict[str, param value] 从命令行能够获取的参数所解析而成
    print('finish')

    # 根据主参数与任务参数，为Template创建模块
    print('instantiating task model...', end=' ... ')
    task_model = instantiate_template_from_param_dict(task_template, working_params, working_model.model_registry)
    print('finish')
    print('running...')
    run_template_from_param_dict(task_model, working_params, working_model.model_registry)


if __name__ == '__main__':
    sys_arg = sys.argv[1:]
    if '+' in sys_arg:
        delim_idx = sys_arg.index('+')
        config_arg, model_arg = sys_arg[:delim_idx], sys_arg[delim_idx + 1:]
        args = readCommand(config_arg)
        runCommand(args, model_arg)
    else:
        args = readCommand(sys_arg)
        runCommand(args)
