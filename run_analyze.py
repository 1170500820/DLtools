import sys
import argparse
from type_def import *
import funcy as fc


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

    # 获取参数
    options = parser.parse_args(argv)
    #   options的属性中，不含下划线的都是读取到的
    opt_args = list(fc.remove(lambda x: x[0] == '_', options.__dir__()))
    # options.__dir__()
    return_dict = {}
    for ags in opt_args:
        return_dict[ags] = getattr(options, ags)
    return return_dict


def runCommand(param_dict: Dict[str, Any]):
    # print('get commands:\n', param_dict)

    pass

if __name__ == '__main__':
    args = readCommand(sys.argv[1:])
    runCommand(args)
