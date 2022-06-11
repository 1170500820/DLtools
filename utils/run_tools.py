"""
参数处理相关的工具
"""
from type_def import *
import inspect
import argparse
import funcy as fc
from loguru import logger

default_usage_str = """
使用自动构建的参数parser
"""


class ExpandedTrainer:
    def __init__(self,
                 train_loader: ... = None,
                 valid_loader: ... = None,
                 model: ... = None,
                 lossFunc: ... = None,
                 optimizers: list = None,
                 evaluator: ... = None,
                 recorder: ... = None,
                 train_output_to_loss_format: Callable = None,
                 eval_output_to_read_format: Callable = None,
                 total_epoch=20,
                 print_info_freq=40,
                 eval_freq_batch=200,
                 eval_freq_epoch=11,
                 eval_start_epoch=1,
                 eval_start_batch=10,
                 model_save_epoch=100,
                 model_save_path='.',
                 grad_acc_step=1,
                 do_eval=True,
                 use_cuda=True
                 ):
        print('I\'m here')


"""
从函数/模块中获取参数信息
自动从template中获取该template所需的参数

get_template_params只对template的__init__, __call__, 和内置参数配置列表进行读取，
而get_template_recursive会对template所需的子模块进行递归的排查，找出所有的参数以及子模块
"""


def get_template_params(template_class: ...) -> Tuple[List[MyArgs], StrList, StrList]:
    """
    该函数获取template_class的参数。
    如果template_class是一个class（正常用法），则返回该class的自带参数定义、__init__参数定义与__call__参数定义。
        对于__init__与__call__的参数，有如下规则
            其中annotation为Ellipsis，则为一个模块，需要对该模块递归地调用本函数，[[进一步使用参数初始化]]
            而annotation不为Ellipsis的，根据其notation，为其默认构建参数解析器
        对于自带参数
            自带参数是按照argparser的args格式构建的list
            list中每个dict都包含用argparser构建该参数所需要的信息

    如果template_class是一个function，todo
    :param template_class:
    :return: Tuple[List[MyArgs], StrList, StrList]
        - 第一个list是参数的list
        - 第二个list是子模块的class的__init__调用的子模块的list
        - 第三个list是子模块的class的__call__调用的子模块的list
    """
    def get_function_params_from_sig(sig) -> Tuple[List[MyArgs], StrList]:
        """
        本方法将sig转化为参数信息的列表，利用该列表可以创建一个arg parser
        sig是使用inspect.signature获取到的一个模块的方法的参数列表

        细节：
        忽略self, kwargs, args
        其中self是__init__与__call__都具有的参数，不考虑
        如果submodule没有定义__init__，则会解析出其默认参数args与kwargs，这里也直接去掉。
        :param sig:
        :return:
        """
        call_param_dict = dict(sig.parameters)
        param_model_lst, param_value_lst = [], []
        for k, v in call_param_dict.items():
            if v.annotation == ...:
                param_model_lst.append(v.name)
            else:
                if v.name == 'self' or v.name == 'kwargs' or v.name == 'args':
                    continue
                if hasattr(v, 'default') and v.default is not inspect._empty:
                    param_type = type(v.default)
                elif v.annotation is inspect._empty:
                    param_type = str
                else:
                    param_type = v.annotation

                default_value = None
                if hasattr(v, 'default') and v.default is not inspect._empty:
                    default_value = v.default

                param_value_lst.append({
                    'name': '--' + v.name,
                    'dest': v.name,
                    'type': param_type,
                    'default': default_value,
                })
        return param_value_lst, param_model_lst

    # 首先判断是不是函数
    if inspect.isfunction(template_class):
        # 如果是函数，则直接解析signature，然后获得参数与子模块
        func_sig = inspect.signature(template_class)
        func_param_value_lst, func_param_model_lst = get_function_params_from_sig(func_sig)
        return func_param_value_lst, [], func_param_model_lst

    # 不是函数，那么接下来按模型处理
    if hasattr(template_class, '_arg_lst'):
        inner_arg_lst = getattr(template_class, '_arg_lst')
    else:
        inner_arg_lst = []

    # 获取__init__中的初始化参数
    if hasattr(template_class, '__call__'):
        init_sig = inspect.signature(template_class.__init__)
        init_param_value_lst, init_param_model_lst = get_function_params_from_sig(init_sig)
    else:
        init_param_value_lst, init_param_model_lst = [], []

    # 用同样的方法获取__call__中的参数
    if hasattr(template_class, '__call__'):
        call_sig = inspect.signature(template_class.__call__)
        call_param_value_lst, call_param_model_lst = get_function_params_from_sig(call_sig)
    else:
        call_param_value_lst, call_param_model_lst = [], []

    # 合并内置参数与值参数
    param_dicts = inner_arg_lst
    param_dest_set = set(x['dest'] for x in inner_arg_lst)
    for elem in init_param_value_lst:
        if elem['dest'] not in param_dest_set:  # 去除已经包含在内置参数中的
            param_dicts.append(elem)
            param_dest_set.add(elem['dest'])
    for elem in call_param_value_lst:  # 先加入init，再加入call
        if elem['dest'] not in param_dest_set:
            param_dicts.append(elem)
            param_dest_set.add(elem['dest'])
    return param_dicts, init_param_model_lst, call_param_model_lst


def _get_template_params_without_repeat(
        template_class: ...,
        submodule_set: Set[str],
        module_registry: Dict[str, Any]) \
        -> Tuple[List[MyArgs], StrList, StrList]:
    """
    本函数执行递归查询，对一个template中出现的子模块继续查询参数，直到所有会被用到的模块的参数都已经找到

    细节：
    对于template_class中找到的子模块，首先判断其是否已经出现过。只会对没出现过的进行递归搜索
    :param template_class: 需要解析参数的template
    :param submodule_set: 所有已经出现过的子模块的名字
    :param module_registry: 模块注册表
    :return: Tuple[List[MyArgs], StrList, StrList]
    """
    # 先获取该template的信息
    param_dicts, init_submodules, call_submodules = get_template_params(template_class)

    param_dicts_dest_set = set(list(x['dest'] for x in param_dicts))
    common_submodules = init_submodules + call_submodules

    # 删除submodule list中已经出现过的子模块
    # common_submodule_set = set(common_submodules) - submodule_set
    # common_submodule_list = list(common_submodule_set)
    init_submodules = list(set(init_submodules) - submodule_set)
    call_submodules = list(set(call_submodules) - submodule_set)

    # 更新submodule_set
    submodule_set.update(init_submodules)
    submodule_set.update(call_submodules)

    # 接下来递归的读取submodule的信息，先init再call
    #  init
    for elem_module in init_submodules:
        elem_module_class = module_registry[elem_module]
        elem_param_dicts, elem_init_submodules, elem_call_submodules = _get_template_params_without_repeat(elem_module_class, submodule_set, module_registry)
        # 加入不重复的param
        for elem_param in elem_param_dicts:
            if elem_param['dest'] not in param_dicts_dest_set:
                param_dicts_dest_set.add(elem_param['dest'])
                param_dicts.append(elem_param)

        # 分别更新init_submodules与call_submodules
        init_submodules.extend(elem_init_submodules)
        init_submodules.extend(elem_call_submodules)
    #  call
    for elem_module in call_submodules:
        elem_module_class = module_registry[elem_module]
        elem_param_dicts, elem_init_submodules, elem_call_submodules = _get_template_params_without_repeat(elem_module_class, submodule_set, module_registry)
        # 加入不重复的param
        for elem_param in elem_param_dicts:
            if elem_param['dest'] not in param_dicts_dest_set:
                param_dicts_dest_set.add(elem_param['dest'])
                param_dicts.append(elem_param)

        # 分别更新init_submodules与call_submodules
        call_submodules.extend(elem_init_submodules)
        call_submodules.extend(elem_call_submodules)

    # 去重
    init_submodules = list(set(init_submodules))
    call_submodules = list(set(call_submodules))

    return param_dicts, init_submodules, call_submodules


def get_template_params_recursive(template_class: ..., module_registry: Dict[str, Any]):
    """
    递归的获取template的参数以及其参数模块的参数

    实现上，只是对_get_template_params_without_repeat的一个包装
    :param template_class:
    :param module_registry:
    :return: Tuple[List[MyArgs], StrList, StrList]
        - template的参数以及其所有子模块的参数的列表
        - template的子模块
    """
    param_dicts, init_submodules, call_submodules = _get_template_params_without_repeat(template_class, set(), module_registry)
    return param_dicts, init_submodules, call_submodules


"""
命令行参数处理
"""


def extract_command(parser, command_str_lst: StrList):
    """
    给一个argparser和一个命令行的string list，解析出所有参数，以dict形式返回
    :param parser:
    :param command_str_lst:
    :return: Dict[arg name, arg value]
    """
    # 获取参数
    options = parser.parse_args(command_str_lst)
    # options的属性中，不含下划线的都是读取到的
    opt_args = list(fc.remove(lambda x: x[0] == '_', options.__dir__()))
    # options.__dir__()
    return_dict = {}
    for ags in opt_args:
        return_dict[ags] = getattr(options, ags)
    return return_dict


def parse_extra_command(arg_list: List[MyArgs], model_args: StrList):
    """
    为参数创建parser，并读取参数
    :param arg_list: 参数配置列表
        list of dict
            {
                name: 默认前面已经有'--'了，这里直接用
                dest:
                type: 是类型，而不是str
                help
            }
    :param model_args: 从命令行读取到的参数
    :return:
    """
    if len(model_args) == 0:
        return {}
    extra_parser = argparse.ArgumentParser('extra')
    for elem_arg in arg_list:
        name, dest, atype = elem_arg['name'], elem_arg['dest'], elem_arg['type']
        helpstr = '' if 'help' not in elem_arg else elem_arg['help']
        default_v = None if 'default' not in elem_arg else elem_arg['default']
        if atype == bool:
            # 布尔类型变量是不需要default的，因为他们不出现就是默认为False
            extra_parser.add_argument(name, dest=dest, action='store_true')
        else:
            if default_v:
                extra_parser.add_argument(name, dest=dest, type=atype, help=helpstr, default=default_v)
            else:
                extra_parser.add_argument(name, dest=dest, type=atype, help=helpstr)

    return extract_command(extra_parser, model_args)


"""
将一个template实例化
"""


def get_obj_with_param(factory, params_dict: Dict[str, Any], extra_dict: Dict[str, Any]):
    """
    找出factory所需参数，然后从两个dict中给他找
    :param factory:
    :param params_dict:
    :param extra_dict:
    :return:
    """
    input_dict = {}
    factory_params = list(inspect.getfullargspec(factory)[0])
    if factory_params[0] == 'self':
        factory_params = factory_params[1:]

    for elem_arg in factory_params:
        if elem_arg in extra_dict:  # extra的优先级更高
            input_dict[elem_arg] = extra_dict[elem_arg]
        elif elem_arg in params_dict:
            input_dict[elem_arg] = params_dict[elem_arg]
        else:
            pass
    return factory(**input_dict)


def instantiate_class_with_dict(any_class, param_dict):
    """
    对于一个实现了__init__方法的class，利用param_dict作为参数，对其进行实例化
    是get_obj_with_param的一个更好的实现，get_obj_with_param可以算是历史遗产了
    :param any_class:  class
    :param param_dict: 所有参数，包括submodule和普通参数
    :return:
    """
    factory_params = list(inspect.getfullargspec(any_class.__init__)[0])
    if inspect.isfunction(any_class):
        factory_params = list(inspect.getfullargspec(any_class)[0])
    if factory_params[0] == 'self':
        factory_params = factory_params[1:]

    filtered_param_dict = {}  # 参数配置列表中有些参数用不到，需要按照factory_param过一遍
    for elem_arg in factory_params:
        if elem_arg in param_dict:
            filtered_param_dict[elem_arg] = param_dict[elem_arg]
    instance_of_this_class = any_class(**filtered_param_dict)
    return instance_of_this_class


def instantiate_template_from_param_dict(template_class, param_dict: Dict[str, Any], module_registry: dict):
    """
    利用param_dict递归地构造template_class，返回一个初始化完成的对象
    如果template_class.__init__的参数中有其他template，则会递归的将其初始化，然后
    初始化当前template
    :param template_class:
    :param param_dict: 只包含参数，不包含模块的dict
    :param module_registry:
    :return:
    """
    # 首先获取内置参数和子模块列表
    param_lst, init_module_lst, call_module_lst = get_template_params_recursive(template_class, module_registry)

    # 首先初始化每一个子模块
    submodule_dict = {}
    for elem_submodule_name in init_module_lst:
        submodule_class = module_registry[elem_submodule_name]
        submodule = instantiate_template_from_param_dict(submodule_class, param_dict, module_registry)
        submodule_dict[elem_submodule_name] = submodule

    # 然后用子模块和参数初始化当前模型
    param_dict.update(submodule_dict)
    current_model = instantiate_class_with_dict(template_class, param_dict)

    return current_model


def instantiate_template_from_param_str(template_class, param_str: StrList, module_registry: Dict[str, Any]):
    # 获取内置参数和子模块列表
    param_lst, init_module_lst, call_module_lstg = get_template_params_recursive(template_class, module_registry)
    param_dict = parse_extra_command(param_lst, param_str)

    # 然后用子模块和参数初始化当前模型
    current_model = instantiate_template_from_param_dict(template_class, param_dict, module_registry)

    return current_model


"""
运行一个instance
"""


def run_instance_with_dict(instance, param_dict):
    """
    对一个实现了__call__的class的实例，直接利用param_dict中的参数运行
    :param instance:
    :param param_dict:
    :return:
    """
    factory_params = list(inspect.getfullargspec(instance.__call__)[0])
    if factory_params[0] == 'self':
        factory_params = factory_params[1:]

    filtered_param_dict = {}  # 参数配置列表中有些参数用不到，需要按照factory_param过一遍
    for elem_arg in factory_params:
        if elem_arg in param_dict:
            filtered_param_dict[elem_arg] = param_dict[elem_arg]
    print('[run_instance_with_dict]finished initializing all params and submodules, running task now')
    instance_return_info = instance(**filtered_param_dict)
    return instance_return_info


def run_template_from_param_dict(template, param_dict: Dict[str, Any], module_registry: dict):
    # 首先获取内置参数和子模块列表
    param_lst, init_module_lst, call_module_lst = get_template_params_recursive(template.__class__, module_registry)
    call_module_lst.reverse()
    init_module_lst.reverse()

    # 首先初始化每一个子模块
    # print('initializing running essentials...')
    submodule_dict = {}
    for elem_submodule_name in call_module_lst:
        # print(f'initializing {elem_submodule_name}...', end=' ... ')
        submodule_class = module_registry[elem_submodule_name]
        submodule = instantiate_template_from_param_dict(submodule_class, param_dict, module_registry)
        submodule_dict[elem_submodule_name] = submodule
        # print('finish')

        # 然后用子模块和参数运行当前模型
        param_dict.update(submodule_dict)
    # print('running')
    instance_return_info = run_instance_with_dict(template, param_dict)
    return param_dict, instance_return_info


if __name__ == '__main__':
    inner, model, value = get_template_params(ExpandedTrainer)
    print(f'inner:{inner}')
    print(f'model:{model}')
    print(f'value:{value}')
