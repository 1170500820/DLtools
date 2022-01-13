import sys
import argparse
from type_def import *
import funcy as fc
import sys
import importlib
import inspect


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

    # 首先指定最基本的模型目录
    parser.add_argument('--dir', dest='working_dir', type=str, help='挂载的模型的路径', default='work/model.py')

    # 指定训练的数据文件
    # parser.add_argument('--train_data', dest='train_data', type=str, help='训练数据文件名', required=False)
    # parser.add_argument('--val_data', dest='val_data', type=str, help='评价数据文件名', required=False)

    # 或者指定数据的目录
    parser.add_argument('--data_dir', dest='data_dir', type=str, help='数据所在的目录', required=False)

    # 是否使用CUDA
    parser.add_argument('--no_cuda', dest='use_cuda', help='是否在训练中使用CUDA', action='store_false')
    parser.set_defaults(use_cuda=True)

    # 通用的训练参数
    parser.add_argument('--epoch', dest='epoch', type=int, help='训练的最大epoch数', default=10)
    parser.add_argument('--save_epoch', dest='save_epoch', type=int, help='保存模型的频率（按epoch计）', default=10)
    parser.add_argument('--save', '--save_path', dest='save_path', type=str, help='保存模型的路径，默认为\"工作目录checkpoints/default.ckp\"')
    parser.add_argument('--bsz', '--batch_size', dest='bsz', type=int, help='训练数据的batch大小', default=8)
    parser.add_argument('--acc' '--grad_acc', '--grad_acc_step', dest='acc', type=int, help='梯度累积的步数', default=1)
    #   evaluate相关
    parser.add_argument('--do_eval', dest='do_eval', type=bool, help='是否在训练中进行评价', default=True)
    parser.add_argument('--eval_freq_batch', dest='eval_freq_batch', type=int, help='进行eval的频率（每个epoch中按batch计）', default=100)
    parser.add_argument('--eval_freq_epoch', dest='eval_freq_epoch', type=int, help='进行eval的频率（按epoch计）', default=1)
    parser.add_argument('--eval_start_batch', dest='eval_start_batch', type=int, help='进行eval的开始位置（每个epoch中按batch计）', default=100)
    parser.add_argument('--eval_start_epoch', dest='eval_start_epoch', type=int, help='进行eval的开始位置（按epoch计）', default=1)
    #   输出信息相关
    parser.add_argument('--print_info_freq', dest='print_info_freq', type=int, help='打印loss信息的频率（每个epoch中按batch计）', default=20)

    # 获取参数
    options = parser.parse_args(argv)
    #   options的属性中，不含下划线的都是读取到的
    opt_args = list(fc.remove(lambda x: x[0] == '_', options.__dir__()))
    # options.__dir__()
    return_dict = {}
    for ags in opt_args:
        return_dict[ags] = getattr(options, ags)
    return return_dict


def parseExtraCommand(arg_list: List[MyArgs], argv):
    """
    为额外参数创建parser，并读取参数
    :param arg_list:
    :param argv:
    :return:
    """
    if len(arg_list) == 0:
        return {}
    extra_parser = argparse.ArgumentParser('extra')
    for elem_arg in arg_list:
        name = elem_arg.pop('name')
        # elem_arg['name_or_flags'] = elem_arg.pop('name')
        extra_parser.add_argument(*[name], **elem_arg)

    options = extra_parser.parse_args(argv)
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
    # find model
    working_dir = param_dict['working_dir']
    working_model = importlib.import_module(working_dir)
    working_path = '/'.join(working_dir.split('.')[:-1])

    # get args
    working_args = working_model.model_registry['args']
    working_params = parseExtraCommand(working_args, model_args)

    # 根据主参数与任务参数，为Template创建模块
    # get model, lossFunc, optimzers, evaluator and train/val dataset factory
    model = get_obj_with_param(working_model.model_registry['model'], param_dict, working_params)
    optimizers = model.get_optimizers()
    lossFunc = get_obj_with_param(working_model.model_registry['loss'], param_dict, working_params)
    evaluator = get_obj_with_param(working_model.model_registry['evaluator'], param_dict, working_params)
    dataset_factory = working_model.model_registry['dataset']
    train_dataloader, val_dataloader = get_obj_with_param(dataset_factory, param_dict, working_params)
    # train_dataset_factory = working_model.model_registry['train_data']
    # val_dataset_factory = working_model.model_registry['val_data']
    # train_dataloader = get_obj_with_param(train_dataset_factory, param_dict, working_params)
    # val_dataloader = get_obj_with_param(val_dataset_factory, param_dict, working_params)
    recorder = None
    if 'recorder' in working_model.model_registry:
        recorder = get_obj_with_param(working_model.model_registry['recorder'], param_dict, working_params)
    # train_dataloader = train_dataset_factory(param_dict['train_data'], bsz=param_dict['bsz'])
    # val_dataloader = val_dataset_factory(param_dict['val_data'])

    from train.trainer import Trainer
    trainer = Trainer()
    print(f'start training!')
    trainer(
        train_loader=train_dataloader,
        test_loader=val_dataloader,
        model=model,
        lossFunc=lossFunc,
        evaluator=evaluator,
        optimizers=optimizers,
        recorder=recorder,
        total_epoch=param_dict['epoch'],
        print_info_freq=param_dict['print_info_freq'],
        eval_freq_epoch=param_dict['eval_freq_epoch'],
        eval_freq_batch=param_dict['eval_freq_batch'],
        eval_start_epoch=param_dict['eval_start_epoch'],
        eval_start_batch=param_dict['eval_start_batch'],
        model_save_path=working_path,
        model_save_epoch=param_dict['save_epoch'],
        grad_acc_step=param_dict['acc'],
        do_eval=param_dict['do_eval'],
        use_cuda=param_dict['use_cuda']
    )


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
