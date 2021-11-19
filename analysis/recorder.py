"""
用于从model中抽取数据的recorder类
因为recorder与record相互依赖，所以目前放在analysis目录下没什么问题？
"""
from type_def import *
from evaluate.evaluator import BaseEvaluator
import torch
import torch.nn as nn
import pickle
import funcy as fc
import copy
import time


class BaseRecorder:
    """
    Recorder负责记录一个模型的训练、预测的状态

    下面是设计Recorder的一些初始想法
        - Recorder会在模型训练过程中的每个关键时间点，分别接受train parameters，model，input，gt，lossFunc，loss，evaluator
        等作为参数，同时输入当前的batch与epoch（如果按照step算，则输入step。）
        - Recorder会对这些输入的模块、数据，提取其中所需的信息，保存并定期输出。Recorder约定不会对这些模块与数据的内部状态做任何改变。
        - Recorder的输出为Dict[str, Any]，str指明数据的类型，Any为数据的本体。
        - 在recorder.py中计划实现一系列可复用的function与class来模块化地组装一个训练、测试过程所需的recorder

    关于Recorder中设计的一些思考
        - 输出为Dict[str, Any]，就能够与具体的Recorder实现细节解耦合：无论什么样的设计，确定什么样的时间点，使用什么样的输入，记录什
        什么样的数据，最后都会被浓缩为Dict，再根据类型调用一些分析函数。考虑到Recorder在试验阶段可能会经常改变实现，这个非常重要。

    Recorder的内部结构
        [!] Recorder仍然处于试验阶段，许多设计不一定会保留。
        - record: Dict[str, Any] 用于存储所有信息的统一数据结构。注意Recorder输出的不一定是完整的record。不管是采用多次输出还是累积
        输出，Recorder输出方法的重点都在于保证输出的dict的内部一致性。

    Recorder的时间点设定
      · record hook
        记录信息函数
        - record_train_parameters
            在训练开始前，记录当前训练的一些参数
        - record_before_forward(train_input, train_gt, full_model)
        - record_after_forward(model_output, full_model)
            分别在数据输入模型进行forward的前后，记录训练数据，label与输出，以及forward前后的模型本体
        - record_before_backward(loss_func)
        - record_after_backward(loss_output, loss_func)
            分别在数据输入loss进行backward的前后，记录loss_output，以及backward前后的loss本体
        - record_before_evaluate(evaluator)
        - record_after_evaluate(evaluator)
            分别在evaluate前后，记录evaluator的前后状态。
            没有记录pred与gt，是因为默认在evaluator里面会记录好。注意这个设计会导致Recorder与Evaluator耦合，Recorder需要去Evaluator
            中读取一次eval的所有pred与gt，但是Evaluator的规约并没有保证这个。这也是无奈的选择，因为pred与gt必须要经过转换才能存储，否则
            会始终占用显存。但是专门写一个转换函数，就又多了一个需要自己编写的结构（model, loss, evaluator, recorder, dataset
            factory, *converter?），所以此处只好一个妥协，选择了模型的编写上的方便。
      · check hook
        记录位置函数
        - train_checkin(step_info)
            在每个train循环，也就是一个forward-backward过程前调用。输入的step_info用于记录当前的时间点。
            一般情况下，如果是step模式，step_info就是int类型；若是epoch模式，step_info就是(epoch, batch)元组。不过这里并没有对其具体
            类型作约束。
        - train_checkpoint
            在每个forward-backward循环结束后调用，将存储的信息输出。
            每个train_checkpoint调用一定会和前面的一个train_checkin调用对应，涵盖有且仅有一个F-B过程。
            每次都会调用，但不一定每次都会输出。
        - eval_checkin(step_info)
            在每个eval循环之前调用。step_info的形式暂不确定，只是觉得evaluator可能也需要一个计数工具，所以保留这个参数
        - eval_checkpoint
            在每个eval循环之后调用，一定会遇一个eval_checkin对应。
            同样，每次都会调用，但并不意味着每次都要输出。
        - checkpoint
            在整个模型训练、测试流程的结束阶段调用。大多数时候在这之前我就喊停了，所以一般不会运行到这里。
            不过还是得兜个底。
    """

    record = {}

    def record_train_parameters(self, parameters: Dict[str, Any]):
        self.record['train_parameters'] = parameters

    def record_before_forward(self, train_input: Dict[str, Any], train_gt: Dict[str, Any], full_model: nn.Module):
        pass

    def record_after_forward(self, model_output: Dict[str, Any], full_model: nn.Module):
        pass

    def record_before_backward(self, loss_func: nn.Module):
        pass

    def record_after_backward(self, loss_output: torch.Tensor, loss_func: nn.Module):
        pass

    def record_before_evaluate(self, evaluator: BaseEvaluator):
        pass

    def record_after_evaluate(self, model: nn.Module, evaluator: BaseEvaluator, eval_result: Dict[str, Any]):
        pass

    def train_checkin(self, step_info):
        pass

    def train_checkpoint(self):
        pass

    def eval_checkin(self, step_info = None):
        pass

    def eval_checkpoint(self):
        pass

    def checkpoint(self):
        pass


class NaiveRecorder(BaseRecorder):
    """
    一个简单的recorder的例子

    先做一个pickle输出的范例出来吧。

    发现保存才是比较关键而且比较复杂的问题。
    提供的接口能够组合成的保存方式太多了，反而不知道如何下手了
    有几个关键问题：
        - 每次直接输出新的覆盖旧的，还是每次在旧的上面增添

    """
    def __init__(self, save_path):
        self.filename = '_'.join(list(map(str, list(time.localtime())[:6])))
        self.save_name = save_path + '/' + self.filename
        self.evaluate_cnt = 0
        self.record['preds'], self.record['gts'], self.record['scores'] = [], [], []
        pickle.dump(self.record, open(self.save_name, 'wb'))

    def record_before_evaluate(self, evaluator: BaseEvaluator):
        preds = evaluator.pred_lst
        gts = evaluator.gt_lst
        self.record['preds'].append(preds)
        self.record['gts'].append(gts)

    def record_after_evaluate(self, model: nn.Module, evaluator: BaseEvaluator, eval_result: Dict[str, Any]):
        self.record['scores'].append(eval_result)

    def eval_checkpoint(self):
        pickle.dump(self.record, open(self.save_name, 'wb'))
        self.evaluate_cnt += 1


def update_sequence_dict(seq_data_dict: dict, model_attr_dict: dict, step_key: str = None, step: int = None):
    """
    给定用来存的数据dict，以及刚刚读取到的属性dict，将属性通过sequence的形式更新到数据中
    如果提供了step和step_key，则同时更新step信息
    :param seq_data_dict: 用来装数据内容的dict，应当是是record['data']
    :param model_attr_dict: 从模组中抽取出来的record属性，已经进行了deepcopy
    :param step_key: 存储step值的list所对应的key
    :param step: 当前的key值
    :return:
    """
    # 首先将特殊属性与数据本体分开存放在不同的dict当中
    d_data, d_attr = {}, {}
    for key, value in model_attr_dict.items():
        if key[0] == '_':
            d_attr[key] = value
        else:
            d_data[key] = value

    # 接下来将数据更新到用于存放的dict中
    # 如果是第一次存放，那么需要创建用于存放的list，还得创建属性列表
    for key, value in d_data.items():
        if key not in seq_data_dict:  # 不存在，因此这时候是第一次存放
            seq_data_dict[key] = [value]
            if '_' + key in d_attr:
                dimension_lst = ['step'] + d_attr['_' + key]
            else:
                dimension_lst = ['step', 'Unknown']
            seq_data_dict['_' + key] = dimension_lst
        else:  # 存在，那么只需要把数据append进行即可
            seq_data_dict[key].append(value)
    if '_' + step_key not in seq_data_dict:
        seq_data_dict['_' + step_key] = ['step', 'step cnt int']
    if step_key is not None and step is not None:
        if step_key not in seq_data_dict:
            seq_data_dict[step_key] = [step]
        else:
            seq_data_dict[step_key].append(step)


class AutoRecorder(BaseRecorder):
    """
    AutoRecorder自动将model、loss和evaluator当中的前缀的RECORDER_的dict中的内容读取

    AutoRecorder是一个sequence recorder，意为它将会对每次record到的内容进行扩充。
    从model和loss读取到的内容将被组织为train_step维度。
    而从evaluator读取到到内容将被组织为evaluate_step维度。
    record是BaseRecorder的内置属性，有key值meta和data。
    meta保留，目的为存放当前record的一些配置信息
    data是一个dict，存储所有从其他组件属性中record到的数据。



    1, 记录训练参数，key=train_parameters
    2, 在每次forward后记录一次model参数。
        在key=model下.step存在key=model_step


    """
    def __init__(self, save_path, train_record_freq=50, eval_record_freq=1):
        """

        :param save_path:
        :param train_record_freq: 每多少次forward-backward，进行一次checkpoint
        :param eval_record_freq: 每多少次eval，进行一次checkpoint
        """
        self.filename = 'record-' + '_'.join(list(map(str, list(time.localtime())[:6])))
        self.save_name = save_path + '/' + self.filename
        self.evaluate_cnt = 0

        # 分别用train_step和eval_step记录当前所处的位置
        # 用train_freq与eval_freq记录保存的间隔
        # 每个checkpoint点，需要判断是否整除
        self.train_freq, self.eval_freq = train_record_freq, eval_record_freq
        self.train_step, self.eval_step = 0, 0

        self.record['meta'] = []
        self.record['data'] = {
            'train_step': [],
            'evaluate_step': [],
            'loss_step': []
        }
        pickle.dump(self.record, open(self.save_name, 'wb'))

    def record_train_parameters(self, parameters: Dict[str, Any]):
        super(AutoRecorder, self).record_train_parameters(parameters)
        pickle.dump(self.record, open(self.save_name, 'wb'))

    def record_after_forward(self, model_output: Dict[str, Any], full_model: nn.Module):
        if self.train_step % self.train_freq != 0:
            return
        model_attrs = list(fc.remove(lambda x: x[:9] != 'RECORDER_', dir(full_model)))
        d = {}

        for elem_attr in model_attrs:
            d.update(copy.deepcopy(getattr(full_model, elem_attr)))

        update_sequence_dict(self.record['data'], d, 'train_step', self.train_step)

    def record_after_backward(self, loss_output: torch.Tensor, loss_func: nn.Module):
        if self.train_step % self.train_freq != 0:
            return
        model_attrs = list(fc.remove(lambda x: x[:9] != 'RECORDER_', dir(loss_func)))
        d = {}
        for elem_attr in model_attrs:
            d.update(copy.deepcopy(getattr(loss_func, elem_attr)))

        update_sequence_dict(self.record['data'], d, 'loss_step', self.train_step)

    def record_after_evaluate(self, model: nn.Module, evaluator: BaseEvaluator, eval_result: Dict[str, Any]):
        if self.eval_step % self.eval_freq != 0:
            return
        model_attrs = list(fc.remove(lambda x: x[:9] != 'RECORDER_', dir(evaluator)))
        d = {}
        for elem_attr in model_attrs:
            d.update(copy.deepcopy(getattr(evaluator, elem_attr)))

        update_sequence_dict(self.record['data'], d, 'evaluate_step', self.eval_step)

    def train_checkin(self, step_info):
        self.train_step += 1

    def eval_checkin(self, step_info=None):
        self.eval_step += 1

    def train_checkpoint(self):
        pickle.dump(self.record, open(self.save_name, 'wb'))

    def eval_checkpoint(self):
        pickle.dump(self.record, open(self.save_name, 'wb'))
