from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
import numpy as np

from type_def import *
from evaluate.evaluator import dict2fstring, BaseEvaluator
from analysis.recorder import BaseRecorder
from torch.utils.data import DataLoader
import pickle


def convert_to_cuda(input_dict: dict, multi: bool = False):
    result_dict = {}
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            if multi:
                result_dict[k] = v.cuda(non_blocking=True)
            else:
                result_dict[k] = v.cuda()
        else:
            result_dict[k] = v
    return result_dict


def convert_to_cuda_multigpu(input_dict: dict):
    result_dict = {}
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            result_dict[k] = v.cuda(non_blocking=True)
        else:
            result_dict[k] = v
    return result_dict


class Trainer:
    """
    一个通用的训练流程
    需要提供训练所需的组件

    加入了对单机多卡的支持：
    - 所以eval、record与log输出只会在主卡进行
    """
    main_local_rank = 0

    def __call__(
            self,
            train_val_data: ...,
            model: ...,
            loss: ...,
            evaluator: ...,
            recorder: ...,
            local_rank: int = -1,
            train_loader: DataLoader = None,
            test_loader: DataLoader = None,
            total_epoch=40,
            print_info_freq=50,
            eval_freq_batch=250,
            eval_freq_epoch=1,
            eval_start_epoch=1,
            eval_start_batch=10,
            model_save_epoch=100,
            model_save_path='.',
            grad_acc_step=1,
            do_eval=True,
            use_cuda=True,
            save_best=False,
            control_name: str = 'default'):
        if train_val_data is not None:
            train_loader, test_loader = train_val_data
        lossFunc = loss
        if local_rank in [-1, 0]:
            print(f'Training Settings:'
                  f'\n\ttotal_epoch:{total_epoch or "None"}'
                  f'\n\tprint_info_freq:{print_info_freq or "None"}'
                  f'\n\teval_freq_epoch:{eval_freq_epoch or "None"}'
                  f'\n\teval_freq_batch:{eval_freq_batch or "None"}'
                  f'\n\teval_start_epoch:{eval_start_epoch or "None"}'
                  f'\n\tmodel_save_epoch:{model_save_epoch or "None"}'
                  f'\n\tmodel_save_name:{model_save_path or "None"}'
                  f'\n\tgrad_acc_step:{grad_acc_step or "None"}'
                  f'\n')
            print(f'Data Info:'
                  f'\n\ttrain data cnt:{len(train_loader.dataset)}'
                  # f'\n\tval data cnt:{len(test_loader.dataset or [])}'
                  )
        print(f'模型参数:{sum([np.prod(list(p.size())) for p in model.parameters()])}')
        if local_rank != -1:
            device = torch.device('cuda', local_rank)
            model.to(device)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
            optimizers = model.module.get_optimizers()
        elif use_cuda:
            model.cuda()
            optimizers = model.get_optimizers()
        else:  # 可能是想用gpu训练吧
            optimizers = model.get_optimizers()

        new_high_score = False

        for i_epoch in range(total_epoch):  # todo 加入梯度累积
            epoch_avg_loss = 0.0
            for i_batch, train_sample in enumerate(iter(train_loader)):
                model.train()
                if recorder and local_rank in [-1, self.main_local_rank]:
                    recorder.train_checkin((i_epoch, i_batch))
                train_input, train_gt = train_sample
                if recorder and local_rank in [-1, self.main_local_rank]:
                    recorder.record_before_forward(train_input=train_input, train_gt=train_gt, full_model=model)
                if use_cuda:
                    train_input, train_gt = convert_to_cuda(train_input), convert_to_cuda(train_gt)
                model_output = model(**train_input)
                if recorder and local_rank in [-1, self.main_local_rank]:
                    recorder.record_after_forward(model_output=model_output, full_model=model)
                    recorder.record_before_backward(loss_func=lossFunc)
                loss = lossFunc(**model_output, **train_gt)
                loss = loss / grad_acc_step
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 500)
                if recorder and local_rank in [-1, self.main_local_rank]:
                    recorder.record_after_backward(loss_output=loss, loss_func=lossFunc)
                    recorder.train_checkpoint()
                epoch_avg_loss += float(loss)
                if (i_batch + 1) % print_info_freq == 0 and local_rank in [-1, self.main_local_rank]:
                    print(
                        f'epoch {i_epoch + 1} |batch {i_batch + 1} |loss:{loss.float():<8.5f} |avg:{epoch_avg_loss / (i_batch + 1):<8.5f}| norm: {norm.cpu().item()}')
                if (i_batch + 1) % grad_acc_step == 0:
                    # 只有在一次梯度累积完成之后，才可以进行step与evaluate
                    # 否则会浪费梯度
                    for opt in optimizers:
                        opt.step()
                    model.zero_grad()
                    if ((i_batch + 1) % eval_freq_batch == 0
                        and (i_epoch + 1) >= eval_start_epoch) \
                            and ((i_epoch + 1) % eval_freq_epoch == 0) \
                            and (((i_epoch + 1) < eval_start_epoch) or ((i_batch + 1) >= eval_start_batch)) and local_rank in [-1, self.main_local_rank]:
                        # evaluate
                        model.eval()
                        if recorder:
                            recorder.eval_checkin()
                        for test_sample in tqdm(iter(test_loader)):
                            inputs, gts = test_sample
                            if use_cuda:
                                inputs, gts = convert_to_cuda(inputs), convert_to_cuda(gts)
                            model_output = model(**inputs)
                            evaluator.eval_single(**model_output, **gts)
                            del model_output
                            del gts
                        if recorder:
                            recorder.record_before_evaluate(evaluator)
                        eval_info = evaluator.eval_step()
                        print(dict2fstring(eval_info))

                        if new_high_score and save_best:
                            print('reaching new high score, saving...')
                            torch.save(model.state_dict(),
                                   model_save_path + '/checkpoint/' + f'{control_name}-save-best.pth')
                            pickle.dump(model.init_params,
                                    open(model_save_path + '/checkpoint/' + f'{control_name}-init_param-best.pk', 'wb'))
                        model.train()
                        if recorder:
                            recorder.eval_checkpoint()
            model.zero_grad()
            if (i_epoch + 1) % model_save_epoch == 0 and local_rank in [-1, self.main_local_rank] and save_best:
                print(f'saving model as [{model_save_path}/save-{i_epoch + 1}.pth]...')
                # torch.save(model.state_dict(), model_save_name)
                torch.save(model.state_dict(), model_save_path + '/checkpoint/' + f'{control_name}-save-{i_epoch + 1}-final.pth')
                pickle.dump(model.init_params, open(model_save_path + '/checkpoint/' + f'{control_name}-init_param-final.pk', 'wb'))
                print(f'finished saving')
        if recorder and local_rank in [-1, self.main_local_rank]:
            recorder.checkpoint()


class ExpandedTrainer:
    def __call__(self,
                 train_loader: DataLoader = None,
                 valid_loader: DataLoader = None,
                 model: nn.Module = None,
                 lossFunc: nn.Module = None,
                 optimizers: list = None,
                 evaluator: BaseEvaluator = None,
                 recorder: BaseRecorder = None,
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
        """

        :param train_loader:
        :param valid_loader:
        :param model:
        :param lossFunc:
        :param optimizers:
        :param evaluator:
        :param recorder:
        :param train_output_to_loss_format:
        :param eval_output_to_read_format:
        :param total_epoch:
        :param print_info_freq:
        :param eval_freq_batch:
        :param eval_freq_epoch:
        :param eval_start_epoch:
        :param eval_start_batch:
        :param model_save_epoch:
        :param model_save_path:
        :param grad_acc_step:
        :param do_eval:
        :param use_cuda:
        :return:
        """
        # 将模型移到cuda
        if use_cuda:
            model.cuda()

        # 开始training loop
        for i_epoch in range(total_epoch):  # todo 加入梯度累积
            epoch_avg_loss = 0.0

            # 对每一个batch
            for i_batch, train_sample in enumerate(iter(train_loader)):

                # 更新recorder状态
                if recorder:
                    recorder.train_checkin((i_epoch, i_batch))

                # 获得该batch对input与gt，并更新recorder状态；如果模型在cuda下运行，自动将tensor移动到cuda
                train_input, train_gt = train_sample
                if recorder:
                    recorder.record_before_forward(train_input=train_input, train_gt=train_gt, full_model=model)
                if use_cuda:
                    train_input, train_gt = convert_to_cuda(train_input), convert_to_cuda(train_gt)

                # 将训练数据送入模型，并更新recorder状态
                model_output = model(**train_input)
                if recorder:
                    recorder.record_after_forward(model_output=model_output, full_model=model)
                    recorder.record_before_backward(loss_func=lossFunc)
                converted_model_output = train_output_to_loss_format(**model_output)

                # 计算loss并backward；更新recorder状态
                loss = lossFunc(**converted_model_output, **train_gt)
                loss = loss / grad_acc_step
                loss.backward()
                if recorder:
                    recorder.record_after_backward(loss_output=loss, loss_func=lossFunc)
                    recorder.train_checkpoint()

                # 计算loss value，并显示统计信息，打印到终端
                epoch_avg_loss += float(loss)
                if (i_batch + 1) % print_info_freq == 0:
                    print(
                        f'epoch {i_epoch + 1} |batch {i_batch + 1} |loss:{loss.float():<8.5f} |avg:{epoch_avg_loss / (i_batch + 1):<8.5f}|')
                if (i_batch + 1) % grad_acc_step == 0:
                    # 只有在一次梯度累积完成之后，才可以进行step与evaluate
                    # 否则会浪费梯度
                    for opt in optimizers:
                        opt.step()
                    model.zero_grad()
                    if ((i_batch + 1) % eval_freq_batch == 0
                        and (i_epoch + 1) >= eval_start_epoch) \
                            and ((i_epoch + 1) % eval_freq_epoch == 0) \
                            and (((i_epoch + 1) < eval_start_epoch) or ((i_batch + 1) >= eval_start_batch)):
                        # evaluate
                        model.eval()
                        if recorder:
                            recorder.eval_checkin()
                        for test_sample in tqdm(iter(valid_loader)):
                            inputs, gts = test_sample
                            if use_cuda:
                                inputs, gts = convert_to_cuda(inputs), convert_to_cuda(gts)
                            model_output = model(**inputs)
                            converted_output = eval_output_to_read_format(**model_output)
                            evaluator.eval_single(**converted_output, **gts)
                            del model_output
                            del converted_output
                            del gts
                        if recorder:
                            recorder.record_before_evaluate(evaluator)
                        eval_info = evaluator.eval_step()
                        if recorder:
                            recorder.record_after_evaluate(model, evaluator, eval_info)
                        print(dict2fstring(eval_info))
                        model.train()
                        if recorder:
                            recorder.eval_checkpoint()
            model.zero_grad()
            if (i_epoch + 1) % model_save_epoch == 0:
                print(f'saving model as [{model_save_path}/save-{i_epoch + 1}.pth]...')
                # torch.save(model.state_dict(), model_save_name)
                torch.save(model.state_dict(), model_save_path + '/checkpoint/' + f'{model.__class__.__name__}-save-{i_epoch + 1}.pth')
                pickle.dump(model.init_params, open(model_save_path + '/checkpoint/' + f'{model.__class__.__name__}-init_param.pk', 'wb'))
                print(f'finished saving')
        if recorder:
            recorder.checkpoint()


class MultiGPUTrainer:
    """
    支持多GPU运行的Trainer
    """
    main_local_rank = 0  # 默认以序号为0作为主进程卡

    def __call__(
            self,
            train_val_data: ...,
            model: ...,
            loss: ...,
            evaluator: ...,
            recorder: ...,
            local_rank: int = -1,
            train_loader: DataLoader = None,
            test_loader: DataLoader = None,
            total_epoch=40,
            print_info_freq=50,
            eval_freq_batch=250,
            eval_freq_epoch=1,
            eval_start_epoch=1,
            eval_start_batch=10,
            model_save_epoch=100,
            model_save_path='.',
            grad_acc_step=1,
            do_eval=True):
        if train_val_data is not None:
            train_loader, test_loader = train_val_data
        lossFunc = loss
        if local_rank == self.main_local_rank:
            print(f'训练基础参数:'
                  f'\n\ttotal_epoch:{total_epoch or "None"}'
                  f'\n\tprint_info_freq:{print_info_freq or "None"}'
                  f'\n\teval_freq_epoch:{eval_freq_epoch or "None"}'
                  f'\n\teval_freq_batch:{eval_freq_batch or "None"}'
                  f'\n\teval_start_epoch:{eval_start_epoch or "None"}'
                  f'\n\tmodel_save_epoch:{model_save_epoch or "None"}'
                  f'\n\tmodel_save_name:{model_save_path or "None"}'
                  f'\n\tgrad_acc_step:{grad_acc_step or "None"}'
                  f'\n')
            print(f'Data Info:'
                  f'\n\ttrain data cnt:{len(train_loader.dataset)}'
                  # f'\n\tval data cnt:{len(test_loader.dataset or [])}'
                  )
        device = torch.device('cuda', local_rank)
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        optimizers = model.module.get_optimizers()

        for i_epoch in range(total_epoch):
            epoch_avg_loss = 0.0
            for i_batch, train_sample in enumerate(iter(train_loader)):
                if recorder and local_rank == self.main_local_rank:
                    recorder.train_checkin((i_epoch, i_batch))
                train_input, train_gt = train_sample
                if recorder and local_rank == self.main_local_rank:
                    recorder.record_before_forward(train_input=train_input, train_gt=train_gt, full_model=model)
                train_input, train_gt = convert_to_cuda_multigpu(train_input), convert_to_cuda_multigpu(train_gt)
                model_output = model(**train_input)
                if recorder and local_rank == self.main_local_rank:
                    recorder.record_after_forward(model_output=model_output, full_model=model)
                    recorder.record_before_backward(loss_func=lossFunc)
                loss = lossFunc(**model_output, **train_gt)
                loss = loss / grad_acc_step
                loss.backward()
                if recorder and local_rank == self.main_local_rank:
                    recorder.record_after_backward(loss_output=loss, loss_func=lossFunc)
                    recorder.train_checkpoint()
                epoch_avg_loss += float(loss)
                if (i_batch + 1) % print_info_freq == 0 and local_rank == self.main_local_rank:
                    print(
                        f'epoch {i_epoch + 1} |batch {i_batch + 1} |loss:{loss.float():<8.5f} |avg:{epoch_avg_loss / (i_batch + 1):<8.5f}|')
                if (i_batch + 1) % grad_acc_step == 0:
                    # 只有在一次梯度累积完成之后，才可以进行step与evaluate
                    # 否则会浪费梯度
                    for opt in optimizers:
                        opt.step()
                    model.zero_grad()
                    if ((i_batch + 1) % eval_freq_batch == 0
                        and (i_epoch + 1) >= eval_start_epoch) \
                            and ((i_epoch + 1) % eval_freq_epoch == 0) \
                            and (((i_epoch + 1) < eval_start_epoch) or ((i_batch + 1) >= eval_start_batch)) and local_rank in [-1, self.main_local_rank]:
                        # evaluate
                        model.eval()
                        if recorder:
                            recorder.eval_checkin()
                        for test_sample in tqdm(iter(test_loader)):
                            inputs, gts = test_sample
                            inputs, gts = convert_to_cuda_multigpu(inputs), convert_to_cuda_multigpu(gts)
                            model_output = model(**inputs)
                            evaluator.eval_single(**model_output, **gts)
                            del model_output
                            del gts
                        if recorder:
                            recorder.record_before_evaluate(evaluator)
                        eval_info = evaluator.eval_step()
                        if recorder:
                            recorder.record_after_evaluate(model, evaluator, eval_info)
                        print(dict2fstring(eval_info))
                        model.train()
                        if recorder:
                            recorder.eval_checkpoint()
            model.zero_grad()
            if (i_epoch + 1) % model_save_epoch == 0 and local_rank == self.main_local_rank:
                print(f'saving model as [{model_save_path}/save-{i_epoch + 1}.pth]...')
                # torch.save(model.state_dict(), model_save_name)
                torch.save(model.state_dict(), model_save_path + '/checkpoint/' + f'{model.__class__.__name__}-save-{i_epoch + 1}.pth')
                pickle.dump(model.init_params, open(model_save_path + '/checkpoint/' + f'{model.__class__.__name__}-init_param.pk', 'wb'))
                print(f'finished saving')
        if recorder and local_rank == self.main_local_rank:
            recorder.checkpoint()
