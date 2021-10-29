from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from evaluate.evaluator import dict2fstring, BaseEvaluator
from analysis.recorder import BaseRecorder
from torch.utils.data import DataLoader
import pickle


class Trainer:
    """
    一个通用的训练流程
    需要提供训练所需的组件
    """

    def __call__(
            self,
            train_loader: DataLoader = None,
            test_loader: DataLoader = None,
            model: nn.Module = None,
            optimizers: list = None,
            lossFunc: nn.Module = None,
            evaluator: BaseEvaluator = None,
            recorder: BaseRecorder = None,
            total_epoch=20,
            print_info_freq=40,
            eval_freq_batch=200,
            eval_freq_epoch=11,
            eval_start_epoch=1,
            eval_start_batch=10,
            model_save_epoch=100,
            model_save_path='.',
            grad_acc_step=1,
            do_eval=True):
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
        train_param_record = {
            "train_loader": str(train_loader),
            "test_loader": str(test_loader),
            "model": str(model),
            "optimizers": str(optimizers),
            "lossFunc": str(lossFunc),
            "evaluator": str(evaluator),
            "recorder": str(recorder),
            "total_epoch": str(total_epoch),
            "print_info_freq": str(print_info_freq),
            "eval_freq_batch": str(eval_freq_batch),
            'eval_freq_epoch': str(eval_start_epoch),
            "eval_start_epoch": str(eval_start_epoch),
            "model_save_epoch": str(model_save_epoch),
            "model_save_path": str(model_save_path),
            "grad_acc_step": str(grad_acc_step)
        }
        model.cuda()
        def convert_to_cuda(input_dict: dict):
            result_dict = {}
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    result_dict[k] = v.cuda()
                else:
                    result_dict[k] = v
            return result_dict
        for i_epoch in range(total_epoch):  # todo 加入梯度累积
            epoch_avg_loss = 0.0
            for i_batch, train_sample in enumerate(iter(train_loader)):
                if recorder:
                    recorder.train_checkin((i_epoch, i_batch))
                train_input, train_gt = train_sample
                if recorder:
                    recorder.record_before_forward(train_input=train_input, train_gt=train_gt, full_model=model)
                train_input, train_gt = convert_to_cuda(train_input), convert_to_cuda(train_gt)
                model_output = model(**train_input)
                if recorder:
                    recorder.record_after_forward(model_output=model_output, full_model=model)
                    recorder.record_before_backward(loss_func=lossFunc)
                loss = lossFunc(**model_output, **train_gt)
                loss = loss / grad_acc_step
                loss.backward()
                if recorder:
                    recorder.record_after_backward(loss_output=loss, loss_func=lossFunc)
                    recorder.train_checkpoint()
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
                        for test_sample in tqdm(iter(test_loader)):
                            inputs, gts = test_sample
                            inputs, gts = convert_to_cuda(inputs), convert_to_cuda(gts)
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
            if (i_epoch + 1) % model_save_epoch == 0:
                print(f'saving model as [{model_save_path}/save-{i_epoch + 1}.pth]...')
                # torch.save(model.state_dict(), model_save_name)
                torch.save(model.state_dict(), model_save_path + '/checkpoint/' + f'{model.__class__.__name__}-save-{i_epoch + 1}.pth')
                pickle.dump(model.init_params, open(model_save_path + '/checkpoint/' + f'{model.__class__.__name__}-init_param.pk', 'wb'))
                print(f'finished saving')
        if recorder:
            recorder.checkpoint()
