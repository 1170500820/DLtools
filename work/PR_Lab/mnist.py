import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np
from itertools import chain

from work.PR_Lab import mnist_setting

from type_def import *
from evaluate.evaluator import BaseEvaluator, KappaEvaluator, PrecisionEvaluator
from utils import tools
from utils.data import SimpleDataset


class Naive_Linear(nn.Module):
    def __init__(self, input_dim: int = mnist_setting.input_dim_1, label_cnt: int = mnist_setting.label_cnt_1):
        super(Naive_Linear, self).__init__()
        self.input_dim = input_dim
        self.label_cnt = label_cnt


        self.linear1 = nn.Linear(input_dim, 128 * 8)
        self.batch_norm1 = nn.BatchNorm1d(128 * 8)

        self.linear2 = nn.Linear(128 * 8, 64 * 4)
        self.batch_norm2 = nn.BatchNorm1d(64 * 4)

        self.linear3 = nn.Linear(64 * 4, 32 * 4)
        self.batch_norm3 = nn.BatchNorm1d(32 * 4)

        self.linear4 = nn.Linear(32 * 4, 32 * 2)
        self.batch_norm4 = nn.BatchNorm1d(32 * 2)

        self.linear5 = nn.Linear(32 * 2, 10)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        self.linear1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        self.linear2.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.linear3.weight)
        self.linear3.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.linear4.weight)
        self.linear4.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.linear5.weight)
        self.linear5.bias.data.fill_(0)


    def get_optimizers(self):
        params1 = self.linear1.parameters()
        params2 = self.linear2.parameters()
        params3 = self.linear3.parameters()
        params4 = self.linear4.parameters()
        params5 = self.linear5.parameters()
        linear_optimizer = AdamW(params=chain(params1, params2, params3, params3, params4, params5), lr=mnist_setting.lr)
        return [linear_optimizer]
    
    def forward(self, inp: torch.Tensor):
        """
        
        :param inp: (bsz, input_dim)
        :return: 
        """
        out = F.dropout(inp, p=0.1)
        out = self.linear1(out)
        out = self.batch_norm1(out)
        out = F.gelu(out)

        out = F.dropout(out, p=0.1)
        out = self.linear2(out)
        out = self.batch_norm2(out)
        out = F.gelu(out)

        out = F.dropout(out, p=0.1)
        out = self.linear3(out)
        out = self.batch_norm3(out)
        out = F.gelu(out)

        out = F.dropout(out, p=0.1)
        out = self.linear4(out)
        out = self.batch_norm4(out)
        out = F.gelu(out)

        out = F.dropout(out, p=0.1)
        out = self.linear5(out)  # (bsz, label_cnt)

        out = F.softmax(out, dim=-1)  # (bsz, label_cnt)
        return {
            "mnist_result": out  # (bsz, label_cnt)
        }


class MNIST_Loss(nn.Module):
    def forward(self, mnist_result: torch.Tensor, mnist_label: torch.Tensor):
        """
        
        :param mnist_result: (bsz, label_cnt)
        :param mnist_label: (bsz)
        :return: 
        """
        loss = F.cross_entropy(mnist_result, mnist_label)
        return loss


class MNIST_Evaluator(BaseEvaluator):
    def __init__(self):
        super(MNIST_Evaluator, self).__init__()
        self.results = []
        self.gts = []
        self.kappa_evaluator = KappaEvaluator()
        self.precision_evaluator = PrecisionEvaluator()

    def eval_single(self, mnist_result: torch.Tensor, mnist_gt: torch.Tensor):
        """

        :param mnist_result: (1, label_cnt)
        :param mnist_gt: (1)
        :return:
        """
        mnist_result = mnist_result.squeeze()  # (label_cnt)
        _, max_idx = torch.max(mnist_result, 0)  # (1)
        max_idx = float(max_idx.clone().detach())
        self.kappa_evaluator.eval_single(max_idx, mnist_gt)
        self.precision_evaluator.eval_single(max_idx, mnist_gt)

    def eval_step(self) -> Dict[str, Any]:
        kappa = self.kappa_evaluator.eval_step()
        prec = self.precision_evaluator.eval_step()
        kappa.update(prec)
        return kappa


def train_dataset_factory(samples: List[List[float]], labels: List[int], bsz = mnist_setting.bsz, shuffle=True):
    data_dicts = list({'sample': np.array(x), 'label': y} for (x, y) in zip(samples, labels))
    train_dataset = SimpleDataset(data_dicts)

    def collate_fn(lst):
        dict_of_data = tools.transpose_list_of_dict(lst)
        sample = torch.tensor(np.stack(dict_of_data['sample']), dtype=torch.float)
        label = torch.tensor(np.stack(list(np.array(x) for x in dict_of_data['label'])), dtype=torch.long)
        return {
            "inp": sample,  # (bsz, input_dim),
        }, {
            "mnist_label": label  # (bsz)
        }
    train_dataloader = DataLoader(train_dataset, batch_size=bsz, shuffle=shuffle, collate_fn=collate_fn)
    return train_dataloader


def val_dataset_factory(samples: List[List[float]], labels: List[int]):
    data_dicts = list({'sample': np.array(x), 'label': y} for (x, y) in zip(samples, labels))
    val_dataset = SimpleDataset(data_dicts)

    def collate_fn(lst):
        dict_of_data = tools.transpose_list_of_dict(lst)
        sample = torch.tensor(np.stack(dict_of_data['sample']), dtype=torch.float)

        return {
            'inp': sample,  # (1, input_dim)
        }, {
            'mnist_gt': dict_of_data['label'][0]
        }

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return val_dataloader


def dataset_factory(file_dir: str, bsz: int = mnist_setting.bsz, shuffle=True):
    label_path, sample_path = 'TrainLabels.csv', 'TrainSamples.csv'
    if file_dir[-1] != '/':
        file_dir += '/'
    label_path = file_dir + label_path
    sample_path = file_dir + sample_path

    labels = list(int(x) for x in open(label_path, 'r').read().strip().split('\n'))  # List[int]
    samples = open(sample_path, 'r').read().strip().split('\n')
    samples = list(list(float(x) for x in a.split(',')) for a in samples)  # List[List[float]]

    train_samples, val_samples = tools.split_list_with_ratio(samples, mnist_setting.train_val_split)
    train_labels, val_labels = tools.split_list_with_ratio(labels, mnist_setting.train_val_split)

    train_dataloader = train_dataset_factory(train_samples, train_labels, bsz=bsz, shuffle=shuffle)
    val_dataloader = val_dataset_factory(val_samples, val_labels)

    return train_dataloader, val_dataloader


model_registry = {
    "model": Naive_Linear,
    "loss": MNIST_Loss,
    "evaluator": MNIST_Evaluator,
    "dataset": dataset_factory,
    'args': [
        {'name': "--file_dir", 'dest': 'file_dir', 'type': str, 'help': '训练/测试数据文件的路径'}
    ]
}
