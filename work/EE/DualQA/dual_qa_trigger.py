import json
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import BertModel
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from itertools import chain
from loguru import logger

from utils import tools, tokenize_tools, batch_tool
from utils.data import SimpleDataset
from work.EE.EE_utils import *
from evaluate.evaluator import BaseEvaluator, KappaEvaluator, PrecisionEvaluator
from dataset.ee_dataset import load_FewFC_ee, load_Duee_ee_formated
from analysis.recorder import NaiveRecorder
from work.EE.DualQA import dualqa_utils, dualqa_settings
from work.EE.DualQA.dual_qa import SharedEncoder, SimilarityModel, FlowAttention, SharedProjection
from work.EE.DualQA.dualqa_utils import concat_token_for_evaluate
from models.model_utils import get_init_params


class TriggerClassifier(nn.Module):
    """
    判断Trigger是哪一个词
    """
    def __init__(self, hidden: int):
        super(TriggerClassifier, self).__init__()

        self.start_classifier = nn.Linear(hidden, 1, bias=False)
        self.end_classifier = nn.Linear(hidden, 1, bias=False)
        self.softmax1 = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.start_classifier.weight)
        torch.nn.init.xavier_uniform_(self.end_classifier.weight)

    def forward(self, G: torch.Tensor, context_mask: torch.Tensor):
        """

        :param G: (bsz, C, hidden) 融合了其他信息的句子表示
        :param context_mask: (bsz, C) bool Tensor
        :return:
        """
        start_digit = self.start_classifier(G)  # (bsz, C, 1)
        end_digit = self.end_classifier(G)  # (bsz, C, 1)

        start_digit = F.leaky_relu(start_digit)  # (bsz, C, 1)
        end_digit = F.leaky_relu(end_digit)  # (bsz, C, 1)

        start_digit = start_digit.squeeze(dim=-1)  # (bsz, C)
        end_digit = end_digit.squeeze(dim=-1)  # (bsz, C)
        
        start_prob = self.softmax1(start_digit)  # (bsz, C)
        end_prob = self.softmax1(end_digit)  # (bsz, C)
        
        start_prob = start_prob.masked_fill(mask=context_mask, value=torch.tensor(1e-10))
        end_prob = end_prob.masked_fill(mask=context_mask, value=torch.tensor(1e-10))
        
        if len(start_prob.shape) == 1:
            start_prob = start_prob.unsqueeze(dim=0)
            end_prob = end_prob.unsqueeze(dim=0)
        return start_prob, end_prob  # both (bsz, C)


class TriggerWordClassifier(nn.Module):
    """
    判断一个词是不是Trigger
    0代表否
    1代表是
    """
    def __init__(self, hidden: int, max_seq_len: int = 256):
        super(TriggerWordClassifier, self).__init__()

        self.max_seq_len = max_seq_len
        self.hidden = hidden

        self.conv3 = nn.Conv2d(1, 32, (3, hidden))
        self.conv4 = nn.Conv2d(1, 32, (4, hidden))
        self.conv5 = nn.Conv2d(1, 32, (5, hidden))
        self.conv6 = nn.Conv2d(1, 32, (6, hidden))

        self.classifier = nn.Linear(128, 2)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0)

    def forward(self, G: torch.Tensor):
        G = G.unsqueeze(dim=1)  # (bsz, 1, C, hidden)
        G_conv3 = self.conv3(G).squeeze(dim=-1)  # (bsz, 32, C - 2)
        G_conv4 = self.conv4(G).squeeze(dim=-1)  # (bsz, 32, C - 3)
        G_conv5 = self.conv5(G).squeeze(dim=-1)  # (bsz, 32, C - 4)
        G_conv6 = self.conv6(G).squeeze(dim=-1)  # (bsz, 32, C - 5)

        G_max3, _ = torch.max(G_conv3, dim=-1)  # (bsz, 32)
        G_max4, _ = torch.max(G_conv4, dim=-1)  # (bsz, 32)
        G_max5, _ = torch.max(G_conv5, dim=-1)  # (bsz, 32)
        G_max6, _ = torch.max(G_conv6, dim=-1)  # (bsz, 32)

        G_max = torch.cat([G_max3, G_max4, G_max5, G_max6], dim=-1)  # (bsz, 128)

        G_digits = self.classifier(G_max)  # (bsz, 2)

        G_probs = F.softmax(G_digits, dim=-1)  # (bsz, 2)

        return G_probs


class DualQA_Trigger(nn.Module):
    def __init__(self, plm_path: str = dualqa_settings.plm_path, plm_lr: float = dualqa_settings.plm_lr, linear_lr: float = dualqa_settings.linear_lr):
        super(DualQA_Trigger, self).__init__()
        self.init_params = get_init_params(locals())

        self.plm_path = plm_path
        self.shared_encoder = SharedEncoder(plm_path)
        self.hidden = self.shared_encoder.bert.config.hidden_size

        self.flow_attention = FlowAttention(self.hidden)
        self.shared_projection = SharedProjection(self.hidden)
        self.trigger_classifier = TriggerClassifier(self.hidden)
        self.trigger_word_classifier = TriggerWordClassifier(self.hidden)

        self.linear_lr = linear_lr
        self.plm_lr = plm_lr

    def forward(self,
                context_input_ids: torch.Tensor,
                context_token_type_ids: torch.Tensor,
                context_attention_mask: torch.Tensor,
                T_input_ids: torch.Tensor = None,
                T_token_type_ids: torch.Tensor = None,
                T_attention_mask: torch.Tensor = None,
                TWord_input_ids: torch.Tensor = None,
                TWord_token_type_ids: torch.Tensor = None,
                TWord_attention_mask: torch.Tensor = None):
        """
        C - 上下文（context）
        T - 用于判断句子中哪个词是Trigger的询问句子
            这个句子中哪个词是触发词？（补充语）
        TWord - 用于判断句子中一个词是不是Trigger的询问句子
            这个句子中[]这个词是不是触发词？
        分别是C，QT， QTWord的input_ids, token_type_ids, attention_mask

        与原DualQA相同。
        训练时，同时使用T和TWord
        预测时，会根据None判断使用哪一个：如果Tword为None，则预测trigger是哪个词。如果T为None，则预测一个词是否为Trigger
        T和TWord不能同时为None

        :param context_input_ids: (bsz, C)
        :param context_token_type_ids:
        :param context_attention_mask:
        :param T_input_ids: (bsz, T) 与T_token_type_ids和T_attention_mask不同时为None
        :param T_token_type_ids:
        :param T_attention_mask:
        :param TWord_input_ids: (bsz, TW) 与TWord_token_type_ids和TWord_attention_mask不同时为None
        :param TWord_token_type_ids:
        :param Tword_attention_mask:
        :return:
        """
        if T_input_ids is None and TWord_input_ids is None:
            raise Exception('[DualQA_Trigger]T与TWord输入不能同时为None')

        start_probs, end_probs, trigger_word_pred = None, None, None

        H = self.shared_encoder(
            input_ids=context_input_ids,
            token_type_ids=context_token_type_ids,
            attention_mask=context_attention_mask)

        if T_input_ids is not None:
            UT = self.shared_encoder(
                input_ids=T_input_ids,
                token_type_ids=T_token_type_ids,
                attention_mask=T_attention_mask)
            # (bsz, QT, hidden)
            H_T_hat, UT_hat = self.flow_attention(H, UT, context_attention_mask, T_attention_mask)  # both (bsz, C, hidden)
            GT = self.shared_projection(H, H_T_hat, UT_hat)  # (bsz, C, hidden)
            probs_mask = (1 - context_attention_mask).bool()
            start_probs, end_probs = self.trigger_classifier(GT, probs_mask)  # both (bsz, C)

        if TWord_input_ids is not None:
            UTWord = self.shared_encoder(
                input_ids=TWord_input_ids,
                token_type_ids=TWord_token_type_ids,
                attention_mask=TWord_attention_mask)
            # (QTWord, hidden)
            H_TWord_hat, UTWord_hat = self.flow_attention(H, UTWord, context_attention_mask, TWord_attention_mask)  # both (bsz, C, hidden)
            GTWord = self.shared_projection(H, H_TWord_hat, UTWord_hat)  # (bsz, C, hidden)
            trigger_word_pred = self.trigger_word_classifier(GTWord)  # (bsz, 2)

        return {
            'start_probs': start_probs,
            'end_probs': end_probs,
            'trigger_word_pred': trigger_word_pred
        }

    def get_optimizers(self):
        # plm param
        plm_params = self.shared_encoder.parameters()

        # linear param
        flow_params = self.flow_attention.parameters()
        projection_params = self.shared_projection.parameters()

        T_cls_params = self.trigger_classifier.parameters()
        TWord_cls_params = self.trigger_word_classifier.parameters()

        # optimizers
        plm_optimizer = AdamW(params=plm_params, lr=self.plm_lr)
        linear_optimizer = AdamW(params=chain(flow_params, projection_params, T_cls_params, TWord_cls_params))

        return [plm_optimizer, linear_optimizer]


class DualQA_Trigger_Loss(nn.Module):
    def forward(self,
                start_probs: torch.Tensor,
                end_probs: torch.Tensor,
                trigger_word_pred: torch.Tensor,
                start_T_label: torch.Tensor,
                end_T_label: torch.Tensor,
                TWord_label: torch.Tensor):
        """

        :param start_probs: (bsz, C)
        :param end_probs: (bsz, C)
        :param trigger_word_pred: (bsz, 2)
        :param start_T_label: (bsz, C)
        :param end_T_label: (bsz, C)
        :param TWord_label: (bsz)
        :return:
        """
        T_start_loss = F.binary_cross_entropy(start_probs, start_T_label)
        T_end_loss = F.binary_cross_entropy(end_probs, end_T_label)
        T_loss = T_start_loss + T_end_loss

        TWord_loss = F.cross_entropy(trigger_word_pred, TWord_label)

        loss = T_loss + TWord_loss

        return loss



class DualQA_Trigger_Evaluator(BaseEvaluator):
    def __init__(self):
        super(DualQA_Trigger_Evaluator, self).__init__()
        self.trigger_precision_evaluator = PrecisionEvaluator()
        self.trigger_word_precision_evaluator = PrecisionEvaluator()

        self.pred_lst = []
        self.gt_lst = []

    def eval_single(self,
                    start_probs: torch.Tensor,
                    end_probs: torch.Tensor,
                    trigger_word_pred: torch.Tensor,
                    T_gt: str,
                    TWord_label: int,
                    tokens: List[str],
                    threshold: int = 0.5):
        """

        :param start_probs: (bsz, C)
        :param end_probs: (bsz, C)
        :param trigger_word_pred:
        :param T_gt:
        :param TWord_label:
        :return:
        """
        start_probs = start_probs.squeeze().clone().detach().cpu()
        start_digits = (start_probs > threshold).int().tolist()
        start_probs = np.array(start_probs)  # (C)
        end_probs = end_probs.squeeze().clone().detach().cpu()
        end_digits = (end_probs > threshold).int().tolist()
        end_probs = np.array(end_probs)  # (C)
        start_position = int(np.argsort(start_probs)[-1])
        end_position = int(np.argsort(end_probs)[-1])
        span = (start_position, end_position)
        trigger_pred = concat_token_for_evaluate(tokens, span)

        word_pred = int(np.argsort(trigger_word_pred.squeeze().clone().detach().cpu())[-1])

        self.trigger_precision_evaluator.eval_single(trigger_pred, T_gt)
        self.trigger_word_precision_evaluator.eval_single(word_pred, TWord_label)

        self.pred_lst.append({
            'trigger_pred': trigger_pred,
            'trigger_word_pred': word_pred
        })
        self.gt_lst.append({
            'trigger_gt': T_gt,
            'trigger_word_gt': TWord_label
        })

    def eval_step(self) -> Dict[str, Any]:
        trigger_p = self.trigger_precision_evaluator.eval_step()
        trigger_word_p = self.trigger_word_precision_evaluator.eval_step()

        trigger_p = tools.modify_key_of_dict(trigger_p, lambda x: 'trigger_' + x)
        trigger_word_p = tools.modify_key_of_dict(trigger_word_p, lambda x: 'trigger_word_' + x)

        result = {}
        result.update(trigger_p)
        result.update(trigger_word_p)

        self.pred_lst = []
        self.gt_lst = []
        return result


def train_dataset_factory(data_dicts: List[dict], bsz: int = dualqa_settings.default_bsz, shuffle: bool = True):
    """

    :param data_dicts:
    :param bsz:
    :param shuffle:
    :return:
    """
    dataset = SimpleDataset(data_dicts)
    def collate_fn(lst):
        """

        :param lst:
        :return:
        """
        data_dict = tools.transpose_list_of_dict(lst)
        context_input_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['input_ids']), dtype=torch.long)
        context_token_type_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['token_type_ids']), dtype=torch.long)
        context_attention_mask = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['attention_mask']), dtype=torch.long)
        T_input_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['T_input_ids']), dtype=torch.long)
        T_token_type_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['T_token_type_ids']), dtype=torch.long)
        T_attention_mask = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['T_attention_mask']), dtype=torch.long)
        TWord_input_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['TWord_input_ids']), dtype=torch.long)
        TWord_token_type_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['TWord_token_type_ids']), dtype=torch.long)
        TWord_attention_mask = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['TWord_attention_mask']), dtype=torch.long)

        start_T_label = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['start_T_label']), dtype=torch.float)
        end_T_label = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['end_T_label']), dtype=torch.float)
        TWord_label = torch.tensor(data_dict['TWord_label'], dtype=torch.long)

        return {
           'context_input_ids': context_input_ids,
           'context_token_type_ids': context_token_type_ids,
           'context_attention_mask': context_attention_mask,
           'T_input_ids': T_input_ids,
           'T_token_type_ids': T_token_type_ids,
           'T_attention_mask': T_attention_mask,
           'TWord_input_ids': TWord_input_ids,
           'TWord_token_type_ids': TWord_token_type_ids,
           'TWord_attention_mask': TWord_attention_mask
               }, {
            'start_T_label': start_T_label,
            'end_T_label': end_T_label,
            'TWord_label': TWord_label
        }


    dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def val_dataset_factory(data_dicts: List[dict]):
    """

    :param data_dicts:
    :return:
    """
    dataset = SimpleDataset(data_dicts)
    def collate_fn(lst):
        """

        :param lst:
        :return:
        """
        data_dict = tools.transpose_list_of_dict(lst)
        context_input_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['input_ids']), dtype=torch.long)
        context_token_type_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['token_type_ids']), dtype=torch.long)
        context_attention_mask = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['attention_mask']), dtype=torch.long)
        T_input_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['T_input_ids']), dtype=torch.long)
        T_token_type_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['T_token_type_ids']), dtype=torch.long)
        T_attention_mask = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['T_attention_mask']), dtype=torch.long)
        TWord_input_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['TWord_input_ids']), dtype=torch.long)
        TWord_token_type_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['TWord_token_type_ids']), dtype=torch.long)
        TWord_attention_mask = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['TWord_attention_mask']), dtype=torch.long)

        return {
           'context_input_ids': context_input_ids,
           'context_token_type_ids': context_token_type_ids,
           'context_attention_mask': context_attention_mask,
           'T_input_ids': T_input_ids,
           'T_token_type_ids': T_token_type_ids,
           'T_attention_mask': T_attention_mask,
           'TWord_input_ids': TWord_input_ids,
           'TWord_token_type_ids': TWord_token_type_ids,
           'TWord_attention_mask': TWord_attention_mask
               }, {
            'T_gt': data_dict['T_gt'][0],
            'TWord_label': data_dict['TWord_label'][0],
            'tokens': data_dict['token'][0]
        }


    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return dataloader


def dataset_factory(dataset_type: str, train_file: str, valid_file: str, bsz: int):
    bsz = int(bsz)
    train_data_dicts = pickle.load(open(train_file, 'rb'))
    valid_data_dicts = pickle.load(open(valid_file, 'rb'))
    # valid_data_dicts = list(json.loads(x) for x in open(valid_file, 'r', encoding='utf-8').read().strip().split('\n'))

    train_dataloader = train_dataset_factory(train_data_dicts, bsz=bsz)
    val_dataloader = val_dataset_factory(valid_data_dicts)
    return train_dataloader, val_dataloader


model_registry = {
    'model': DualQA_Trigger,
    'loss': DualQA_Trigger_Loss,
    'evaluator': DualQA_Trigger_Evaluator,
    'train_val_data': dataset_factory,
    'recorder': NaiveRecorder
}


if __name__ == '__main__':
    train_loader, val_loader = dataset_factory('FewFC', 'temp_data/train.DualQA_Trigger.FewFC.labeled.pk', 'temp_data/valid.DualQA_Trigger.FewFC.tokenized.pk', bsz=1)
    train_samples, valid_samples = [], []
    for elem in train_loader:
        train_samples.append(elem)
    for elem in val_loader:
        valid_samples.append(elem)