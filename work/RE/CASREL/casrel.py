import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from itertools import chain
from torch.optim import AdamW
import numpy as np

from transformers import BertModel, BertTokenizerFast


from type_def import *
from evaluate.evaluator import BaseEvaluator, F1_Evaluator
from work.RE import RE_settings, RE_utils
from work.RE.RE_utils import Triplet, convert_lists_to_triplet_casrel
from utils import tools, tokenize_tools, batch_tool
from analysis.recorder import NaiveRecorder
from utils.data import SimpleDataset
from dataset import re_dataset


class CASREL(nn.Module):
    def __init__(self,
                 relation_cnt: int = len(RE_settings.relations[RE_settings.default_datatype]),
                 bert_plm_path: str = RE_settings.default_plm[RE_settings.default_datatype],
                 plm_lr: float = RE_settings.plm_lr,
                 others_lr: float = RE_settings.others_lr
                 ):
        super(CASREL, self).__init__()

        # 保存初始化参数
        self.relation_cnt = relation_cnt
        self.plm_path = bert_plm_path
        self.plm_lr = plm_lr
        self.others_lr = others_lr

        # 记载bert预训练模型
        self.bert = BertModel.from_pretrained(bert_plm_path)
        self.hidden = self.bert.config.hidden_size

        # 初始化分类器
        self.subject_start_cls = nn.Linear(self.hidden, 1)
        self.subject_end_cls = nn.Linear(self.hidden, 1)
        self.object_start_cls = nn.Linear(self.hidden, self.relation_cnt)
        self.object_end_cls = nn.Linear(self.hidden, self.relation_cnt)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.subject_start_cls.weight)
        self.subject_start_cls.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.subject_end_cls.weight)
        self.subject_end_cls.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.object_start_cls.weight)
        self.object_start_cls.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.object_end_cls.weight)
        self.object_end_cls.bias.data.fill_(0)

    def forward(self,
                input_ids: torch.Tensor,
                token_type_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                subject_gt_start: torch.Tensor = None,
                subject_gt_end: torch.Tensor = None):
        """
        在train模式下，
            - subject_gt_start/end均不为None，将用于模型第二步的训练
            - 返回值
                - subject_start_result
                - subject_end_result
                - object_start_result
                - object_end_result
        在eval模式下，subject_gt_start/end为None（就算不为None，也不会用到）
        :param input_ids: (bsz, seq_l)
        :param token_type_ids: (bsz, seq_l)
        :param attention_mask: (bsz, seq_l)
        :param subject_gt_start: (bsz, seq_l) 只包含1和0的向量。
        :param subject_gt_end: (bsz, seq_l)
        :return:
        """
        def calculate_embed_with_subject(bert_embed, subject_start, subject_end):
            """
            计算subject的表示向量subject_repr
            :param bert_embed: (bsz, seq_l, hidden)
            :param subject_start: (bsz, 1, seq_l)
            :param subject_end: (bsz, 1, seq_l)
            :return:
            """
            start_repr = torch.bmm(subject_start, bert_embed)  # (bsz, 1, hidden)
            end_repr = torch.bmm(subject_end, bert_embed)  # (bsz, 1, hidden)
            subject_repr = (start_repr + end_repr) / 2  # (bsz, 1, hidden)

            # h_N + v^k_{sub}
            embed = bert_embed + subject_repr
            return embed

        # 获取BERT embedding
        result = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        output, _ = result[0], result[1]  # output: (bsz, seq_l, hidden)

        mask = (1 - attention_mask).bool()  # (bsz, seq_l)
        # 获取subject的预测
        subject_start_result = self.subject_start_cls(output)  # (bsz, seq_l, 1)
        subject_end_result = self.subject_end_cls(output)  # (bsz, seq_l, 1)
        subject_start_result = torch.sigmoid(subject_start_result)  # (bsz, seq_l, 1)
        subject_end_result = torch.sigmoid(subject_end_result)  # (bsz, seq_l, 1)
        subject_start_result = subject_start_result.masked_fill(mask=mask.unsqueeze(dim=-1), value=0)  # (bsz, seq_l, 1)
        subject_end_result = subject_end_result.masked_fill(mask=mask.unsqueeze(dim=-1), value=0)  # (bsz, seq_l, 1)
        if self.training:
            gt_start, gt_end = subject_gt_start.unsqueeze(dim=1), subject_gt_end.unsqueeze(dim=1)
            # both (bsz, 1, seq_l)
            embed = calculate_embed_with_subject(output, gt_start, gt_end)  # (bsz, seq_l, hidden)

            # 计算object的预测
            object_start_result = self.object_start_cls(embed)  # (bsz, seq_l, relation_cnt)
            object_end_result = self.object_end_cls(embed)  # (bsz, seq_l, relation_cnt)

            object_start_result = torch.sigmoid(object_start_result)  # (bsz, seq_l, relation_cnt)
            object_end_result = torch.sigmoid(object_end_result)  # (bsz, seq_l, relation_cnt)
            relation_cnt = object_end_result.shape[-1]
            ro_mask = mask.unsqueeze(dim=-1)  # (bsz, seq_l, 1)
            ro_mask = ro_mask.repeat(1, 1, relation_cnt)  # (bsz, seq_l, relation_cnt)
            ro_attention_mask = (1 - ro_mask.float())  # (bsz, seq_l, relation_cnt)
            object_start_result = object_start_result.masked_fill(mask=ro_mask, value=0)  # (bsz, seq_l, relation_cnt)
            object_end_result = object_end_result.masked_fill(mask=ro_mask, value=0)  # (bsz, seq_l, relation_cnt)
            return {
                "subject_start_result": subject_start_result,  # (bsz, seq_l, 1)
                "subject_end_result": subject_end_result,  # (bsz, seq_l, 1)
                "object_start_result": object_start_result,  # (bsz, seq_l, relation_cnt)
                "object_end_result": object_end_result,  # （bsz, seq_l, relation_cnt)
                'subject_mask': attention_mask,  # (bsz, seq_l)
                'ro_mask': ro_attention_mask,  # (bsz, seq_l, relation_cnt)
            }
        else:  # eval模式。该模式下，bsz默认为1
            # 获取subject的预测：cur_spans - SpanList
            subject_start_result = subject_start_result.squeeze()  # (seq_l)
            subject_end_result = subject_end_result.squeeze()  # (seq_l)
            subject_start_int = (subject_start_result > 0.5).int().tolist()
            subject_end_int = (subject_end_result > 0.5).int().tolist()
            seq_l = len(subject_start_int)
            subject_spans = tools.argument_span_determination(subject_start_int, subject_end_int, subject_start_result, subject_end_result)  # SpanList

            # 迭代获取每个subject所对应的object预测
            object_spans = []  #  List[List[SpanList]]  (subject, relation, span number)
            for elem_span in subject_spans:
                object_spans_for_current_subject = []  # List[SpanList]  (relation, span number)
                temporary_start, temporary_end = torch.zeros(1, 1, seq_l).cuda(), torch.zeros(1, 1, seq_l).cuda()  # both (1, 1, seq_l)
                temporary_start[0][0][elem_span[0]] = 1
                temporary_end[0][0][elem_span[1]] = 1
                embed = calculate_embed_with_subject(output, temporary_start, temporary_end)  # (1, seq_l, hidden)

                # 计算object的预测
                object_start_result = self.object_start_cls(embed)  # (1, seq_l, relation_cnt)
                object_end_result = self.object_end_cls(embed)  # (1, seq_l, relation_cnt)

                object_start_result = object_start_result.squeeze().T  # (relation_cnt, seq_l)
                object_end_result = object_end_result.squeeze().T  # (relation_cnt, seq_l)

                object_start_result = torch.sigmoid(object_start_result)
                object_end_result = torch.sigmoid(object_end_result)
                for (object_start_r, object_end_r) in zip(object_start_result, object_end_result):
                    o_start_int = (object_start_r > 0.5).int().tolist()
                    o_end_int = (object_end_r > 0.5).int().tolist()
                    cur_spans = tools.argument_span_determination(o_start_int, o_end_int, object_start_r, object_end_r)  # SpanList
                    object_spans_for_current_subject.append(cur_spans)
                object_spans.append(object_spans_for_current_subject)

            return {
                "pred_subjects": subject_spans,  # SpanList
                "pred_objects": object_spans  # List[List[SpanList]]
            }


    def get_optimizers(self):
        plm_params = self.bert.parameters()
        ss_params = self.subject_start_cls.parameters()
        se_params = self.subject_end_cls.parameters()
        os_params = self.object_start_cls.parameters()
        oe_params = self.object_end_cls.parameters()
        plm_optimizer = AdamW(params=plm_params, lr=self.plm_lr)
        linear_optimizer = AdamW(params=chain(ss_params, se_params, os_params, oe_params), lr=self.others_lr)
        return [plm_optimizer, linear_optimizer]


class CASREL_subject_part(nn.Module):
    def __init__(self,
                 relation_cnt: int = len(RE_settings.relations[RE_settings.default_datatype]),
                 bert_plm_path: str = RE_settings.default_plm[RE_settings.default_datatype],
                 plm_lr: float = RE_settings.plm_lr,
                 others_lr: float = RE_settings.others_lr
                 ):
        super(CASREL_subject_part, self).__init__()

        # 保存初始化参数
        self.relation_cnt = relation_cnt
        self.plm_path = bert_plm_path
        self.plm_lr = plm_lr
        self.others_lr = others_lr

        # 记载bert预训练模型
        self.bert = BertModel.from_pretrained(bert_plm_path)
        self.hidden = self.bert.config.hidden_size

        # 初始化分类器
        self.subject_start_cls = nn.Linear(self.hidden, 1)
        self.subject_end_cls = nn.Linear(self.hidden, 1)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.subject_start_cls.weight)
        self.subject_start_cls.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.subject_end_cls.weight)
        self.subject_end_cls.bias.data.fill_(0)

    def forward(self,
                input_ids: torch.Tensor,
                token_type_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                subject_gt_start: torch.Tensor = None,
                subject_gt_end: torch.Tensor = None):
        """
        在train模式下，
            - subject_gt_start/end均不为None，将用于模型第二步的训练
            - 返回值
                - subject_start_result
                - subject_end_result
                - object_start_result
                - object_end_result
        在eval模式下，subject_gt_start/end为None（就算不为None，也不会用到）
        :param input_ids: (bsz, seq_l)
        :param token_type_ids: (bsz, seq_l)
        :param attention_mask: (bsz, seq_l)
        :param subject_gt_start: (bsz, seq_l) 只包含1和0的向量。
        :param subject_gt_end: (bsz, seq_l)
        :return:
        """
        def calculate_embed_with_subject(bert_embed, subject_start, subject_end):
            """
            计算subject的表示向量subject_repr
            :param bert_embed: (bsz, seq_l, hidden)
            :param subject_start: (bsz, 1, seq_l)
            :param subject_end: (bsz, 1, seq_l)
            :return:
            """
            start_repr = torch.matmul(subject_start, bert_embed)  # (bsz, 1, hidden)
            end_repr = torch.matmul(subject_end, bert_embed)  # (bsz, 1, hidden)
            subject_repr = (start_repr + end_repr) / 2  # (bsz, 1, hidden)

            # h_N + v^k_{sub}
            embed = bert_embed + subject_repr
            return embed

        # 获取BERT embedding
        result = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        output, _ = result[0], result[1]  # output: (bsz, seq_l, hidden)

        mask = (1 - attention_mask).bool()
        # 获取subject的预测
        subject_start_result = self.subject_start_cls(output)  # (bsz, seq_l, 1)
        subject_end_result = self.subject_end_cls(output)  # (bsz, seq_l, 1)
        subject_start_result = F.sigmoid(subject_start_result)  # (bsz, seq_l, 1)
        subject_end_result = F.sigmoid(subject_end_result)  # (bsz, seq_l, 1)
        subject_start_result = subject_start_result.masked_fill(mask=mask.unsqueeze(dim=-1), value=0)
        subject_end_result = subject_end_result.masked_fill(mask=mask.unsqueeze(dim=-1), value=0)
        return {
            "subject_start_result": subject_start_result,
            'subject_end_result': subject_end_result
        }


    def get_optimizers(self):
        plm_params = self.bert.parameters()
        ss_params = self.subject_start_cls.parameters()
        se_params = self.subject_end_cls.parameters()
        plm_optimizer = AdamW(params=plm_params, lr=self.plm_lr)
        linear_optimizer = AdamW(params=chain(ss_params, se_params), lr=self.others_lr)
        return [plm_optimizer, linear_optimizer]


class CASREL_Loss(nn.Module):
    def __init__(self, lamb: float = 0.6):
        super(CASREL_Loss, self).__init__()
        self.lamb = lamb
        # self.focal_weight = tools.FocalWeight()

    def forward(self,
                subject_start_result: torch.Tensor,
                subject_end_result: torch.Tensor,
                object_start_result: torch.Tensor,
                object_end_result: torch.Tensor,
                subject_mask: torch.Tensor,
                ro_mask: torch.Tensor,
                subject_start_label: torch.Tensor,
                subject_end_label: torch.Tensor,
                object_start_label: torch.Tensor,
                object_end_label: torch.Tensor):
        """

        :param subject_start_result: (bsz, seq_l, 1)
        :param subject_end_result: (bsz, seq_l, 1)
        :param object_start_result: (bsz, seq_l, relation_cnt)
        :param object_end_result: (bsz, seq_l, relation_cnt)
        :param subject_mask: (bsz, seq_l) 用于处理subject的mask
        :param ro_mask: (bsz, seq_l) 用于处理relation-object的mask
        :param subject_start_label: (bsz, seq_l)
        :param subject_end_label: (bsz, seq_l)
        :param object_start_label: (bsz, seq_l, relation_cnt)
        :param object_end_label: (bsz, seq_l, relation_cnt)
        :return:
        """
        # 将subject_result的形状与label对齐
        subject_start_result = subject_start_result.squeeze(-1)  # (bsz, seq_l)
        subject_end_result = subject_end_result.squeeze(-1)  # (bsz, seq_l)

        # 计算weight
        # subject_start_focal_weight = self.focal_weight(subject_start_label, subject_start_result)
        # subject_end_focal_weight = self.focal_weight(subject_end_label, subject_end_result)
        # object_start_focal_weight = self.focal_weight(object_start_label, object_start_result)
        # object_end_focal_weight = self.focal_weight(object_end_label, object_end_result)

        # 计算loss
        subject_start_loss = F.binary_cross_entropy(subject_start_result, subject_start_label, reduction='none')
        subject_end_loss = F.binary_cross_entropy(subject_end_result, subject_end_label, reduction='none')
        object_start_loss = F.binary_cross_entropy(object_start_result, object_start_label, reduction='none')
        object_end_loss = F.binary_cross_entropy(object_end_result, object_end_label, reduction='none')

        ss_loss = torch.sum(subject_start_loss * subject_mask) / torch.sum(subject_mask)
        se_loss = torch.sum(subject_end_loss * subject_mask) / torch.sum(subject_mask)
        os_loss = torch.sum(object_start_loss * ro_mask) / torch.sum(ro_mask)
        oe_loss = torch.sum(object_end_loss * ro_mask) / torch.sum(ro_mask)

        loss = self.lamb * (ss_loss + se_loss) + (1 - self.lamb) * (os_loss + oe_loss)
        return loss


class CASREL_subject_part_Loss(nn.Module):
    def forward(self,
                subject_start_result: torch.Tensor,
                subject_end_result: torch.Tensor,
                subject_start_label: torch.Tensor,
                subject_end_label: torch.Tensor,
                object_start_label, object_end_label):
        """

        :param subject_start_result: (bsz, seq_l, 1)
        :param subject_end_result: (bsz, seq_l, 1)
        :param subject_start_label: (bsz, seq_l)
        :param subject_end_label: (bsz, seq_l)
        """
        # 将subject_result的形状与label对齐
        subject_start_result = subject_start_result.squeeze()  # (bsz, seq_l)
        subject_end_result = subject_end_result.squeeze()  # (bsz, seq_l)

        # 计算loss
        subject_start_loss = F.binary_cross_entropy(subject_start_result, subject_start_label,
                                                    1 + subject_start_label * 3)
        subject_end_loss = F.binary_cross_entropy(subject_end_result, subject_end_label, 1 + subject_end_label * 3)

        loss = subject_start_loss + subject_end_loss

        return loss


class CASREL_Evaluator(BaseEvaluator):
    def __init__(self):
        super(CASREL_Evaluator, self).__init__()
        self.total_f1_evaluator = F1_Evaluator()
        self.subject_f1_evaluator = F1_Evaluator()
        self.pred_lst, self.gt_lst = [], []

    def eval_single(self,
                pred_subjects: SpanList,
                pred_objects: List[List[SpanList]],
                # subjects_gt: SpanList,
                # objects_gt: List[List[SpanList]],
                gt_triplets: List[Tuple],
                tokens: List[str]):
        """

        :param pred_subjects:
        :param pred_objects:
        :param subjects_gt:
        :param objects_gt:
        :return:
        """
        gt_triplets = list(tuple(x) for x in gt_triplets)
        pred_subjects = list(tuple(x) for x in pred_subjects)
        gt_subject_spans = list(set((x[1], x[2]) for x in gt_triplets))
        pred_triplets = convert_lists_to_triplet_casrel(pred_subjects, pred_objects)  # List[Triplet]
        # gt_triplets = convert_lists_to_triplet_casrel(subjects_gt, objects_gt)  # List[Triplet]
        self.pred_lst.append([pred_triplets, pred_subjects, pred_objects])
        self.gt_lst.append([gt_triplets, tokens])
        self.total_f1_evaluator.eval_single(pred_triplets, gt_triplets)
        self.subject_f1_evaluator.eval_single(pred_subjects, gt_subject_spans)

    def eval_step(self) -> Dict[str, Any]:
        f1_result = self.total_f1_evaluator.eval_step()
        subject_f1 = self.subject_f1_evaluator.eval_step()
        f1_result = tools.modify_key_of_dict(f1_result, lambda x: 'total_' + x)
        subject_f1 = tools.modify_key_of_dict(subject_f1, lambda x: 'subject_' + x)
        self.pred_lst = []
        self.gt_lst = []
        subject_f1.update(f1_result)
        return subject_f1


class CASREL_subject_part_Evaluator(BaseEvaluator):
    def __init__(self):
        super(CASREL_subject_part_Evaluator, self).__init__()
        self.f1_evaluator = F1_Evaluator()
        self.pred_lst, self.gt_lst = [], []

    def eval_single(self,
                    subject_start_result: torch.Tensor,
                    subject_end_result: torch.Tensor,
                    gt_triplets: List[Tuple],
                    tokens: List[str]):

        subject_start_result = subject_start_result.squeeze()
        subject_end_result = subject_end_result.squeeze()

        gt_subject_spans = list(set((x[1], x[2]) for x in gt_triplets))
        subject_start_bool = (subject_start_result > 0.5).int().tolist()
        subject_end_bool = (subject_end_result > 0.5).int().tolist()
        spans = tools.argument_span_determination(subject_start_bool, subject_end_bool, subject_start_result, subject_end_result)
        self.f1_evaluator.eval_single(spans, gt_subject_spans)
        self.pred_lst.append([subject_start_result.tolist(), subject_end_result.tolist(), spans])
        self.gt_lst.append(gt_subject_spans)

    def eval_step(self) -> Dict[str, Any]:
        f1_result = self.f1_evaluator.eval_step()
        self.pred_lst, self.gt_lst = [], []
        return f1_result


def train_dataset_factory(data_dicts: List[dict], bsz: int = RE_settings.default_bsz, shuffle: bool = RE_settings.default_shuffle):
    """
    data_dicts中的每一个dict包括
    - text
    - input_ids
    - token_type_ids
    - attention_mask
    - offset_mapping
    - token

    - subject
    - subject_start_label {seq_l:, label_indexes:}
    - subject_end_label (same)
    - subject_occur
    - subject_start_gt {seq_l: , label_index: }
    - subject_end_gt

    - relation_object
    - relation_to_object_start_label {seq_l: , relation_cnt: , label_per_relation: }
    - relation_to_object_end_label

    :param data_dicts:
    :param bsz:
    :param shuffle:
    :return:
    """
    train_dataset = SimpleDataset(data_dicts)

    def collate_fn(lst):
        """
        dict in lst contains:
        :param lst:
        :return:
        """
        data_dict = tools.transpose_list_of_dict(lst)
        bsz = len(lst)

        # generate basic input
        input_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['input_ids']), dtype=torch.long)
        token_type_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['token_type_ids']), dtype=torch.long)
        attention_mask = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['attention_mask']), dtype=torch.long)
        seq_l = input_ids.shape[1]
        # all (bsz, max_seq_l)

        # generate subject gt for phase 2
        start_gt_info, end_gt_info = tools.transpose_list_of_dict(data_dict['subject_start_gt']), tools.transpose_list_of_dict(data_dict['subject_end_gt'])
        start_indexes, end_indexes = start_gt_info['label_index'], end_gt_info['label_index']
        start_gt = torch.zeros((bsz, seq_l)).scatter(dim=1, index=torch.LongTensor(start_indexes).unsqueeze(-1), src=torch.ones(bsz, 1))
        end_gt = torch.zeros((bsz, seq_l)).scatter(dim=1, index=torch.LongTensor(end_indexes).unsqueeze(-1), src=torch.ones(bsz, 1))
        # both (bsz, gt_l)

        # generate subject label
        start_label_info, end_label_info = tools.transpose_list_of_dict(data_dict['subject_start_label']), tools.transpose_list_of_dict(data_dict['subject_end_label'])
        start_label_indexes, end_label_indexes = start_label_info['label_indexes'], end_label_info['label_indexes'] # list of list
        start_labels, end_labels = [], []
        for elem_start_indexes, elem_end_indexes in zip(start_label_indexes, end_label_indexes):
            start_label_cnt, end_label_cnt = len(elem_start_indexes), len(elem_end_indexes)
            start_labels.append(torch.zeros(seq_l).scatter(dim=0, index=torch.LongTensor(elem_start_indexes), src=torch.ones(start_label_cnt)))
            end_labels.append(torch.zeros(seq_l).scatter(dim=0, index=torch.LongTensor(elem_end_indexes), src=torch.ones(end_label_cnt)))
        start_label = torch.stack(start_labels)
        end_label = torch.stack(end_labels)
        # both (bsz, seq_l)

        # generate object-relation label
        ro_start_info, ro_end_info = tools.transpose_list_of_dict(data_dict['relation_to_object_start_label']), tools.transpose_list_of_dict(data_dict['relation_to_object_end_label'])
        relation_cnt = ro_start_info['relation_cnt'][0]
        start_label_pre_relation, end_label_per_relation = ro_start_info['label_per_relation'], ro_end_info['label_per_relation']
        ro_start_label, ro_end_label = torch.zeros((bsz, seq_l, relation_cnt)), torch.zeros((bsz, seq_l, relation_cnt))
        for i_batch in range(bsz):
            for i_rel in range(relation_cnt):
                ro_cur_start_label_indexes = start_label_pre_relation[i_batch][i_rel]
                ro_cur_end_label_indexes = end_label_per_relation[i_batch][i_rel]
                for elem in ro_cur_start_label_indexes:
                    ro_start_label[i_batch][elem][i_rel] = 1
                for elem in ro_cur_end_label_indexes:
                    ro_end_label[i_batch][elem][i_rel] = 1

        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'subject_gt_start': start_gt,
            'subject_gt_end': end_gt
               }, {
            'subject_start_label': start_label,
            'subject_end_label': end_label,
            'object_start_label': ro_start_label,
            'object_end_label': ro_end_label
        }

    train_dataloader = DataLoader(train_dataset, batch_size=bsz, shuffle=shuffle, collate_fn=collate_fn)

    return train_dataloader


def dev_dataset_factory(data_dicts: List[dict]):

    valid_dataset = SimpleDataset(data_dicts)

    def collate_fn(lst):
        data_dict = tools.transpose_list_of_dict(lst)

        # generate basic input
        input_ids = torch.tensor(data_dict['input_ids'][0], dtype=torch.long).unsqueeze(0)
        token_type_ids = torch.tensor(data_dict['token_type_ids'][0], dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor(data_dict['attention_mask'][0], dtype=torch.long).unsqueeze(0)
        # all (1, seq_l)

        gt_triplets = data_dict['eval_triplets'][0]
        tokens = data_dict['token'][0]

        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
               }, {
            'gt_triplets': gt_triplets,
            'tokens': tokens
        }

    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return valid_dataloader

def dataset_factory(train_file: str, valid_file: str, bsz: int = RE_settings.default_bsz, shuffle: bool = RE_settings.default_shuffle):
    train_data_dicts = pickle.load(open(train_file, 'rb'))
    valid_data_dicts = pickle.load(open(valid_file, 'rb'))

    train_dataloader = train_dataset_factory(train_data_dicts, bsz=bsz, shuffle=shuffle)
    valid_dataloader = dev_dataset_factory(valid_data_dicts)

    return train_dataloader, valid_dataloader

    # for elem in train_dicts:
    #     offset_mapping = elem['offset_mapping']
    #     for elem_d in elem['triplets']:
    #         RE_utils.add_token_span_to_re_data(elem_d, offset_mapping)
    # for elem in dev_dicts:
    #     offset_mapping = elem['offset_mapping']
    #     for elem_d in elem['triplets']:
    #         RE_utils.add_token_span_to_re_data(elem_d, offset_mapping)




# model_registry = {
#     "model": CASREL_subject_part,
#     "evaluator": CASREL_subject_part_Evaluator,
#     'loss': CASREL_subject_part_Loss,
#     'train_val_data': dataset_factory,
#     'recorder': NaiveRecorder
# }
model_registry = {
    "model": CASREL,
    "evaluator": CASREL_Evaluator,
    'loss': CASREL_Loss,
    'train_val_data': dataset_factory,
    'recorder': NaiveRecorder
}

if __name__ == '__main__':
    train_loader, val_loader = dataset_factory('temp_data/train.duie.final.pk', 'temp_data/valid.duie.eval_final.pk')
    train_data, val_data = [], []
    limit = 10
    for (elem1, elem2) in zip(train_loader, val_loader):
        train_data.append(elem1)
        val_data.append(elem2)
        limit -= 1
        if limit <= 0:
            break