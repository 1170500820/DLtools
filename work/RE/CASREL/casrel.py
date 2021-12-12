import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

from transformers import BertModel, BertTokenizerFast, AdamW


from type_def import *
from evaluate.evaluator import BaseEvaluator, F1_Evaluator
from work.RE import RE_settings, RE_utils
from work.RE.RE_utils import Triplet, convert_lists_to_triplet_casrel
from utils import tools, tokenize_tools
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

        # 获取subject的预测
        subject_start_result = self.subject_start_cls(output)  # (bsz, seq_l, 1)
        subject_end_result = self.subject_end_cls(output)  # (bsz, seq_l, 1)
        subject_start_result = F.sigmoid(subject_start_result)  # (bsz, seq_l, 1)
        subject_end_result = F.sigmoid(subject_end_result)  # (bsz, seq_l, 1)
        if self.training:
            gt_start, gt_end = subject_gt_start.unsqueeze(dim=1), subject_gt_end.unsqueeze(dim=1)
            # both (bsz, 1, seq_l)
            embed = calculate_embed_with_subject(output, gt_start, gt_end)  # (bsz, seq_l, hidden)

            # 计算object的预测
            object_start_result = self.object_start_cls(embed)  # (bsz, seq_l, relation_cnt)
            object_end_result = self.object_end_cls(embed)  # (bsz, seq_l, relation_cnt)
            return {
                "subject_start_result": subject_start_result,  # (bsz, seq_l, 1)
                "subject_end_result": subject_end_result,  # (bsz, seq_l, 1)
                "object_start_result": object_start_result,  # (bsz, seq_l, relation_cnt)
                "object_end_result": object_end_result  # （bsz, seq_l, relation_cnt)
            }
        else:  # eval模式。该模式下，bsz默认为1
            # 获取subject的预测：cur_spans - SpanList
            subject_start_result = subject_start_result.squeeze()  # (seq_l)
            subject_end_result = subject_end_result.squeeze()  # (seq_l)
            subject_start_int = (subject_start_result > 0.5).int().tolist()
            subject_end_int = (subject_end_result > 0.5).int().tolist()
            seq_l = len(subject_start_int)
            cur_spans = tools.argument_span_determination(subject_start_int, subject_end_int, subject_start_result, subject_end_result)  # SpanList

            # 迭代获取每个subject所对应的object预测
            object_spans = []  #  List[List[SpanList]]  (subject, relation, span number)
            for elem_span in cur_spans:
                object_spans_for_current_subject = []  # List[SpanList]  (relation, span number)
                temporary_start, temporary_end = torch.zeros(1, 1, seq_l), torch.zeros(1, 1, seq_l)  # both (1, 1, seq_l)
                temporary_start[0][0][elem_span[0]] = 1
                temporary_end[0][0][elem_span[1]] = 1
                embed = calculate_embed_with_subject(output, temporary_start, temporary_end)  # (1, seq_l, hidden)

                # 计算object的预测
                object_start_result = self.object_start_cls(embed)  # (1, seq_l, relation_cnt)
                object_end_result = self.object_end_cls(embed)  # (1, seq_l, relation_cnt)

                object_start_result = object_start_result.squeeze().T  # (relation_cnt, seq_l)
                object_end_result = object_end_result.squeeze().T  # (relation_cnt, seq_l)
                for (object_start_r, object_end_r) in zip(object_start_result, object_end_result):
                    o_start_int = (object_start_r > 0.5).int().tolist()
                    o_end_int = (object_end_r > 0.5).int().tolist()
                    cur_spans = tools.argument_span_determination(o_start_int, o_end_int, object_start_r, object_end_r)  # SpanList
                    object_spans_for_current_subject.append(cur_spans)
                object_spans.append(object_spans_for_current_subject)

            return {
                "pred_subjects": cur_spans,  # SpanList
                "pred_objects": object_spans  # List[List[SpanList]]
            }


class CASREL_Loss(nn.Module):
    def __init__(self, lamb: float = 0.8):
        super(CASREL_Loss, self).__init__()
        self.lamb = lamb

    def forward(self,
                subject_start_result: torch.Tensor,
                subject_end_result: torch.Tensor,
                object_start_result: torch.Tensor,
                object_end_result: torch.Tensor,
                subject_start_label: torch.Tensor,
                subject_end_label: torch.Tensor,
                object_start_label: torch.Tensor,
                object_end_label: torch.Tensor):
        """

        :param subject_start_result: (bsz, seq_l, 1)
        :param subject_end_result: (bsz, seq_l, 1)
        :param object_start_result: (bsz, seq_l, relation_cnt)
        :param object_end_result: (bsz, seq_l, relation_cnt)
        :param subject_start_label: (bsz, seq_l)
        :param subject_end_label: (bsz, seq_l)
        :param object_start_label: (bsz, seq_l, relation_cnt)
        :param object_end_label: (bsz, seq_l, relation_cnt)
        :return:
        """
        # 将subject_result的形状与label对齐
        subject_start_result = subject_start_result.squeeze()  # (bsz, seq_l)
        subject_end_result = subject_end_result.squeeze()  # (bsz, seq_l)

        # 计算loss
        subject_start_loss = F.binary_cross_entropy(subject_start_result, subject_start_label)
        subject_end_loss = F.binary_cross_entropy(subject_end_result, subject_end_label)
        object_start_loss = F.binary_cross_entropy(object_start_result, object_start_label)
        object_end_loss = F.binary_cross_entropy(object_end_result, object_end_label)

        loss = self.lamb * (subject_start_loss + subject_end_loss) + (1 - self.lamb) * (object_start_loss + object_end_loss)
        return {
            "loss": loss
        }


class CASREL_Evaluator(BaseEvaluator):
    def __init__(self):
        super(CASREL_Evaluator, self).__init__()
        self.f1_evaluator = F1_Evaluator()

    def eval_single(self,
                pred_subjects: SpanList,
                pred_objects: List[List[SpanList]],
                subjects_gt: SpanList,
                objects_gt: List[List[SpanList]]):
        """

        :param pred_subjects:
        :param pred_objects:
        :param subjects_gt:
        :param objects_gt:
        :return:
        """
        pred_triplets = convert_lists_to_triplet_casrel(pred_subjects, pred_objects)  # List[Triplet]
        gt_triplets = convert_lists_to_triplet_casrel(subjects_gt, objects_gt)  # List[Triplet]
        self.f1_evaluator.eval_single(pred_triplets, gt_triplets)

    def eval_step(self) -> Dict[str, Any]:
        f1_result = self.f1_evaluator.eval_step()
        return f1_result


def train_dataset_factory(data_dicts: List[dict], bsz: int = RE_settings.default_bsz, shuffle: bool = RE_settings.default_shuffle):
    """
    data_dicts中的每一个dict包括
    - text
    - input_ids
    - token_type_ids
    - attention_mask
    - offset_mapping
    - triplets(contains token span)
        - subject
        - subject_occur
        - subject_token_span
        - object
        - object_occur
        - object_token_span
        - relation
    :param data_dicts:
    :param bsz:
    :param shuffle:
    :return:
    """
    def split_dict_by_subject(d) -> List[dict]:
        """
        d是一个包含句子的tokenize结果，以及所有triplets（包括char span与token span）
        本函数将d按照triplets中的subject划分，一个subject可能对应多个object_with_relation
        同时也会保存所有subject的SpanList
        :param d:
        :return:
        """
        subject_set = set()  # 用于判断subject是否已存在
        subject_dict = {}  # 存放用于作为subject gt与object label的所有数据
        subject_list: SpanList = []  # 存放用于作为subject label的token span
        regular_info = tools.split_dict(d, ['triplets'])
        for elem in d['triplets']:
            if elem['subject'] not in subject_set:
                subject_set.add(elem['subject'])
                subject_dict[elem['subject']] = {
                    "subject_occur": elem['subject_occur'],
                    "subject_token_span": elem['subject_token_span'],
                    "objects": [elem['object']],
                    "object_occurs": [elem['object_occur']],
                    "object_token_spans": [elem['object_token_span']],
                    'relations': [elem['relation']]
                }
                subject_list.append(elem['subject_token_span'])
            else:
                subject_dict[elem['subject']]['objects'].append(elem['object'])
                subject_dict[elem['subject']]['object_occurs'].append(elem['object_occur'])
                subject_dict[elem['subject']]['objects_token_spans'].append(elem['object_token_span'])
                subject_dict[elem['subject']]['relations'].append(elem['relation'])
        regular_info['subject_spans'] = subject_list
        results = []
        for k, v in subject_dict.items():
            new_dict = {
                "cur_objects": v['objects'],
                "cur_object_occurs": v['object_occurs'],
                "cur_object_token_spans": v['object_token_span'],
                "cur_relations": v['relations']
            }
            new_dict.update(regular_info)
            results.append(new_dict)
        return results

    data_dicts = tools.map_operation_to_list_elem(split_dict_by_subject, data_dicts)
    train_dataset = SimpleDataset(data_dicts)

    def collate_fn(lst):
        """
        dict in lst contains:
        - input_ids
        - tokens
        - token_type_ids
        - attention_mask
        - offset_mapping
        - subject_spans
        --
        - subject_occurs
        - subject_token_spans
        - cur_relations
        - cur_objects
        - cur_object_occurs
        - cur_object_token_spans
        :param lst:
        :return:
        """
        dicts = tools.transpose_list_of_dict(lst)





def dev_dataset_factory(data_dicts: List[dict]):
    pass


def dataset_factory(data_type: str, data_dir: str, bsz: int = RE_settings.default_bsz, shuffle: bool = RE_settings.default_shuffle):
    if data_type == 'NYT':
        data = RE_utils.load_NYT_re(data_dir)
        train_dicts, dev_dicts = data['train'], data['valid']
    elif data_type == 'WebNLG':
        data = RE_utils.load_WebNLG_re(data_dir)
        train_dicts, dev_dicts = data['train'], data['valid']
    elif data_type == 'duie':
        data = RE_utils.load_duie_re(data_dir)
        train_dicts, dev_dicts = data['train'], data['valid']
    else:
        raise Exception(f'[dataset_factory]遇到未知的数据集！[{data_type}]')

    # 首先生成train数据
    # Version-tokenize
    lst_tokenizer = tokenize_tools.bert_tokenizer(plm_path=RE_settings.default_plm[data_type])
    train_dicts = re_dataset.tokenize_re_dataset(train_dicts, lst_tokenizer)

    # Version-获得charSpan
    train_dicts = re_dataset.add_char_span_re_dataset(train_dicts)

    # Version-获得tokenSpan
    train_dicts = re_dataset.add_token_span_re_dataset(train_dicts)

    # Version-获得subject总体label
    train_dicts = re_dataset.generate_subject_labels(train_dicts)

    # Version-根据subject重组
    train_dicts = re_dataset.rearrange_by_subject(train_dicts)

    # Version-获取relation>object label
    train_dicts = re_dataset.generate_relation_to_object_labels(train_dicts, RE_settings.relations[data_type])

    # Version-获取与relation-object对应的subject gt
    train_dicts = re_dataset.generate_subject_gt_for_relation_object_pair(train_dicts)


    for elem in train_dicts:
        offset_mapping = elem['offset_mapping']
        for elem_d in elem['triplets']:
            RE_utils.add_token_span_to_re_data(elem_d, offset_mapping)
    for elem in dev_dicts:
        offset_mapping = elem['offset_mapping']
        for elem_d in elem['triplets']:
            RE_utils.add_token_span_to_re_data(elem_d, offset_mapping)




model_registry = {
    "model": CASREL,
    "evaluator": CASREL_Evaluator,

}
