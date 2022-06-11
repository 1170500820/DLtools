"""
该模型删除了原JointEE中触发词的部分，而用论元来代替触发词的部分。

简而言之，这个模型会直接忽略触发词，而直接去预测论元
"""
from type_def import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertModel

import numpy as np
from itertools import chain
import pickle
import copy

from evaluate.evaluator import BaseEvaluator
from work.EE import EE_settings
from work.EE.JointEE_rebuild import jointee_settings
from work.EE.JointEE_rebuild.jointee_mask import TriggerExtractionLayerMask_woSyntactic
from models.model_utils import get_init_params
from analysis.recorder import NaiveRecorder
from utils import tools, tokenize_tools


def output_convert(event_types: StrList, arguments: List[List[SpanList]], role_types: List[str], content: str, offset_mapping=None):
    """
    将模型输出转换为DuEE比赛的输出格式
    :param event_types:
    :param arguments:
    :param role_types:
    :param content:
    :param offset_mapping:
    :return:
    """
    result = {
        'id': None,
        'event_list': None
    }
    event_list = []
    # convert arguments span
    for i_event in range(len(arguments)):
        cur_event_type = event_types[i_event]
        predicted_arguments = []
        for i_roletype in range(len(arguments[i_event])):
            cur_roletype = role_types[i_roletype]
            temp = list(map(lambda x: tokenize_tools.tokenSpan_to_charSpan((x[0] + 1, x[1] + 1), offset_mapping), arguments[i][j][k]))
            new_temp = list((x[0], x[1] + 1) for x in temp)
            arguments[i_event][i_roletype] = new_temp
            for elem_span in new_temp:
                word = content[elem_span[0]: elem_span[1]]
                predicted_arguments.append({
                    'argument': word,
                    'role': cur_roletype
                })
        event_list.append({
            'event_type': cur_event_type,
            'arguments': predicted_arguments
        })
    result['event_list'] = event_list
    return result


class JointEE_RoleOnly(nn.Module):
    def __init__(self,
                 plm_path=jointee_settings.plm_path,
                 n_head=jointee_settings.n_head,
                 d_head=jointee_settings.d_head,
                 hidden_dropout_prob=0.3,
                 plm_lr=EE_settings.plm_lr,
                 others_lr=EE_settings.others_lr,
                 argument_threshold=jointee_settings.argument_extraction_threshold,
                 dataset_type: str = 'FewFC',
                 use_cuda: bool = False,
                 eps: float = 1e-15):
        super(JointEE_RoleOnly, self).__init__()
        self.init_params = get_init_params(locals())  # 默认模型中包含这个东西。也许是个不好的设计
        # store init params

        if dataset_type == 'FewFC':
            self.role_types = EE_settings.role_types
        elif dataset_type == 'Duee':
            self.role_types = EE_settings.duee_role_types
        else:
            raise Exception(f'{dataset_type}数据集不存在！')

        self.plm_path = plm_path
        self.n_head = n_head
        self.d_head = d_head
        self.hidden_dropout_prob = hidden_dropout_prob
        self.plm_lr = plm_lr
        self.others_lr = others_lr
        self.argument_threshold = argument_threshold
        self.use_cuda = use_cuda

        # initiate network structures
        #   Sentence Representation
        #self.sentence_representation = SentenceRepresentation(self.plm_path, self.use_cuda)
        self.tokenizer = BertTokenizerFast.from_pretrained(plm_path)
        self.PLM = BertModel.from_pretrained(plm_path)

        self.hidden_size = self.PLM.config.hidden_size

        #   Conditional Layer Normalization
        self.weight_map = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.bias_map = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.t = nn.Parameter(torch.tensor(self.hidden_size, dtype=torch.float))
        self.bias = nn.Parameter(torch.tensor(self.hidden_size, dtype=torch.float))
        self.eps = eps
        self.reset_weight_and_bias()

        #   Argument Extraction
        #       因为这里是把
        self.aem = TriggerExtractionLayerMask_woSyntactic(
            num_heads=self.n_head,
            hidden_size=self.hidden_size,
            d_head=self.d_head,
            dropout_prob=self.hidden_dropout_prob)
        #   Triggered Sentence Representation
        #     不需要

        self.trigger_spanses = []
        self.argument_spanses = []

    def reset_weight_and_bias(self):
        """
        初始化的作用是在训练的开始阶段不让CLN起作用
        :return:
        """
        nn.init.ones_(self.t)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.weight_map.weight)
        nn.init.zeros_(self.bias_map.weight)

    def _conditional_layer_normalization(self,
                                         representation: torch.Tensor,
                                         condition: torch.Tensor,
                                         attention_mask: torch.Tensor):
        """

        :param representation: (bsz, max_real_seq_l, hidden)
        :param condition: (bsz, 1, hidden)
        :return:
        """
        weight = self.weight_map(condition) + self.t  # (bsz, 1, hidden_size)
        bias = self.bias_map(condition) + self.bias  # (bsz, 1, hidden_size)

        repr_mean = torch.mean(representation, dim=-1, keepdim=True)  # (bsz, max_real_seq_l, 1)
        repr_var = torch.var(representation, dim=-1, unbiased=False, keepdim=True)  # (bsz, max_real_seq_l, 1)

        offseted_repr = representation - repr_mean  # (bsz, max_real_seq_l, hidden)  broadcasted
        delta = torch.sqrt(repr_var + self.eps)  # (bsz, max_real_seq_l, 1)
        normed_repr = offseted_repr / delta  # (bsz, max_real_seq_l, hidden)  broadcasted
        denormed_repr = torch.multiply(weight, normed_repr) + bias  # (bsz, max_real_seq_l, hidden)
        # 重新将不存在的token部分的值置0
        masked_denormd_repr = denormed_repr * attention_mask.unsqueeze(-1)  # (bsz, max_real_seq_l, hidden)

        H_s_type = masked_denormd_repr  # (bsz, max_real_seq_l, hidden)

        return H_s_type

    def _sentence_representation(self, sentences: List[str], event_types: List[str]) -> Tuple[torch.Tensor, dict]:
        """
        将事件类型与句子拼在一起，即获得下面的形式
            [CLS] <事件类型> [SEP] <原句> [SEP]
        然后tokenize
        :param sentences:
        :param event_types: 每个句子所包含的一个事件类型
        :return: (bsz, max_expanded_seq_l, hidden) embed与tokenized
        """
        # 将事件类型与句子拼接在一起
        concated = []
        for elem_sent, elem_type in zip(sentences, event_types):
            concated_sentence = f'{elem_type}[SEP]{elem_sent}'
            concated.append(concated_sentence)

        # tokenize
        tokenized = self.tokenizer(concated, padding=True, truncation=True, return_tensors='pt',
                                   return_offsets_mapping=True)
        # 对不包含类型对原句进行tokenize。只需要其中的offset_mapping
        tokenized_no_type = self.tokenizer(sentences, padding=True, truncation=True, return_offsets_mapping=True)
        offsets_mapping = tokenized_no_type['offset_mapping']
        if self.use_cuda:
            input_ids, token_type_ids, attention_mask = tokenized['input_ids'].cuda(), tokenized[
                'token_type_ids'].cuda(), \
                                                        tokenized['attention_mask'].cuda()
        else:
            input_ids, token_type_ids, attention_mask = tokenized['input_ids'], tokenized['token_type_ids'], \
                                                        tokenized['attention_mask']
        tokenized = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'offset_mapping': offsets_mapping
        }

        # 获得预训练embedding
        output = self.PLM(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        embed = output[0]  # (bsz, max_seq_l, hidden)

        return embed, tokenized

    def _slice_embedding(self, embed: torch.Tensor, origin_tokenized: dict):
        """
        将embed进行切分，获取事件类型与句子部分的embedding
        对事件类型embedding进行MeanPooling
        计算句子部分的attention_mask
        :param embed: (bsz, max_expanded_seq_l, hidden)
        :param origin_tokenized:
        :return:
        """
        bsz = embed.shape[0]
        # 将事件类型切分出来
        sep_positions = (origin_tokenized['input_ids'] == 102).nonzero().T[1].tolist()
        H_cs, H_ss, real_seq_ls = [], [], []
        # H_cs: 对事件类型进行MeanPooling之后得到的tensor，每一个都是 ()
        # H_ss: 句子的token（不包括[CLS]与[SEP]）
        # real_seq_ls: 每个句子的真实长度（也就是不包括[CLS]与[SEP]））
        for i_batch in range(bsz):
            # 先切出事件类型，然后进行MeanPooling
            cur_H_c = embed[i_batch][1: sep_positions[i_batch * 2]]  # (type word length, hidden)
            pooled_cur_H_c = torch.mean(cur_H_c, dim=0)  # (hidden)
            pooled_cur_H_c = pooled_cur_H_c.unsqueeze(0)  # (1, hidden)
            H_cs.append(pooled_cur_H_c)
            # 接下来切出句子部分
            cur_H_s = embed[i_batch][sep_positions[i_batch * 2] + 1: sep_positions[i_batch * 2 + 1]]
            # (seq_l without special token, hidden)
            H_ss.append(cur_H_s)
            real_seq_ls.append(cur_H_s.shape[0])

        # pad and stack, 获得新的attention_mask
        max_seq_l = max(real_seq_ls)
        new_attention_masks = []
        for i in range(len(H_ss)):
            cur_seq_l = H_ss[i].shape[0]
            cur_attention_mask = torch.cat([torch.ones(cur_seq_l), torch.zeros(max_seq_l - cur_seq_l)])  # (max_seq_l)
            if self.use_cuda:
                cur_attention_mask = cur_attention_mask.cuda()
            new_attention_masks.append(cur_attention_mask)
            pad_tensor = torch.zeros(max_seq_l - cur_seq_l, self.hidden_size)
            if self.use_cuda:
                pad_tensor = pad_tensor.cuda()
            H_ss[i] = torch.cat([H_ss[i], pad_tensor])  # (max_seq_l, hidden)
        new_attention_mask = torch.stack(new_attention_masks)  # (bsz, max_real_seq_l)
        H_s = torch.stack(H_ss)  # (bsz, max_real_seq_l, hidden)
        H_c = torch.stack(H_cs)  # (bsz, 1, hidden)

        return H_s, H_c, new_attention_mask

    def forward(self,
                sentences: List[str],
                event_types: Union[List[str], List[StrList]]):
        """

        :param sentences:
        :param event_types:
        :return:
        """
        if self.training:
            # 将事件类型与句子拼接在一起
            embed, tokenized = self._sentence_representation(sentences, event_types)  # (bsz, max_seq_l, hidden)

            H_s, H_c, attention_mask = self._slice_embedding(embed, tokenized)
            # H_s: (bsz, max_real_seq_l, hidden)
            # H_c: (bsz, 1, hidden)
            # attention_mask: (bsz, max_real_seq_l, hidden)

            # Conditional Layer Normalization
            H_s_type = self._conditional_layer_normalization(H_s, H_c, attention_mask)  # (bsz, max_real_seq_l, hidden)

            argument_start, argument_end = self.aem(H_s_type, attention_mask)
            # both (bsz, max_real_seq_l, hidden)

            return {
                "argument_start": argument_start,
                "argument_end": argument_end
            }
        else:  # eval phase
            if len(sentences) != 1:
                raise Exception('eval模式下一次只预测单个句子!')
            cur_spanses = []  # List[List[SpanList]] 每个事件类型所对应的所有论元
            offset_mappings = []  #  List[OffsetMapping]
            for elem_sentence_type in event_types[0]:
                # elem_sentence_type: str
                embed, tokenized = self._sentence_representation(sentences, [elem_sentence_type])  # (bsz, max_real_seq_l, hidden)
                offset_mappings = tokenized['offset_mapping']
                H_s, H_c, attention_mask = self._slice_embedding(embed, tokenized)
                H_s_type = self._conditional_layer_normalization(H_s, H_c, attention_mask)  # (bsz, max_real_seq_l, hidden)

                # Argument Extraction
                argument_start_tensor, argument_end_tensor = self.aem(H_s_type, attention_mask)  # (bsz, max_seq_l, role_cnt)
                argument_start_tensor = argument_start_tensor.squeeze(0).T
                argument_end_tensor = argument_end_tensor.squeeze(0).T
                # both (role_cnt, max_real_seq_l)

                # (max_seq_l)
                argument_start_result = (argument_start_tensor > self.argument_threshold).int().tolist()
                argument_end_result = (argument_end_tensor > self.argument_threshold).int().tolist()

                role_for_cur_type: List[SpanList] = []
                for i_role in range(len(self.role_types)):
                    cur_arg_spans = tools.argument_span_determination(argument_start_result[i_role], argument_end_result[i_role], argument_start_tensor[i_role], argument_end_tensor[i_role])
                    role_for_cur_type.append(cur_arg_spans)
                cur_spanses.append(role_for_cur_type)  # List[List[SpanList]]

            result = output_convert(event_types[0], cur_spanses, self.role_types, sentences[0])
            return {
                'pred': result
            }


class ArgumentExtractionLayerMask_Direct(nn.Module):
    """
    因为不需要相对位置编码（RPE），直接对论元进行预测，因此称之为"Direct"
    初始化参数:
        - n_heads
        - hidden_size
        - d_head
        - dropout_prob
        - syntactic_size

    输入:
        cln_embeds: tensor (bsz, seq_l, hidden_size)

    输出:
        starts: tensor (bsz, seq_l)
        ends: tensor (bsz, seq_l)
    """
    def __init__(self, num_heads, hidden_size, d_head, dropout_prob, dataset_type: str = 'FewFC'):
        super(ArgumentExtractionLayerMask_Direct, self).__init__()
        # store params
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.d_head = d_head
        self.dropout_prob = dropout_prob

        if dataset_type == 'FewFC':
            self.role_types = EE_settings.role_types
        elif dataset_type == 'Duee':
            self.role_types = EE_settings.duee_role_types
        else:
            raise Exception(f'{dataset_type}数据集不存在！')

        # initiate network structures
        #   self-attention
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout_prob, batch_first=True)
        #   lstm
        # self.lstm = nn.LSTM(self.hidden_size * 2, self.hidden_size // 2,
        #                     dropout=self.dropout_prob, bidirectional=True, batch_first=True)
        # FCN for finding triggers
        self.fcn_start = nn.Linear(self.hidden_size * 2, len(self.role_types))
        self.fcn_end = nn.Linear(self.hidden_size * 2, len(self.role_types))

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.fcn_end.weight)
        self.fcn_end.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.fcn_start.weight)
        self.fcn_start.bias.data.fill_(0)

    def forward(self, cln_embeds: torch.Tensor, attention_mask: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        注意bsz=1的情况
        :param cln_embeds: (bsz, seq_l, hidden_size)， 经过CLN处理的句子embeddings
        :param attention_mask (bsz, seq_l)
        :return:
        """
        # self attention (multihead attention)
        attn_out, attn_out_weights = self.self_attn(cln_embeds, cln_embeds, cln_embeds, key_padding_mask=attention_mask)
        # todo attn_out: (bsz, seq_l, hidden) ?

        # concatenation
        final_repr = torch.cat((cln_embeds, attn_out), dim=-1)
        # final_repr: (bsz, seq_l, hidden * 2)

        # lstm_out, (_, __) = self.lstm(final_repr)
        # # lstm_out: (bsz, seq_l, hidden)

        # linear
        start_logits, end_logits = self.fcn_start(final_repr), self.fcn_end(final_repr)
        # ot both (bsz, seq_l, role_cnt)

        # sigmoid
        starts, ends = torch.sigmoid(start_logits), torch.sigmoid(end_logits)
        # both (bsz, seq_l, role_cnt)
        return starts, ends


class JointEE_RoleOnly_Loss(nn.Module):
    def __init__(self, pref: float = 2):
        super(JointEE_RoleOnly_Loss, self).__init__()
        self.pref = pref
        self.pos_pref_weight = tools.PosPrefWeight(pref)

    def forward(self,
                argument_start: torch.Tensor,
                argument_end: torch.Tensor,
                argument_label_start: torch.Tensor,
                argument_label_end: torch.Tensor,
                mask: torch.Tensor):
        """

        :param argument_start:
        :param argument_end: (bsz, max_real_seq_l, role_cnt)
        :param argument_label_start:
        :param argument_label_end: (bsz, max_real_seq_l, role_cnt)
        :param mask: (bsz, seq_l)
        :return:
        """
        mask = mask.unsqueeze(-1)  # (bsz, seq_l, 1)
        bsz = mask.shape[0]
        role_cnt = argument_start.shape[-1]

        argument_start_losses, argument_end_losses = [], []
        for i_batch in range(bsz):
            start_weight = self.arg_pos_pref_weight(argument_label_start[i_batch]).cuda()
            end_weight = self.arg_pos_pref_weight(argument_label_end[i_batch]).cuda()
            start_loss = F.binary_cross_entropy(argument_start[i_batch], argument_label_start[i_batch], start_weight,
                                                reduction='none')
            end_loss = F.binary_cross_entropy(argument_end[i_batch], argument_label_end[i_batch], end_weight,
                                              reduction='none')
            start_loss = torch.sum(start_loss * mask[i_batch]) / (torch.sum(mask[i_batch]) * role_cnt)
            end_loss = torch.sum(end_loss * mask[i_batch]) / (torch.sum(mask[i_batch]) * role_cnt)
            argument_start_losses.append(start_loss)
            argument_end_losses.append(end_loss)
        loss = sum(argument_start_losses) + sum(argument_end_losses)

        return loss


class JointEE_RoleOnly_Evaluator(BaseEvaluator):
    def __init__(self):
        super(JointEE_RoleOnly_Evaluator, self).__init__()
        self.gt_lst = []
        self.pred_lst = []
        self.info_dict = {
            'main': 'argument f1'
        }

    def eval_single(self, pred, gt):
        self.gt_lst.append(copy.deepcopy(gt))
        self.pred_lst.append(copy.deepcopy(pred))

    def eval_step(self) -> Dict[str, Any]:
        arg_total, arg_predict, arg_correct = 0, 0, 0
        for idx, elem_gt in enumerate(self.gt_lst):
            elem_pred = self.pred_lst[idx]

            # 首先计算都有的事件类型



def train_dataset_factory(data_dicts: List[dict], bsz: int = EE_settings.default_bsz, shuffle: bool = EE_settings.default_shuffle, dataset_type: str = 'Duee'):
    if dataset_type == 'FewFC':
        event_types = EE_settings.event_types_full
        role_types = EE_settings.role_types
    elif dataset_type == 'Duee':
        event_types = EE_settings.duee_event_types
        role_types = EE_settings.duee_role_types
    else:
        raise Exception(f'{dataset_type}数据集不存在！')
    train_dataset = SimpleDataset(data_dicts)

    def collate_fn(lst):
        """
        expect output:

        {
            sentence,
            event_type,
            trigger_span_gt,
        }, {
            trigger_label_start,
            trigger_label_end,
            argument_label_start,
            argument_label_end,
        }
        :param lst:
        :return:
        """
        data_dict = tools.transpose_list_of_dict(lst)
        bsz = len(lst)

        sentence_lst = data_dict['content']
        input_ids = data_dict['input_ids']
        max_seq_l = max(list(len(x) for x in input_ids)) - 2
        event_type_lst = data_dict['event_type']
        trigger_span_gt_lst = data_dict['trigger_token_span']
        arg_spans_lst = data_dict['argument_token_spans']

        trigger_label_start, trigger_label_end = torch.zeros((bsz, max_seq_l, 1)), torch.zeros((bsz, max_seq_l, 1))
        argument_label_start, argument_label_end = torch.zeros((bsz, max_seq_l, len(role_types))), torch.zeros((bsz, max_seq_l, len(role_types)))

        for i_batch in range(bsz):
            # trigger
            trigger_span = trigger_span_gt_lst[i_batch]
            trigger_label_start[i_batch][trigger_span[0] - 1] = 1
            trigger_label_end[i_batch][trigger_span[1] - 1] = 1
            # argument
            for e_role in arg_spans_lst[i_batch]:
                role_type_idx, role_span = e_role
                argument_label_start[i_batch][role_span[0] - 1][role_type_idx] = 1
                argument_label_end[i_batch][role_span[1] - 1][role_type_idx] = 1

        new_trigger_span_list = []
        for elem in trigger_span_gt_lst:
            new_trigger_span_list.append([elem[0] - 1, elem[1] - 1])

        return {
            'sentences': sentence_lst,
            'event_types': event_type_lst,
            'triggers': new_trigger_span_list
               }, {
            'trigger_label_start': trigger_label_start,
            'trigger_label_end': trigger_label_end,
            'argument_label_start': argument_label_start,
            'argument_label_end': argument_label_end
        }

    train_dataloader = DataLoader(train_dataset, batch_size=bsz, shuffle=shuffle, collate_fn=collate_fn)

    return train_dataloader


def valid_dataset_factory(data_dicts: List[dict], dataset_type: str = 'Duee'):
    if dataset_type == 'FewFC':
        event_types = EE_settings.event_types_full
        role_types = EE_settings.role_types
    elif dataset_type == 'Duee':
        event_types = EE_settings.duee_event_types
        role_types = EE_settings.duee_role_types
    else:
        raise Exception(f'{dataset_type}数据集不存在！')
    valid_dataset = SimpleDataset(data_dicts)

    def collate_fn(lst):
        """
        input:
            - sentences
            - event_types
            - offset_mapping
        eval:
            - gt
        :param lst:
        :return:
        """
        data_dict = tools.transpose_list_of_dict(lst)
        bsz = len(lst)

        sentences = data_dict['content']
        events = data_dict['events'][0]
        event_types = []
        offset_mappings = data_dict['offset_mapping']

        gt = []
        for elem in lst:
            gt.append({
                'id': '',
                'content': elem['content'],
                'events': elem['events']
            })
            event_types.append(list(x['type'] for x in events))
        return {
            'sentences': sentences,
            'event_types': event_types,
            'offset_mappings': offset_mappings
               }, {
            'gt': gt[0]
        }

    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    return valid_dataloader


def dataset_factory(train_file: str, valid_file: str, bsz: int = EE_settings.default_bsz, shuffle: bool = EE_settings.default_shuffle, dataset_type: str = 'Duee'):
    train_data_dicts = pickle.load(open(train_file, 'rb'))
    valid_data_dicts = pickle.load(open(valid_file, 'rb'))
    print(f'dataset_type: {dataset_type}')

    train_dataloader = train_dataset_factory(train_data_dicts, bsz=bsz, shuffle=shuffle, dataset_type=dataset_type)
    valid_dataloader = valid_dataset_factory(valid_data_dicts, dataset_type=dataset_type)

    return train_dataloader, valid_dataloader


def generate_trial_data(dataset_type: str):
    if dataset_type == 'Duee':
        train_file = 'temp_data/train.Duee.labeled.pk'
        valid_file = 'temp_data/valid.Duee.tokenized.pk'
    elif dataset_type == 'FewFC':
        # train_file = 'temp_data/train.PLMEE_Trigger.FewFC.labeled.pk'
        # valid_file = 'temp_data/valid.PLMEE_Trigger.FewFC.gt.pk'
        pass
    else:
        return None, None, None, None

    bsz = 4
    shuffle = False

    train_data_dicts = pickle.load(open(train_file, 'rb'))
    valid_data_dicts = pickle.load(open(valid_file, 'rb'))

    train_dataloader = train_dataset_factory(train_data_dicts, bsz=bsz, shuffle=shuffle, dataset_type=dataset_type)
    valid_dataloader = valid_dataset_factory(valid_data_dicts, dataset_type=dataset_type)

    limit = 5
    train_data, valid_data = [], []
    for idx, (train_sample, valid_sample) in enumerate(list(zip(train_dataloader, valid_dataloader))):
        train_data.append(train_sample)
        valid_data.append(valid_sample)
    return train_dataloader, train_data, valid_dataloader, valid_data


class UseModel:
    pass


model_registry = {
    'model': JointEE_RoleOnly,
    'loss': JointEE_RoleOnly_Loss,
    'evaluator': JointEE_RoleOnly_Evaluator,
    'train_val_data': dataset_factory,
    'recorder': NaiveRecorder,
    'use_model': UseModel
}


if __name__ == '__main__':
    pass
