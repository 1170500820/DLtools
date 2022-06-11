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

from work.EE import EE_settings
from work.EE.JointEE_rebuild import jointee_settings
from work.EE.JointEE_rebuild.jointee_mask import TriggerExtractionLayerMask_woSyntactic
from models.model_utils import get_init_params
from utils import tools


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
                trigger_start_tensor, trigger_end_tensor = self.aem(H_s_type, attention_mask)  # (bsz, max_seq_l, role_cnt)
                trigger_start_tensor, trigger_end_tensor = trigger_start_tensor.squeeze(), trigger_end_tensor.squeeze()
                # (max_seq_l)
                trigger_start_result = (trigger_start_tensor > self.trigger_threshold).int().tolist()
                trigger_end_result = (trigger_end_tensor > self.trigger_threshold).int().tolist()

                for i_role in range(len(self.role_types)):
                    cur_arg_spans = tools.argument_span_determination()
                cur_spans = tools.argument_span_determination(trigger_start_result, trigger_end_result, trigger_start_tensor, trigger_end_tensor)
                # cur_spans: SpanList, triggers extracted from current sentence


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

