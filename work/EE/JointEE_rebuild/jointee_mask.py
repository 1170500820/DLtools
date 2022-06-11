"""
考虑了mask的JointEE
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

from work.EE.EE_utils import *
from work.EE.JointEE_rebuild import jointee_settings
from work.EE.JointEE_rebuild.jointee import JointEE_Evaluator, tokenspans2events, dataset_factory, UseModel, \
    train_dataset_factory, valid_dataset_factory
from utils import tools
from utils.data import SimpleDataset
from analysis.recorder import NaiveRecorder
from models.model_utils import get_init_params
from utils import tokenize_tools


class JointEE_Mask(nn.Module):
    def __init__(self,
                 plm_path=jointee_settings.plm_path,
                 n_head=jointee_settings.n_head,
                 d_head=jointee_settings.d_head,
                 hidden_dropout_prob=0.3,
                 plm_lr=EE_settings.plm_lr,
                 others_lr=EE_settings.others_lr,
                 trigger_threshold=jointee_settings.trigger_extraction_threshold,
                 argument_threshold=jointee_settings.argument_extraction_threshold,
                 dataset_type: str = 'FewFC',
                 use_cuda: bool = False,
                 eps: float = 1e-15):
        super(JointEE_Mask, self).__init__()
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
        self.trigger_threshold = trigger_threshold
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

        #   Trigger Extraction
        self.tem = TriggerExtractionLayerMask_woSyntactic(
            num_heads=self.n_head,
            hidden_size=self.hidden_size,
            d_head=self.d_head,
            dropout_prob=self.hidden_dropout_prob)
        #   Triggered Sentence Representation
        #     不需要

        #   Argument Extraction
        self.aem = ArgumentExtractionModelMask_woSyntactic(
            n_head=self.n_head,
            d_head=self.d_head,
            hidden_size=self.hidden_size,
            dropout_prob=self.hidden_dropout_prob,
            dataset_type=dataset_type)

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


    def _trigger_sentence_representation(self, embed: torch.Tensor, trigger_spans: SpanList, attention_mask: torch.Tensor):
        """
        将触发词的信息加入input_ids当中，然后计算
        :param embed: 不包含<CLS>与<SEP>的embedding，(bsz, max_real_seq_l, hidden)
        :param trigger_spans: [(int, int), ...] 应当是左闭右闭
        :param attention_mask: (bsz, max_real_seq_l, hidden)
        :return:
        """
        bsz = embed.size(0)
        H_cs, H_ss = [], []
        for i_batch in range(bsz):
            cur_span = trigger_spans[i_batch]  # (int, int)
            cur_H_c = embed[i_batch][cur_span[0]:cur_span[1] + 1]  # (trigger length, hidden)
            pooled_cur_H_c = torch.mean(cur_H_c, dim=0)  # (1, hidden)
            H_cs.append(pooled_cur_H_c)
            H_ss.append(embed[i_batch])
        H_c = torch.stack(H_cs).unsqueeze(1)  # (bsz, 1, hidden)
        H_s = torch.stack(H_ss)  # (bsz, max_real_seq_l, hidden)

        H_s_type = self._conditional_layer_normalization(H_s, H_c, attention_mask)  # (bsz, max_real_seq_l, hidden)

        # Relative Positional Encoding
        rpes = []
        for s in range(bsz):
            cur_span = trigger_spans[s]
            rpe = torch.zeros(H_s_type.shape[1])
            for idx in range(len(rpe)):
                if idx <= cur_span[0]:
                    rpe[idx] = idx - cur_span[0]
                elif idx >= cur_span[1]:
                    rpe[idx] = idx - cur_span[1]
            rpes.append(rpe.unsqueeze(dim=1))   # (seq_l, 1)
        RPE = torch.stack(rpes)  # (bsz, seq_l, 1)
        if self.use_cuda:
            RPE = RPE.cuda()
        # todo 还需要生成relative positional embeddings
        return H_s_type, RPE


    def forward(self,
                sentences: List[str],
                event_types: Union[List[str], List[StrList]],
                triggers: SpanList = None,
                offset_mappings = None):
        """
        Train Phase:
            event_types为List[str]，与sentences中的句子一一对应，也与triggers中的每个span一一对应。
            首先通过sentences与event_types获得句子的表示[BERT + Pooling + CLN] (bsz, max_seq_l, hidden)
            然后用TEM预测出trigger；通过triggers中的Span（gt）预测出argument。

                - 注意，TEM预测出的trigger不会用于AEM，二者是没有联系的。

        Eval Phase:
            event_types为List[StrList]，是sentences中的每一个句子的所有包含的事件类型。
            预测出每一个事件类型的触发词，再接着预测论元。

                - 该阶段完全不会用到triggers参数

        :param sentences:
        :param event_types:
        :param triggers: 在训练阶段提供。
        :return:
        """
        bsz = len(sentences)
        if self.training:
            # 将事件类型与句子拼接在一起
            embed, tokenized = self._sentence_representation(sentences, event_types)  # (bsz, max_seq_l, hidden)

            H_s, H_c, attention_mask = self._slice_embedding(embed, tokenized)
            # H_s: (bsz, max_real_seq_l, hidden)
            # H_c: (bsz, 1, hidden)
            # attention_mask: (bsz, max_real_seq_l, hidden)

            # Conditional Layer Normalization
            H_s_type = self._conditional_layer_normalization(H_s, H_c, attention_mask)  # (bsz, max_real_seq_l, hidden)

            # Trigger Extraction
            trigger_start, trigger_end = self.tem(H_s_type, attention_mask)
            # both (bsz, max_real_seq_l, 1)
            arg_H_s_type, rpes = self._trigger_sentence_representation(H_s_type, triggers, attention_mask)
            # arg_H_s_type: (bsz, max_real_seq_l, hidden)

            # Argument Extraction
            arg_start, arg_end = self.aem(arg_H_s_type, rpes, attention_mask)
            # both (bsz, max_real_seq_l, len(role_types))

            return {
                'trigger_start': trigger_start,  # (bsz, max_real_seq_l, 1)
                'trigger_end': trigger_end,
                'argument_start': arg_start,  # (bsz, max_real_seq_l, role_cnt)
                'argument_end': arg_end,
                'mask': attention_mask
            }

        else:  # eval phase
            # sentence_types: List[StrList] during evaluating
            if len(sentences) != 1:
                raise Exception('eval模式下一次只预测单个句子!')
            cur_spanses = []  # List[SpanList]
            arg_spanses = []  # List[List[List[SpanList]]]
            offset_mappings = []  #  List[OffsetMapping]
            for elem_sentence_type in event_types[0]:
                # elem_sentence_type: str
                embed, tokenized = self._sentence_representation(sentences, [elem_sentence_type])  # (bsz, max_real_seq_l, hidden)
                offset_mappings = tokenized['offset_mapping']
                H_s, H_c, attention_mask = self._slice_embedding(embed, tokenized)
                H_s_type = self._conditional_layer_normalization(H_s, H_c, attention_mask)  # (bsz, max_real_seq_l, hidden)

                # Trigger Extraction
                trigger_start_tensor, trigger_end_tensor = self.tem(H_s_type, attention_mask)  # (bsz, max_seq_l, 1)
                trigger_start_tensor, trigger_end_tensor = trigger_start_tensor.squeeze(), trigger_end_tensor.squeeze()
                # (max_seq_l)
                trigger_start_result = (trigger_start_tensor > self.trigger_threshold).int().tolist()
                trigger_end_result = (trigger_end_tensor > self.trigger_threshold).int().tolist()
                cur_spans = tools.argument_span_determination(trigger_start_result, trigger_end_result, trigger_start_tensor, trigger_end_tensor)
                # cur_spans: SpanList, triggers extracted from current sentence

                arg_spans: List[List[SpanList]] = []
                for elem_trigger_span in cur_spans:
                    arg_H_s_type, rpes = self._trigger_sentence_representation(H_s_type, [elem_trigger_span], attention_mask)
                    argument_start_tensor, argument_end_tensor = self.aem(arg_H_s_type, rpes, attention_mask)
                    # (1, max_seq_l, len(role_types))
                    argument_start_tensor = argument_start_tensor.squeeze().T  # (role cnt, max_seq_l)
                    argument_end_tensor = argument_end_tensor.squeeze().T  # (role cnt, max_seq_l)
                    argument_start_result = (argument_start_tensor > self.argument_threshold).int().tolist()
                    argument_end_result = (argument_end_tensor > self.argument_threshold).int().tolist()
                    argument_spans: List[SpanList] = []
                    for idx_role in range(len(self.role_types)):
                        cur_arg_spans: SpanList = tools.argument_span_determination(argument_start_result[idx_role], argument_end_result[idx_role], argument_start_tensor[idx_role].tolist(), argument_end_tensor[idx_role].tolist())
                        argument_spans.append(cur_arg_spans)
                    arg_spans.append(argument_spans)
                cur_spanses.append(cur_spans)
                arg_spanses.append(arg_spans)
            self.trigger_spanses.append(cur_spanses)
            self.argument_spanses.append(arg_spanses)
            result = tokenspans2events(event_types[0], cur_spanses, arg_spanses, self.role_types, sentences[0], offset_mappings[0])
            return {"pred": result}

    def get_optimizers(self):
        repr_plm_params = self.PLM.parameters()
        cln_weight_params = self.weight_map.parameters()
        cln_bias_params = self.bias_map.parameters()
        aem_params = self.aem.parameters()
        tem_params = self.tem.parameters()
        plm_optimizer = AdamW(params=repr_plm_params, lr=self.plm_lr)
        others_optimizer = AdamW(params=chain(aem_params, tem_params, cln_bias_params, cln_weight_params, [self.t, self.bias]), lr=self.others_lr)
        return [plm_optimizer, others_optimizer]


class TriggerExtractionLayerMask_woSyntactic(nn.Module):
    """
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
    def __init__(self, num_heads, hidden_size, d_head, dropout_prob):
        super(TriggerExtractionLayerMask_woSyntactic, self).__init__()
        # store params
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.d_head = d_head
        self.dropout_prob = dropout_prob

        # initiate network structures
        #   self-attention
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout_prob, batch_first=True)
        #   lstm
        # self.lstm = nn.LSTM(self.hidden_size * 2, self.hidden_size // 2,
        #                     dropout=self.dropout_prob, bidirectional=True, batch_first=True)
        # FCN for finding triggers
        self.fcn_start = nn.Linear(self.hidden_size * 2, 1)
        self.fcn_end = nn.Linear(self.hidden_size * 2, 1)

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
        start_logits, end_logits = self.fcn_start(final_repr).squeeze(), self.fcn_end(final_repr).squeeze()
        # ot both (bsz, seq_l, 1), convert to (bsz, seq_l)

        # 需要保证batch维存在
        if len(start_logits.shape) == 1:
            start_logits = start_logits.unsqueeze(dim=0)
            end_logits = end_logits.unsqueeze(dim=0)
        # sigmoid
        starts, ends = torch.sigmoid(start_logits).unsqueeze(dim=-1), torch.sigmoid(end_logits).unsqueeze(dim=-1)  # got both (bsz, seq_l)
        return starts, ends


class ArgumentExtractionModelMask_woSyntactic(nn.Module):
    def __init__(self, n_head, d_head, hidden_size, dropout_prob, dataset_type: str = 'FewFC'):
        """

        :param n_head:
        :param hidden_size:
        :param d_head:True
        :param dropout_prob:
        """
        super(ArgumentExtractionModelMask_woSyntactic, self).__init__()

        if dataset_type == 'FewFC':
            self.role_types = EE_settings.role_types
        elif dataset_type == 'Duee':
            self.role_types = EE_settings.duee_role_types
        else:
            raise Exception(f'{dataset_type}数据集不存在！')
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.d_head = d_head
        self.dropout_prob = dropout_prob

        # initiate network structures
        #   self-attention
        self.self_attn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.n_head, dropout=self.dropout_prob, batch_first=True)
        #   FCN for finding triggers
        self.fcn_start = nn.Linear(self.hidden_size * 2 + 1, len(self.role_types))
        self.fcn_end = nn.Linear(self.hidden_size * 2 + 1, len(self.role_types))
        #   LSTM
        # self.lstm = nn.LSTM(self.hidden_size * 2 + 1, self.hidden_size//2,
        #                     batch_first=True, dropout=self.dropout_prob, bidirectional=True)

        self.init_weights()

        # code for not adding LSTM layer
        # FCN for trigger finding
        # origin + attn(origin) + syntactic + RPE
        # self.fcn_start = nn.Linear(self.hidden_size * 2 + ltp_embedding_dim + 1, len(role_types))
        # self.fcn_end = nn.Linear(self.hidden_size * 2 + ltp_embedding_dim + 1, len(role_types))

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.fcn_end.weight)
        self.fcn_end.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.fcn_start.weight)
        self.fcn_start.bias.data.fill_(0)


    def forward(self, cln_embeds: torch.Tensor, relative_positional_encoding: torch.Tensor, attention_mask: torch.Tensor):
        """

        :param cln_embeds: (bsz, seq_l, hidden_size)
        :param relative_positional_encoding: (bsz, seq_l, 1) todo 无效区域的距离设为inf还是0
        :return:
        """
        # self attention (multihead attention)
        attn_out, attn_out_weights = self.self_attn(cln_embeds, cln_embeds, cln_embeds, key_padding_mask=attention_mask)
        # attn_out: (bsz, seq_l, hidden)

        # concatenation
        final_repr = torch.cat((cln_embeds, attn_out, relative_positional_encoding), dim=-1)
        # final_repr: (bsz, seq_l, 2 * hidden + 1)

        # lstm_out, (_, __) = self.lstm(final_repr)
        # lstm_out: (bsz, seq_l, hidden)

        start_logits, end_logits = self.fcn_start(final_repr), self.fcn_end(final_repr)
        # start_logits and end_logits: (bsz, seq_l, len(role_types))

        starts, ends = torch.sigmoid(start_logits), torch.sigmoid(end_logits)
        # starts and ends: (bsz, seq_l, len(role_types))
        return starts, ends


class JointEE_MaskLoss(nn.Module):
    def __init__(self, lambd=0.1, pref=2):
        super(JointEE_MaskLoss, self).__init__()
        self.lambd = lambd
        self.arg_pos_pref_weight = tools.PosPrefWeight(pref)

    def forward(self,
                trigger_start: torch.Tensor,
                trigger_end: torch.Tensor,
                trigger_label_start: torch.Tensor,
                trigger_label_end: torch.Tensor,
                argument_start: torch.Tensor,
                argument_end: torch.Tensor,
                argument_label_start: torch.Tensor,
                argument_label_end: torch.Tensor,
                mask: torch.Tensor):
        """

        :param trigger_start:
        :param trigger_end: (bsz, seq_l, 1)
        :param trigger_label_start:
        :param trigger_label_end: (bsz, seq_l, 1)
        :param argument_start:
        :param argument_end: (bsz, seq_l, role_cnt)
        :param argument_label_start:
        :param argument_label_end:  (bsz, seq_l, role_cnt)
        :param mask: (bsz, seq_l)
        :return:
        """
        mask = mask.unsqueeze(-1)
        bsz = mask.shape[0]
        #concat_mask = mask[:, 1:-1]  # mask需要裁剪，因为原句没有CLS与SEP
        role_cnt = argument_start.shape[-1]

        trigger_start_losses, trigger_end_losses = [], []
        # trigger loss
        for i_batch in range(bsz):
            start_loss = F.binary_cross_entropy(trigger_start[i_batch], trigger_label_start[i_batch], reduction='none')
            end_loss = F.binary_cross_entropy(trigger_end[i_batch], trigger_label_end[i_batch], reduction='none')
            start_loss = torch.sum(start_loss * mask[i_batch]) / torch.sum(mask[i_batch])
            end_loss = torch.sum(end_loss * mask[i_batch]) / torch.sum(mask[i_batch])
            trigger_start_losses.append(start_loss)
            trigger_end_losses.append(end_loss)
        trigger_loss = sum(trigger_start_losses) + sum(trigger_end_losses)

        # argument loss
        argument_start_losses, argument_end_losses = [], []
        for i_batch in range(bsz):
            start_weight = self.arg_pos_pref_weight(argument_label_start[i_batch])
            end_weight = self.arg_pos_pref_weight(argument_label_end[i_batch])
            start_loss = F.binary_cross_entropy(argument_start[i_batch], argument_label_start[i_batch], start_weight, reduction='none')
            end_loss = F.binary_cross_entropy(argument_end[i_batch], argument_label_end[i_batch], end_weight, reduction='none')
            start_loss = torch.sum(start_loss * mask[i_batch]) / (torch.sum(mask[i_batch]) * role_cnt)
            end_loss = torch.sum(end_loss * mask[i_batch]) / (torch.sum(mask[i_batch]) * role_cnt)
            argument_start_losses.append(start_loss)
            argument_end_losses.append(end_loss)
        argument_loss = sum(argument_start_losses) + sum(argument_end_losses)
        # argument loss

        loss = self.lambd * trigger_loss + (1 - self.lambd) * argument_loss
        return loss


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


model_registry = {
    "model": JointEE_Mask,
    'loss': JointEE_MaskLoss,
    "evaluator": JointEE_Evaluator,
    'train_val_data': dataset_factory,
    'recorder': NaiveRecorder,
    'use_model': UseModel
}

if __name__ == '__main__':
    pass
