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
from models.model_utils import get_init_params

"""
模型部分
"""


class SharedEncoder(nn.Module):
    def __init__(self, plm_path: str):
        super(SharedEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(plm_path)

    def forward(self, input_ids, token_type_ids, attention_mask):
        """

        :param input_ids:
        :param token_type_ids:
        :param attention_mask:
        :return:
        """
        result = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)
        return result[0]


class SimilarityModel(nn.Module):
    def __init__(self, hidden: int):
        super(SimilarityModel, self).__init__()
        self.hidden = hidden

        # 首先做一个简单的双层MLP
        self.l1_sim = nn.Linear(3 * hidden, hidden)
        self.l2_sim = nn.Linear(hidden, 1)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.l1_sim.weight)
        self.l1_sim.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.l2_sim.weight)
        self.l2_sim.bias.data.fill_(0)

    def forward(self, h: torch.Tensor, u: torch.Tensor):
        """

        :param h: (bsz, |C|, hidden)
        :param u: (bsz, |Q|, hidden)
        :return:
        """
        C, Q = h.shape[1], u.shape[1]
        h1 = h.unsqueeze(dim=2)  # (bsz, |C|, 1, hidden)
        u1 = u.unsqueeze(dim=1)  # (bsz, 1, |Q|, hidden)
        h2 = h1.expand(-1, -1, Q, -1)  # (bsz, |C|, |Q|, hidden)
        u2 = u1.expand(-1, C, -1, -1)  # (bsz, |C|, |Q|, hidden)
        elm_wise_mul = torch.mul(h2, u2)  # (bsz, |C|, |Q|, hidden)
        concatenated = torch.cat([h2, u2, elm_wise_mul], dim=-1)  # (bsz, |C|, |Q|, 3 * hidden)
        l1_output = F.leaky_relu(self.l1_sim(concatenated))  # (bsz, |C|, |Q|, hidden)
        l2_output = F.leaky_relu(self.l2_sim(l1_output))  # (bsz, |C|, |Q|, 1)
        S = l2_output.squeeze(dim=-1)  # (bsz, |C|, |Q|)
        return S  # (bsz, |C|, |Q|)


class FlowAttention(nn.Module):
    def __init__(self, hidden: int):
        super(FlowAttention, self).__init__()
        self.hidden = hidden
        # self.similarity_A = SimilarityModel(hidden)
        # self.similarity_R = SimilarityModel(hidden)
        self.similarity_common = SimilarityModel(hidden)
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)

    def forward(self, H: torch.Tensor, U: torch.Tensor, context_attention_mask: torch.Tensor, question_attention_mask: torch.Tensor):
        """
        计算句子嵌入H，与问句嵌入U的flow attention表示H_hat与U_hat
        其中H_hat与U_hat的维度均为(C, hidden)

        :param H: (bsz, C, hidden)
        :param U: (bsz, Q, hidden)
        :param context_attention_mask: (bsz, C)
        :param question_attention_mask: (bsz, Q)
        :return:
        """
        C, Q = H.shape[1], U.shape[1]
        S = self.similarity_common(H, U)  # (bsz, C, Q)

        # mask S
        context_bool_mask = (1 - context_attention_mask.unsqueeze(2)).bool()
        question_bool_mask = (1 - question_attention_mask.unsqueeze(1)).bool()
        S.masked_fill_(mask=context_bool_mask, value=torch.tensor(-1e9))
        S.masked_fill_(mask=question_bool_mask, value=torch.tensor(-1e9))

        # C2Q attention
        A = F.softmax(S, dim=-1)  # (bsz, C, Q)
        U_hat = torch.bmm(A, U)  # (bsz, C, hidden)

        # Q2C attention
        S = torch.max(S, 2)[0]  # (bsz, C)
        b = F.softmax(S, dim=-1)  # (bsz, C)
        q2c = torch.bmm(b.unsqueeze(1), H)  # (bsz, 1, hidden) = bmm( (bsz, 1, C), (bsz, C, hidden) )
        H_hat = q2c.repeat(1, C, 1)  # (bsz, C, hidden), tiled T times

        return H_hat, U_hat


class SharedProjection(nn.Module):
    """
    SharedProjection实际上已经并入FlowAttention
    paper也没有讲清楚二者的输入分别是什么，我也不好分开
    """
    def __init__(self, hidden):
        super(SharedProjection, self).__init__()
        self.hidden = hidden
        self.l1 = nn.Linear(hidden * 4, hidden * 2)
        self.l2 = nn.Linear(hidden * 2, hidden)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.l1.weight)
        self.l1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.l2.weight)
        self.l2.bias.data.fill_(0)

    def forward(self, H, H_hat, U_hat):
        """
        计算从问句与原句经过FlowAttention后的表示，结合在一起的映射
        :param H: (bsz, C, hidden)
        :param H_hat: (bsz, C, hidden)
        :param U_hat: (bsz, C, hidden)
        :return:
        """
        # combining
        G = torch.cat([H, U_hat, H.mul(U_hat), H.mul(H_hat)], dim=-1)  # (bsz, C, hidden * 4)

        G = self.l1(G)  # (bsz, C, hidden * 2)
        G = F.leaky_relu(self.l2(G))  # (bsz, C, hidden)

        return G  # (bsz, C, hidden)


class ArgumentClassifier(nn.Module):
    """EAR part
    """
    def __init__(self, hidden: int):
        super(ArgumentClassifier, self).__init__()
        self.hidden = hidden
        self.start_classifier = nn.Linear(hidden, 1, bias=False)
        self.end_classifier = nn.Linear(hidden, 1, bias=False)
        self.softmax1 = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.start_classifier.weight)
        torch.nn.init.xavier_uniform_(self.end_classifier.weight)

    def forward(self, G: torch.Tensor, context_mask: torch.Tensor):
        """

        :param G: (bsz, C, hidden)
        :param context_mask: (bsz, C) bool tensor
        :return:
        """
        start_digit = self.start_classifier(G)  # (bsz, C, 1)
        end_digit = self.end_classifier(G)  # (bsz, C, 1)

        start_digit = F.leaky_relu(start_digit)  # (bsz, C, 1)
        end_digit = F.leaky_relu(end_digit)  # (bsz, C, 1)

        start_digit = start_digit.squeeze(dim=-1)  # (bsz, C)
        end_digit = end_digit.squeeze(dim=-1)  # (bsz, C)

        # start_digit = start_digit.masked_fill(mask=context_mask, value=torch.tensor(-1e10))
        # end_digit = end_digit.masked_fill(mask=context_mask, value=torch.tensor(-1e10))

        start_prob = self.softmax1(start_digit)  # (bsz, C)
        end_prob = self.softmax1(end_digit)  # (bsz, C)
        # print(start_prob)

        start_prob = start_prob.masked_fill(mask=context_mask, value=torch.tensor(1e-10))
        end_prob = end_prob.masked_fill(mask=context_mask, value=torch.tensor(1e-10))

        # start_prob, end_prob = start_prob.squeeze(), end_prob.squeeze()  # both (bsz, C) or (C) if bsz == 1
        if len(start_prob.shape) == 1:
            start_prob = start_prob.unsqueeze(dim=0)
            end_prob = end_prob.unsqueeze(dim=0)
        return start_prob, end_prob  # both (bsz, C)


class RoleClassifier(nn.Module):
    """ERR part
    根据原论文，默认这个max_seq_len=256
    """

    def __init__(self, hidden: int, max_seq_len: int = 256, label_cnt: int = 19):
        """

        :param hidden:
        :param max_seq_len:
        """
        super(RoleClassifier, self).__init__()
        self.max_seq_len = max_seq_len
        self.hidden = hidden
        self.label_cnt = label_cnt

        self.conv3 = nn.Conv2d(1, 32, (3, hidden))
        self.conv4 = nn.Conv2d(1, 32, (4, hidden))
        self.conv5 = nn.Conv2d(1, 32, (5, hidden))
        self.conv6 = nn.Conv2d(1, 32, (6, hidden))

        self.classifier = nn.Linear(128, label_cnt + 1)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0)

    def forward(self, G: torch.Tensor):
        """
        原论文的表述有些疑问，所以我先按自己的想法实现一个
        :param G: (bsz, C, hidden)
        :return:
        """
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

        G_digits = self.classifier(G_max)  # (bsz, label_cnt)

        G_probs = F.softmax(G_digits, dim=-1)  # (bsz, label_cnt)

        return G_probs


class DualQA(nn.Module):
    def __init__(self, plm_path: str = dualqa_settings.plm_path, plm_lr: float = dualqa_settings.plm_lr, linear_lr: float = dualqa_settings.linear_lr):
        super(DualQA, self).__init__()
        self.init_params = get_init_params(locals())
        self.plm_path = plm_path
        self.shared_encoder = SharedEncoder(plm_path)
        self.hidden = self.shared_encoder.bert.config.hidden_size

        self.flow_attention = FlowAttention(self.hidden)
        self.shared_projection = SharedProjection(self.hidden)
        self.arg_classifier = ArgumentClassifier(self.hidden)
        self.role_classifier = RoleClassifier(self.hidden)

        self.linear_lr = linear_lr
        self.plm_lr = plm_lr

    def forward(self,
                context_input_ids: torch.Tensor,
                context_token_type_ids: torch.Tensor,
                context_attention_mask: torch.Tensor,
                EAR_input_ids: torch.Tensor = None,
                EAR_token_type_ids: torch.Tensor = None,
                EAR_attention_mask: torch.Tensor = None,
                ERR_input_ids: torch.Tensor = None,
                ERR_token_type_ids: torch.Tensor = None,
                ERR_attention_mask: torch.Tensor = None):
        """
        尽量遵循保持输入为纯tensor的原则，整个DualQA的输入分别是C，QA，QR的input_ids，token_type_ids，attention_mask

        在训练时，C，QA，QR均会作为输入，QA与QR都不能为None
        在预测时，如果QA部分为None而QR部分不为None，那么就只会进行ERR的预测，输出的EAR部分就为None
        如果QA不为None而QR为None，则对应的，进行EAR的预测，ERR输出None
        QA和QR不能同时为None
        :param context_input_ids: (bsz, C)
        :param context_token_type_ids:
        :param context_attention_mask:
        :param EAR_input_ids: (bsz, QA), 一定与qa_token_type_ids和qa_attention_mask同时为/不为None
        :param EAR_token_type_ids:
        :param EAR_attention_mask:
        :param ERR_input_ids: (bsz, QR), 一定与qr_token_type_ids和qr_attention_mask同时为/不为None
        :param ERR_token_type_ids:
        :param ERR_attention_mask:
        :return:
        """
        if EAR_input_ids is None and ERR_input_ids is None:
            raise Exception('[DualQA]QA与QR输入不能同时为None！')
        start_probs, end_probs, role_pred = None, None, None
        H = self.shared_encoder(
            input_ids=context_input_ids,
            token_type_ids=context_token_type_ids,
            attention_mask=context_attention_mask)

        # (bsz, C, hidden)
        # if self.training:
        #     if random.choice([0, 1]) == 0:
        #         EAR_input_ids = None
        #     else:
        #         ERR_input_ids = None

        if EAR_input_ids is not None:
            UA = self.shared_encoder(
                input_ids=EAR_input_ids,
                token_type_ids=EAR_token_type_ids,
                attention_mask=EAR_attention_mask)
            # (bsz, QA, hidden)
            H_A_hat, UA_hat = self.flow_attention(H, UA, context_attention_mask, EAR_attention_mask)  # both (bsz, C, hidden)
            GA = self.shared_projection(H, H_A_hat, UA_hat)  # (bsz, C, hidden)
            probs_mask = (1 - context_attention_mask).bool()
            start_probs, end_probs = self.arg_classifier(GA, probs_mask)  # both (bsz, C)
        if ERR_input_ids is not None:
            UR = self.shared_encoder(
                input_ids=ERR_input_ids,
                token_type_ids=ERR_token_type_ids,
                attention_mask=ERR_attention_mask)
            # (QR, hidden)
            # bsz, QR, h = UR.shape
            # UR = torch.zeros((bsz, QR, h)).cuda()
            H_R_hat, UR_hat = self.flow_attention(H, UR, context_attention_mask, ERR_attention_mask)  # both (bsz, C, hidden)
            GR = self.shared_projection(H, H_R_hat, UR_hat)  # (bsz, C, hidden)
            role_pred = self.role_classifier(GR)  # (bsz, role_cnt)

        return {
            "start_probs": start_probs,
            'end_probs': end_probs,
            "role_pred": role_pred,
        }

    def get_optimizers(self):
        # plm_params = self.shared_encoder.bert.parameters()
        # flow_params = self.flow_attention.parameters()
        # projection_params = self.shared_projection.parameters()
        # arg_cls_params = self.arg_classifier.parameters()
        # role_cls_params = self.role_classifier.parameters()
        # plm_optimizer = AdamW(params=plm_params, lr=self.plm_lr)
        # linear_optimizer = AdamW(params=chain(flow_params, projection_params, arg_cls_params, role_cls_params), lr=self.linear_lr)
        optimizer = AdamW(params=self.parameters(), lr=self.plm_lr)

        # return [plm_optimizer, linear_optimizer]
        return [optimizer]


class NaiveBERT(nn.Module):
    """
    用于定位模型问题
    """
    def __init__(self, plm_path: str = dualqa_settings.plm_path, plm_lr: float = dualqa_settings.plm_lr, linear_lr: float = dualqa_settings.linear_lr):
        super(NaiveBERT, self).__init__()
        self.plm_path = plm_path
        self.linear_lr = linear_lr
        self.plm_lr = plm_lr
        self.bert = BertModel.from_pretrained(plm_path)

        self.hidden = self.bert.config.hidden_size
        self.arg_linear_start = nn.Linear(self.hidden, 1)
        self.arg_linear_end = nn.Linear(self.hidden, 1)
        # self.role_linear = nn.Linear(self.hidden, len(EE_settings.role_types) + 1)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    def forward(self,
                context_input_ids: torch.Tensor,
                context_token_type_ids: torch.Tensor,
                context_attention_mask: torch.Tensor,
                EAR_input_ids: torch.Tensor = None,
                EAR_token_type_ids: torch.Tensor = None,
                EAR_attention_mask: torch.Tensor = None,
                ERR_input_ids: torch.Tensor = None,
                ERR_token_type_ids: torch.Tensor = None,
                ERR_attention_mask: torch.Tensor = None):
        """
        仅仅使用BERT
        :param context_input_ids:
        :param context_token_type_ids:
        :param context_attention_mask:
        :param EAR_input_ids:
        :param EAR_token_type_ids:
        :param EAR_attention_mask:
        :param ERR_input_ids:
        :param ERR_token_type_ids:
        :param ERR_attention_mask:
        :return:
        """
        input_strs = list(''.join(self.tokenizer.convert_ids_to_tokens(x)) for x in context_input_ids)
        question_strs = list(''.join(self.tokenizer.convert_ids_to_tokens(x)) for x in EAR_input_ids)
        if not self.training:
            print(f'input_strs:{input_strs}\nquesiton_strs:{question_strs}')
        com_input_ids = torch.cat([context_input_ids, EAR_input_ids], dim=1)  # (bsz, C + U, hidden)
        com_token_type_ids = torch.cat([context_token_type_ids, EAR_token_type_ids], dim=1)  # (bsz, C + U, hidden)
        com_attention_mask = torch.cat([context_attention_mask, EAR_attention_mask], dim=1)  # (bsz, C + U, hidden
        result = self.bert(input_ids=com_input_ids, token_type_ids=com_token_type_ids, attention_mask=com_attention_mask)
        # (bsz, |C|, hidden)

        embeds = result[0]  # (bsz, |C| + |Q|, hidden)
        start_output = self.arg_linear_start(embeds).squeeze(-1)  # (bsz, |C| + |Q|)
        end_output = self.arg_linear_end(embeds).squeeze(-1)  # (bsz, |C| + |Q|)
        return {
            'start_probs': start_output,
            'end_probs': end_output,
            'role_pred': torch.zeros(end_output.shape[0], 19).cuda()
        }

    def get_optimizers(self):
        plm_params = self.bert.parameters()
        linear1_param = self.arg_linear_end.parameters()
        linear2_param = self.arg_linear_start.parameters()
        plm_optim = AdamW(params=plm_params, lr=self.plm_lr)
        linear_optim = AdamW(params=chain(linear1_param, linear2_param), lr=self.linear_lr)
        return [plm_optim, linear_optim]


class FocalLoss(nn.Module):
    '''
    Multi-class Focal Loss
    '''
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        # self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C], float32
        target: [N, ], int64
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss


class DualQA_Loss(nn.Module):
    def __init__(self):
        super(DualQA_Loss, self).__init__()
        self.focal_loss = FocalLoss()

    def forward(self,
                start_probs: torch.Tensor,
                end_probs: torch.Tensor,
                role_pred: torch.Tensor,
                argument_start_label: torch.Tensor,
                argument_end_label: torch.Tensor,
                role_label: torch.Tensor):
        """

        :param start_probs: (bsz, C)
        :param end_probs: (bsz, C)
        :param role_pred: (bsz, role_cnt)
        :param argument_start_label: (bsz, 1) 对应着具体的下标
        :param argument_end_label: (bsz, 1)
        :param role_label: (bsz, 1)
        :return:
        """
        if start_probs is None:
            return F.nll_loss(torch.log(role_pred), role_label)
            # return self.focal_loss(role_pred, role_label)
        elif role_pred is None:
            # start_focal = self.focal_loss(start_probs, argument_start_label)
            # end_focal = self.focal_loss(end_probs, argument_end_label)
            start_loss = F.nll_loss(torch.log(start_probs), argument_start_label)
            end_loss = F.nll_loss(torch.log(end_probs), argument_end_label)
            return start_loss + end_loss
            # return start_focal + end_focal
        # 排查label超出边界的情况
        C = start_probs.shape[1]
        role_cnt = role_pred.shape[1]
        start_label_list, end_label_list, role_label_list = argument_start_label.tolist(), argument_end_label.tolist(), role_label.tolist()
        for idx in range(len(start_label_list)):
            if start_label_list[idx] >= C:
                argument_start_label[idx] = 0
                print('label out of border observed')
            if end_label_list[idx] >= C:
                argument_end_label[idx] = 0
        for idx in range(len(role_label_list)):
            if role_label_list[idx] >= role_cnt:
                role_label[idx] = role_cnt - 1
        start_loss = F.nll_loss(torch.log(start_probs), argument_start_label)
        end_loss = F.nll_loss(torch.log(end_probs), argument_end_label)
        role_loss = F.nll_loss(torch.log(role_pred), role_label)
        # start_loss = F.cross_entropy(start_probs, argument_start_label)
        # end_loss = F.cross_entropy(end_probs, argument_end_label)
        # role_loss = F.cross_entropy(role_pred, role_label)
        # argument_loss = start_loss + end_loss
        # loss = role_loss + argument_loss
        # # breakpoint()
        # return loss
        return start_loss + end_loss + role_loss


def concat_token_for_evaluate(tokens: List[str], span: Tuple[int, int]):
    """
    利用预测的span从输入模型的input_ids所对应的token序列中抽取出所需要的词语

    - 删除"##"
    - 对于(0, 0)会直接输出''
    :param tokens:
    :param span:
    :return:
    """
    if span == (0, 0):
        return ''
    result = ''.join(tokens[span[0]: span[1] + 1])
    result = result.replace('##', '')
    return result


class DualQA_Evaluator(BaseEvaluator):
    def __init__(self):
        super(DualQA_Evaluator, self).__init__()
        self.arg_kappa_evaluator = KappaEvaluator()
        self.arg_precision_evaluator = PrecisionEvaluator()
        self.role_kappa_evaluator = KappaEvaluator()
        self.role_precision_evaluator = PrecisionEvaluator()
        self.pred_lst = []
        self.gt_lst = []

    def eval_single(
            self,
            start_probs: torch.Tensor,
            end_probs: torch.Tensor,
            role_pred: torch.Tensor,
            tokens: List[str],
            argument_gt: str,
            role_idx_gt: int,
            threshold: int = 0.5):
        """

        :param start_probs: (1, C)
        :param end_probs: (1, C)
        :param role_pred: (1, role_cnt)
        :param tokens: token序列，用于根据start_probs与end_probs来还原argument
        :param argument_gt: 直接就是要抽取的argument word
        :param role_idx_gt: 直接就是要抽取的role类型的idx
        :param threshold:
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
        # spans = tools.argument_span_determination(start_digits, end_digits, start_probs, end_probs)
        # span = tuple(spans[0])  # 默认取第一个
        span = (start_position, end_position)
        argument_pred = concat_token_for_evaluate(tokens, span)

        role_pred = role_pred.squeeze().clone().detach().cpu()
        max_position = int(np.argsort(role_pred)[-1])

        # self.arg_kappa_evaluator.eval_single(argument_pred, argument_gt)
        self.arg_precision_evaluator.eval_single(argument_pred, argument_gt)
        # self.role_kappa_evaluator.eval_single(max_position, role_idx_gt)
        self.role_precision_evaluator.eval_single(max_position, role_idx_gt)
        self.pred_lst.append({
            'argument_pred': argument_pred,
            'role_pred': max_position
        })
        self.gt_lst.append({
            'argument_gt': argument_gt,
            'role_gt': role_idx_gt
        })

    def eval_step(self) -> Dict[str, Any]:
        # arg_kappa = self.arg_kappa_evaluator.eval_step()
        arg_p = self.arg_precision_evaluator.eval_step()
        # role_kappa = self.role_kappa_evaluator.eval_step()
        role_p = self.role_precision_evaluator.eval_step()

        # arg_kappa = tools.modify_key_of_dict(arg_kappa, lambda x: 'argument_' + x)
        arg_p = tools.modify_key_of_dict(arg_p, lambda x: 'argument_' + x)
        # role_kappa = tools.modify_key_of_dict(role_kappa, lambda x: 'role_' + x)
        role_p = tools.modify_key_of_dict(role_p, lambda x: 'role_' + x)

        result = {}
        result.update(arg_p)
        # result.update(arg_kappa)
        result.update(role_p)
        # result.update(role_kappa)

        self.pred_lst = []
        self.gt_lst = []
        return result

"""
数据处理部分
"""




def split_by_content_type_mention(data_dict: Dict[str, Any]):
    """
    每个样本包含content，type和mention
    trigger被视为一种mention
    :param data_dict:
    :return:
    """
    content = data_dict['content']
    events = data_dict['events']
    result_dicts = []
    for elem_event in events:
        event_type = elem_event['type']
        event_mentions = elem_event['mentions']
        for elem_mention in event_mentions:
            cur_sample = {
                'content': content,
                'event_type': event_type,
                'role': elem_mention['role'],
                'word': elem_mention['word'],
                'span': tuple(elem_mention['span'])
            }
            result_dicts.append(cur_sample)
    return result_dicts


def split_by_content_type(data_dicts: Dict[str, Any]):
    """
    按照content与事件类型划分。
    :param data_dicts:
    :return:
    """
    content = data_dicts['content']
    events = data_dicts['events']
    result_dicts = []
    for elem_event in events:
        event_type = elem_event['type']
        event_mentions = elem_event['mentions']
        cur_sample = {
            'content': content,
            'event_type': event_type,
            "mentions": event_mentions
        }
        result_dicts.append(cur_sample)
    return result_dicts


def split_by_questions(data_dict: Dict[str, Any]):
    content = data_dict['content']
    event_type = data_dict['event_type']
    trigger_info = data_dict['trigger_info']

    result_dicts = []

    other_mentions = data_dict['other_mentions']
    arg_qs = data_dict['argument_questions']
    arg_labels = data_dict['argument_labels']
    role_qs = data_dict['role_questions']
    role_labels = data_dict['role_labels']

    for (elem_mention, elem_arg_q, elem_arg_label, elem_role_q, elem_role_label) in zip(other_mentions, arg_qs,
                                                                                        arg_labels, role_qs,
                                                                                        role_labels):
        result_dicts.append({
            "content": content,
            "event_type": event_type,
            "trigger_info": trigger_info,
            "argument_info": elem_mention,
            "argument_question": elem_arg_q,
            "argument_label": elem_arg_label,
            "role_question": elem_role_q,
            "role_label": elem_role_label
        })

    return result_dicts


"""
直接从joint.py复制过来的
函数是一样，在以后就算遇到不同的情况，变量也不会很多
比较适合标准接口化的，不过我现在还没想好tools.py的最好组织方法，所以先留着
"""


def new_train_dataset_factory(data_dicts: List[dict], bsz: int = dualqa_settings, shuffle: bool = True):
    """
    对FewFC格式的数据进行预处理，包括以下步骤：

    - 去除过长的数据
    - 按照content-事件类型进行拆分，同时将触发词也剥离出来
    - 给事件中的每个论元，构造context, EAR_question, ERR_question
    - tokenize
    - 生成label
    :param data_dicts:
    :return:
    """
    cur_dataset = SimpleDataset(data_dicts)
    def collate_fn(lst):
        """
        需要context, QA, QR的input_ids, token_type_ids, attention_mask
        以及start_label, end_label, role_label
        :param lst:
        :return:
        """
        data_dict = tools.transpose_list_of_dict(lst)
        context_input_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['input_ids']), dtype=torch.long)
        context_token_type_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['token_type_ids']), dtype=torch.long)
        context_attention_mask = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['attention_mask']), dtype=torch.long)
        EAR_input_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['EAR_input_ids']), dtype=torch.long)
        EAR_token_type_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['EAR_token_type_ids']), dtype=torch.long)
        EAR_attention_mask = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['EAR_attention_mask']), dtype=torch.long)
        ERR_input_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['ERR_input_ids']), dtype=torch.long)
        ERR_token_type_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['ERR_token_type_ids']), dtype=torch.long)
        ERR_attention_mask = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['ERR_attention_mask']), dtype=torch.long)
        role_label = torch.tensor(data_dict['role_target'], dtype=torch.long)
        argument_start_label, argument_end_label = torch.tensor(np.array(data_dict['argument_target_start']), dtype=torch.long), \
                                                   torch.tensor(np.array(data_dict['argument_target_end']), dtype=torch.long)
        return {
            'context_input_ids': context_input_ids,
            'context_token_type_ids': context_token_type_ids,
            'context_attention_mask': context_attention_mask,
            'EAR_input_ids': EAR_input_ids,
            'EAR_token_type_ids': EAR_token_type_ids,
            'EAR_attention_mask': EAR_attention_mask,
            'ERR_input_ids': ERR_input_ids,
            'ERR_token_type_ids': ERR_token_type_ids,
            'ERR_attention_mask': ERR_attention_mask
               }, {
            'role_label': role_label,
            'argument_start_label': argument_start_label,
            'argument_end_label': argument_end_label
        }


    cur_dataloader = DataLoader(cur_dataset, batch_size=bsz, shuffle=shuffle, collate_fn=collate_fn)
    return cur_dataloader


def new_val_dataset_factory(data_dicts: List[dict]):
    """
    构造valid数据。包括以下步骤：

    - 去除过长的数据*
        严格来说是不能去除的，但是现在只是跑通流程，不需要管这些细节
    - 按照content-事件类型进行拆分
    - 给事件中每个论元，构造context, EAR_question, ERR_question
    - tokenize
    - 生成gt target
    :param data_dicts:
    :param dataset_type:
    :param stanza_nlp:
    :return:
    """

    cur_dataset = SimpleDataset(data_dicts)
    def collate_fn(lst):
        """
        这次生成的是用于评价的数据
        :param lst:
        :return:
        """
        data_dict = tools.transpose_list_of_dict(lst)
        context_input_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['input_ids']), dtype=torch.long)
        context_token_type_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['token_type_ids']), dtype=torch.long)
        context_attention_mask = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['attention_mask']), dtype=torch.long)
        EAR_input_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['EAR_input_ids']), dtype=torch.long)
        EAR_token_type_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['EAR_token_type_ids']), dtype=torch.long)
        EAR_attention_mask = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['EAR_attention_mask']), dtype=torch.long)
        ERR_input_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['ERR_input_ids']), dtype=torch.long)
        ERR_token_type_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['ERR_token_type_ids']), dtype=torch.long)
        ERR_attention_mask = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['ERR_attention_mask']), dtype=torch.long)

        return {
           'context_input_ids': context_input_ids,
           'context_token_type_ids': context_token_type_ids,
           'context_attention_mask': context_attention_mask,
           'EAR_input_ids': EAR_input_ids,
           'EAR_token_type_ids': EAR_token_type_ids,
           'EAR_attention_mask': EAR_attention_mask,
           'ERR_input_ids': ERR_input_ids,
           'ERR_token_type_ids': ERR_token_type_ids,
           'ERR_attention_mask': ERR_attention_mask
               }, {
            'argument_gt': data_dict['EAR_gt'][0],
            'role_idx_gt': data_dict['ERR_gt'][0],
            'tokens': data_dict['token'][0] if data_dict['token'][0] != '' else '[SEP]'
        }
    val_dataloader = DataLoader(cur_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return val_dataloader


def dataset_factory(dataset_type: str, train_file: str, valid_file: str, bsz: int):
    bsz = int(bsz)
    train_data_dicts = pickle.load(open(train_file, 'rb'))
    valid_data_dicts = list(json.loads(x) for x in open(valid_file, 'r', encoding='utf-8').read().strip().split('\n'))

    train_dataloader = new_train_dataset_factory(train_data_dicts, bsz=bsz)
    val_dataloader = new_val_dataset_factory(valid_data_dicts)
    return train_dataloader, val_dataloader


model_registry = {
    "model": DualQA,
    'loss': DualQA_Loss,
    'evaluator': DualQA_Evaluator,
    'train_val_data': dataset_factory,
    'recorder': NaiveRecorder
}


if __name__ == '__main__':
    train_loader, val_loader = dataset_factory('FewFC', 'temp_data/train.FewFC.labeled.balanced.pk', 'temp_data/valid.FewFC.tokenized.balanced.jsonl', bsz=1)
    train_samples, valid_samples = [], []
    for elem in train_loader:
        train_samples.append(elem)
    for elem in val_loader:
        valid_samples.append(elem)
