import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from utils import tools, tokenize_tools, batch_tool
from utils.data import SimpleDataset
from work.EE.EE_utils import *
from work.EE import EE_settings
import numpy as np
from torch.utils.data import DataLoader
from evaluate.evaluator import BaseEvaluator, KappaEvaluator, PrecisionEvaluator
from dataset.ee_dataset import load_FewFC_ee, load_Duee_ee_formated
import jieba
import stanza
import random
from work.EE.DualQA import dualqa_utils, dualqa_settings

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
        concatenated = torch.cat([h2, u2, h2 * u2], dim=-1)  # (bsz, |C|, |Q|, 3 * hidden)
        l1_output = nn.ReLU(self.l1_sim(concatenated))  # (bsz, |C|, |Q|, hidden)
        l2_output = nn.ReLU(self.l2_sim(l1_output))  # (bsz, |C|, |Q|, 1)
        l2_output = l2_output.squeeze()  # (bsz, |C|, |Q|) or (|C|, |Q|) if bsz == 1
        if len(l2_output.shape) == 2:
            l2_output = l2_output.unsqueeze(dim=0)  # (1, |C|, |Q|)
        return l2_output  # (bsz, |C|, |Q|)


class FlowAttention(nn.Module):
    def __init__(self, hidden: int):
        super(FlowAttention, self).__init__()
        self.hidden = hidden
        self.similarity_A = SimilarityModel(hidden)
        self.similarity_R = SimilarityModel(hidden)
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)

    def forward(self, H: torch.Tensor, U: torch.Tensor):
        """
        计算句子嵌入H，与问句嵌入U的flow attention表示H_hat与U_hat
        其中H_hat与U_hat的维度均为(C, hidden)

        :param H: (bsz, C, hidden)
        :param UA: (bsz, Q, hidden)
        :return:
        """
        C, Q = H.shape[1], U.shape[1]
        S = self.similarity_A(H, U)  # (bsz, C, Q)

        # C2Q attention
        a = self.softmax2(S)  # (bsz, C, Q)
        U_hat = torch.matmul(a, U)  # (bsz, C, hidden)

        # Q2C attention
        S_max, _ = torch.max(S, dim=2)  # (bsz, C)
        b = self.softmax1(S_max)  # (bsz, C)
        b = b.unsqueeze(dim=-1)  # (bsz, C, 1)
        H_hat = b * H  # (bsz, C, hidden)

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
        G = torch.cat([H, U_hat, H_hat * U_hat, H * H_hat], dim=-1)  # (bsz, C, hidden * 4)
        G = F.relu(self.l1(G))  # (bsz, C, hidden * 2)
        G = F.relu(self.l2(G))  # (bsz, C, hidden)

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

    def forward(self, G: torch.Tensor):
        """

        :param G: (bsz, C, hidden)
        :return:
        """
        start_digit = self.start_classifier(G)  # (bsz, C, 1)
        end_digit = self.end_classifier(G)  # (bsz, C, 1)

        start_prob = self.softmax1(start_digit)  # (bsz, C, 1)
        end_prob = self.softmax1(end_digit)  # (bsz, C, 1)

        start_prob, end_prob = start_prob.squeeze(), end_prob.squeeze()  # both (bsz, C) or (C) if bsz == 1
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

        self.classifier = nn.Linear(128, label_cnt)

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
    def __init__(self, plm_path: str):
        super(DualQA, self).__init__()
        self.plm_path = plm_path
        self.shared_encoder = SharedEncoder(plm_path)
        self.hidden = self.shared_encoder.bert.config.hidden_size

        self.flow_attention = FlowAttention(self.hidden)
        self.shared_projection = SharedProjection(self.hidden)
        self.arg_classifier = ArgumentClassifier(self.hidden)
        self.role_classifier = RoleClassifier(self.hidden)

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
        if EAR_input_ids is not None:
            UA = self.shared_encoder(
                input_ids=EAR_input_ids,
                token_type_ids=EAR_token_type_ids,
                attention_mask=EAR_attention_mask)
            # (bsz, QA, hidden)
            H_A_hat, UA_hat = self.flow_attention(H, UA)  # both (bsz, C, hidden)
            GA = self.shared_projection(H, H_A_hat, UA_hat)  # (bsz, C, hidden)
            start_probs, end_probs = self.arg_classifier(GA)  # both (bsz, C)
        if ERR_input_ids is not None:
            UR = self.shared_encoder(
                input_ids=ERR_input_ids,
                token_type_ids=ERR_token_type_ids,
                attention_mask=ERR_attention_mask)
            # (QR, hidden)
            H_R_hat, UR_hat = self.flow_attention(H, UR)  # both (bsz, C, hidden)
            GR = self.shared_projection(H, H_R_hat, UR_hat)  # (bsz, C, hidden)
            role_pred = self.role_classifier(GR)  # (bsz, role_cnt)

        if EAR_input_ids is None:
            return {
                "start_probs": start_probs,
                'end_probs': end_probs,
                "role_pred": role_pred
            }


class DualQA_Loss(nn.Module):
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
        start_loss = F.cross_entropy(start_probs, argument_start_label)
        end_loss = F.cross_entropy(end_probs, argument_end_label)
        role_loss = F.cross_entropy(role_pred, role_label)
        argument_loss = start_loss + end_loss
        loss = role_loss + argument_loss
        return loss


class DualQA_Evaluator(BaseEvaluator):
    def __init__(self):
        super(DualQA_Evaluator, self).__init__()
        self.arg_kappa_evaluator = KappaEvaluator()
        self.arg_precision_evaluator = PrecisionEvaluator()
        self.role_kappa_evaluator = KappaEvaluator()
        self.role_precision_evaluator = PrecisionEvaluator()

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
        start_probs = start_probs.squeeze()
        start_digits = (start_probs > threshold).int().tolist()
        start_probs = np.array(start_probs)  # (C)
        end_probs = end_probs.squeeze()
        end_digits = (end_probs > threshold).int().tolist()
        end_probs = np.array(end_probs)  # (C)
        spans = tools.argument_span_determination(start_digits, end_digits, start_probs, end_probs)
        span = tuple(spans[0])  # 默认取第一个
        argument_pred = ''.join(tokens[span[0]: span[1] + 1])

        role_pred = role_pred.squeeze()
        max_position = int(np.argsort(role_pred)[-1])

        self.arg_kappa_evaluator.eval_single(argument_pred, argument_gt)
        self.arg_precision_evaluator.eval_single(argument_pred, argument_gt)
        self.role_kappa_evaluator.eval_single(max_position, role_idx_gt)
        self.role_precision_evaluator.eval_single(max_position, role_idx_gt)

    def eval_step(self) -> Dict[str, Any]:
        arg_kappa = self.arg_kappa_evaluator.eval_step()
        arg_p = self.arg_precision_evaluator.eval_step()
        role_kappa = self.role_kappa_evaluator.eval_step()
        role_p = self.role_precision_evaluator.eval_step()

        arg_kappa = tools.modify_key_of_dict(arg_kappa, lambda x: 'argument_' + x)
        arg_p = tools.modify_key_of_dict(arg_p, lambda x: 'argument_' + x)
        role_kappa = tools.modify_key_of_dict(role_kappa, lambda x: 'role_' + x)
        role_p = tools.modify_key_of_dict(role_p, lambda x: 'role_' + x)

        result = {}
        result.update(arg_p)
        result.update(arg_kappa)
        result.update(role_p)
        result.update(role_kappa)
        return result

"""
数据处理部分
"""


def split_by_content_type_trigger(data_dict: Dict[str, Any]):
    content = data_dict['content']
    events = data_dict['events']

    result_dicts = []

    for elem_event in events:
        event_type = elem_event['type']
        event_mentions = elem_event['mentions']

        # 分离trigger与argument
        triggers = []
        other_mentions = []
        for elem_mention in event_mentions:
            if elem_event['role'] == 'trigger':
                triggers.append(elem_mention)
            else:
                other_mentions.append(elem_mention)
        if len(triggers) != 1:
            raise Exception(f'[split_by_content_type_trigger]不合法的mentions！包含的trigger个数错误。应当为1，实际为{len(triggers)}')

        cur_sample = {
            "content": content,
            "events": events,
            "event_type": event_type,
            "trigger_info": triggers[0],
            "other_mentions": other_mentions
        }
        result_dicts.append(cur_sample)

    return result_dicts


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


def construct_EAR_ERR_context(data_dict: Dict[str, Any], dataset_type: str, stanza_nlp):
    """
    同时构建EAR,ERR与context

    :param data_dict: [content, event_type, trigger_info, other_mentions]
    :param dataset_type:
    :param stanza_nlp:
    :return:
    """
    content, event_type, trigger_info, mentions = \
        data_dict['content'], data_dict['event_type'], data_dict['trigger_info'], data_dict['other_mentions']

    trigger_span = trigger_info['span']
    # 首先构建context
    context_sentence = f"{event_type}[SEP]{content[:trigger_span[0]]}[SEP]{content[trigger_span[0]:trigger_span[1]]}[SEP]{content[trigger_span[1]:]}"

    # 先构建EAR
    EAR_questions = []
    if dataset_type == 'FewFC':
        schema = EE_settings.event_available_roles
        for key, value in schema.items():
            schema[key] = list(EE_settings.role_types_translate[x] for x in value)
        role_types = list(EE_settings.role_types_translate[x] for x in EE_settings.role_types)
    elif dataset_type == 'Duee':
        schema = EE_settings.duee_event_available_roles
        role_types = EE_settings.duee_role_types
    else:
        raise Exception(f'数据集{dataset_type}不存在！')
    role_index = {v: i for i, v in enumerate(role_types)}
    EAR_results = []
    exist_role = set()
    for elem_mention in event_type['mentions']:
        exist_role.add(elem_mention['role'])
        role, span, word = elem_mention['role'], tuple(elem_mention['span']), elem_mention['word']
        cur_sample = {
            'content': content,
            'event_type': event_type,
            'role': role,
            'span': span,
            'word': word
        }
        EAR_results.append(cur_sample)
    for elem_role in schema[event_type]:
        if elem_role not in exist_role:
            cur_sample = {
                'content': content,
                'event_type': event_type,
                'role': elem_role,
                'span': (0, 0),
                'word': None
            }
            EAR_results.append(cur_sample)
    for elem_info in EAR_results:
        role, word = elem_info['role'], elem_info['word']
        question = f'词语{word}在事件{event_type}中作为什么角色？'
        EAR_questions.append({
            'question': question,
            'label': role,
            'EAR_gt': word if word is not None else ''  # 如果为负例，把None切换成空字符串
        })

    # 然后构建ERR
    ERR_questions = []
    words = list(jieba.cut(content))
    entities = list(x['text'] for x in stanza_nlp(content).ents)
    ERR_results = []
    exist_words = set(x['word'] for x in data_dict['mentions'])
    for elem in exist_words:
        if elem in entities:
            entities.remove(elem)
        if elem in words:
            words.remove(elem)
    for elem_mention in data_dict['mentions']:
        # pos
        role, span, word = elem_mention['role'], tuple(elem_mention['span']), elem_mention['word']
        ERR_results.append({
            'content': content,
            'event_type': event_type,
            "word": word,
            'role': role
        })
    # ERR的数量需要与EAR相同，所以构建剩余的
    pos_cnt = len(ERR_results)
    for elem_result in EAR_results[pos_cnt:]:
        # neg
        random_word = random.choice(entities) if len(entities) != 0 else None
        if random_word is None or random_word in exist_words:
            entities.remove(random_word)
            random_word = random.choice(words)  # words的长度应该不会为0吧
            words.remove(random_word)
        ERR_results.append({
            'content': content,
            'event_type': event_type,
            "word": random_word,
            'role': None
        })
    for elem_info in ERR_results:
        role, word = elem_info['role'], elem_info['word']
        question = f'词语{word}在事件{event_type}中作为什么角色？'
        ERR_questions.append({
            'question': question,
            'label': role,
            'ERR_gt': role_index[role] if role is not None else len(role_index)  # 负例在末尾
        })

    results = []
    for idx, (elem_a, elem_r) in enumerate(zip(EAR_questions, ERR_questions)):
        results.append({
            "context": context_sentence,
            "EAR_question": elem_a['question'],
            "EAR_label": elem_a['label'],
            'ERR_question': elem_r['question'],
            'ERR_label': elem_r['label'],
            'content': content,
            'EAR_gt': elem_a['EAR_gt'],
            'ERR_gt': elem_r['ERR_gt']
        })
    return results


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


def wrapped_find_matches(data_dict: Dict[str, Any]):
    content = data_dict['content']
    tokens = data_dict['token']
    token2origin, origin2token = tools.find_matches(content, tokens)
    data_dict['token2origin'] = token2origin
    data_dict['origin2token'] = origin2token
    return [data_dict]


def generate_EAR_target(data_dict: Dict[str, Any]):
    """
    也就是生成argument的结果的label
    :param data_dict:
    :return:
    """
    argument_label = data_dict['argument_label']  # origin span
    tokens = data_dict['tokens']
    token2origin, origin2token = data_dict['token2origin'], data_dict['origin2token']
    argument_target_start, argument_target_end = np.zeros(len(tokens)), np.zeros(len(tokens))  # (role_cnt)
    token_start, token_end = origin2token[argument_label[0]] - 1, origin2token[argument_label[0] - 1] - 1
    argument_target_start[token_start] = 1
    argument_target_end[token_end] = 1
    data_dict['argument_target_start'] = argument_target_start
    data_dict['argument_target_end'] = argument_target_end
    return [data_dict]


def new_generate_EAR_target(data_dict: Dict[str, Any]):
    """

    :param data_dict:
    :return:
    """
    input_ids = data_dict['input_ids']
    offset_mapping = data_dict['offset_mapping']
    input_ids_length = len(input_ids)
    argument_start_target, argument_end_target = np.zeros(input_ids_length), np.zeros(input_ids_length)
    span = data_dict['EAR_label']
    if span == (0, 0):
        token_span = (0, 0)
    else:
        token_span = (-1, -1)
        for elem in offset_mapping:
            if elem[0] == span[0] + 1:
                token_span = (elem[1], token_span[1])
            if elem[0] == span[1]:
                token_span = (token_span[0], elem[1])
    argument_start_target[token_span[0]] = 1
    argument_end_target[token_span[1]] = 1
    data_dict['argument_target_start'] = argument_start_target
    data_dict['argument_target_end'] = argument_end_target
    return [data_dict]


def generate_ERR_target(data_dict: Dict[str, Any]):
    """
    生成role的结果
    :param data_dict:
    :return:
    """
    role_label = data_dict['role_label']
    role_target = np.zeros(len(EE_settings.role_types))
    role_target[EE_settings.role_index[role_label]] = 1

    data_dict['role_target'] = role_target  # (role_cnt)
    return [data_dict]


def new_generate_ERR_target(data_dict: Dict[str, Any], dataset_type: str):
    if dataset_type == 'FewFC':
        role_types = list(EE_settings.role_types_translate[x] for x in EE_settings.role_types)
    elif dataset_type == 'Duee':
        role_types = EE_settings.duee_role_types
    else:
        raise Exception(f'不存在{dataset_type}数据集!')
    role_index = {v: i for i, v in enumerate(role_types)}
    role_label = data_dict['ERR_label']
    role_target = np.zeros(len(role_types) + 1)  # 在最后面加上一个负例label
    if role_label is None:
        role_target[-1] = 1
    else:
        role_target[role_index[role_label]] = 1
    data_dict['role_target'] = role_target  # (role_cnt)
    return [data_dict]


def generate_EAR_gt(data_dict: Dict[str, Any]):
    data_dict['start_gt']


def generate_ERR_gt(data_dict: Dict[str, Any]):
    pass


def train_dataset_factory(train_filename: str, bsz=4):
    json_lines = tools.read_json_lines(train_filename)
    json_dict_lines = [{'content': x['content'], 'events': x['events']} for x in json_lines]
    # [content, events]

    # 过滤长度非法的content，替换非法字符
    data_dicts = tools.map_operation_to_list_elem(remove_function, json_dict_lines)
    data_dicts = tools.map_operation_to_list_elem(remove_illegal_length, data_dicts)
    # get [content, events]

    # 按content-事件类型-触发词-进行划分
    data_dicts = tools.map_operation_to_list_elem(split_by_content_type_trigger, data_dicts)
    # [content, event_type, trigger_info, other_mentions]

    # 生成问句对与label
    # *需要进一步划分问题则在此部分之前插入
    # data_dicts = tools.map_operation_to_list_elem(generate_EAR_questions_and_labels, data_dicts)
    # data_dicts = tools.map_operation_to_list_elem(generate_ERR_questions_and_labels, data_dicts)
    # [content, event_type, trigger_info, other_mentions, argument_questions, argument_labels, role_questions, role_labels]

    # 按照问句划分
    data_dicts = tools.map_operation_to_list_elem(split_by_questions, data_dicts)
    # [content, event_type, trigger_info, argument_info, argument_question, argument_label, role_question, role_label]

    # tokenize
    data_dict = tools.transpose_list_of_dict(data_dicts)
    lst_tokenizer = tools.bert_tokenizer()
    content_result = lst_tokenizer(data_dict['content'])
    content_result = tools.transpose_list_of_dict(content_result)
    content_result = tools.modify_key_of_dict(content_result, lambda x: x)
    arg_q_result = lst_tokenizer(data_dict['argument_question'])
    arg_q_result = tools.transpose_list_of_dict(arg_q_result)
    arg_q_result = tools.modify_key_of_dict(arg_q_result, lambda x: 'argument_' + x)
    role_q_result = lst_tokenizer(data_dict['role_question'])
    role_q_result = tools.transpose_list_of_dict(role_q_result)
    role_q_result = tools.modify_key_of_dict(role_q_result, lambda x: 'role_' + x)
    data_dict.update(content_result)
    data_dict.update(arg_q_result)
    data_dict.update(role_q_result)
    data_dicts = tools.transpose_dict_of_list(data_dict)
    # [
    # content, event_type, trigger_info, other_mentions,
    # input_ids, token_type_ids, attention_mask,
    # argument_question, argument_label, argument_input_ids, argument_token_type_ids, argument_attention_mask,
    # role_question, role_label, role_input_ids, role_token_type_ids, role_attention_mask]

    # 计算match
    data_dicts = tools.map_operation_to_list_elem(wrapped_find_matches, data_dicts)
    # [
    # content, event_type, trigger_info, other_mentions, match
    # input_ids, token_type_ids, attention_mask,
    # argument_question, argument_label, argument_input_ids, argument_token_type_ids, argument_attention_mask,
    # role_question, role_label, role_input_ids, role_token_type_ids, role_attention_mask]

    # 将label进行tensor化
    data_dicts = tools.map_operation_to_list_elem(generate_EAR_target, data_dicts)
    data_dicts = tools.map_operation_to_list_elem(generate_ERR_target, data_dicts)
    # [
    # content, event_type, trigger_info, other_mentions, match
    # input_ids, token_type_ids, attention_mask,
    # argument_question, argument_label, argument_target,
    # argument_input_ids, argument_token_type_ids, argument_attention_mask,
    # role_question, role_label, role_target,
    # role_input_ids, role_token_type_ids, role_attention_mask]

    # produce dataset
    train_dataset = SimpleDataset(data_dicts)

    def collate_fn(lst):
        dict_of_data = tools.transpose_list_of_dict(lst)

        input_ids = [np.array(x) for x in dict_of_data['input_ids']]
        token_type_ids = [np.array(x) for x in dict_of_data['token_type_ids']]
        attention_mask = [np.array(x) for x in dict_of_data['attention_mask']]
        argument_input_ids = [np.array(x) for x in dict_of_data['argument_input_ids']]
        argument_token_type_ids = [np.array(x) for x in dict_of_data['argument_token_type_ids']]
        argument_attention_mask = [np.array(x) for x in dict_of_data['argument_attention_mask']]
        role_input_ids = [np.array(x) for x in dict_of_data['role_input_ids']]
        role_token_type_ids = [np.array(x) for x in dict_of_data['role_token_type_ids']]
        role_attention_mask = [np.array(x) for x in dict_of_data['role_attention_mask']]

        input_ids = torch.tensor(tools.batchify_ndarray(input_ids))
        token_type_ids = torch.tensor(tools.batchify_ndarray(token_type_ids))
        attention_mask = torch.tensor(tools.batchify_ndarray(attention_mask))
        argument_input_ids = torch.tensor(tools.batchify_ndarray(argument_input_ids))
        argument_token_type_ids = torch.tensor(tools.batchify_ndarray(argument_token_type_ids))
        argument_attention_mask = torch.tensor(tools.batchify_ndarray(argument_attention_mask))
        role_input_ids = torch.tensor(tools.batchify_ndarray(role_input_ids))
        role_token_type_ids = torch.tensor(tools.batchify_ndarray(role_token_type_ids))
        role_attention_mask = torch.tensor(tools.batchify_ndarray(role_attention_mask))

        argument_target_start = torch.tensor(tools.batchify_ndarray(dict_of_data['argument_target_start']))
        argument_target_end = torch.tensor(tools.batchify_ndarray(dict_of_data['argument_target_end']))
        role_target = torch.tensor(tools.batchify_ndarray(dict_of_data['role_target']))

        return {
                   'input_ids': input_ids,  # (bsz, seq_l)
                   'token_type_ids': token_type_ids,  # (bsz, seq_l)
                   'attention_mask': attention_mask,  # (bsz, seq_l)
                   'argument_input_ids': argument_input_ids,  # (bsz, seq_l)
                   'argument_token_type_ids': argument_token_type_ids,  # (bsz, seq_l)
                   'argument_attention_mask': argument_attention_mask,  # (bsz, seq_l)
                   'role_input_ids': role_input_ids,  # (bsz, seq_l)
                   'role_token_type_ids': role_token_type_ids,  # (bsz, seq_l)
                   'role_attention_mask': role_attention_mask,  # (bsz, seq_l)
               }, {
                   'argument_target_start': argument_target_start,  # (bsz, seq_l)
                   'argument_target_end': argument_target_end,  # (bsz, seq_l)
                   'role_target': role_target  # (bsz, role_cnt)
               }

    train_dataloader = DataLoader(train_dataset, batch_size=bsz, shuffle=True, collate_fn=collate_fn)
    return train_dataloader


def val_dataset_factory(val_filename: str):
    pass


def new_train_dataset_factory(data_dicts: List[dict], dataset_type: str, stanza_nlp, bsz: int = dualqa_settings, shuffle: bool = True):
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
    # 去除content过长的sample
    data_dicts = tools.map_operation_to_list_elem(remove_illegal_length, data_dicts)
    # [content, events]

    # 按content-事件类型-触发词-进行划分
    data_dicts = tools.map_operation_to_list_elem(split_by_content_type_trigger, data_dicts)
    # [content, event_type, trigger_info, other_mentions]

    # 同时构造context，EAR问题与ERR问题。
    results = []
    for elem in data_dicts:
        results.extend(construct_EAR_ERR_context(elem, dataset_type, stanza_nlp))
    # [context, content, EAR_question, EAR_label, ERR_question, ERR_label]

    # tokenize
    data_dict = tools.transpose_list_of_dict(results)
    lst_tokenizer = tokenize_tools.bert_tokenizer()
    context_result = lst_tokenizer(data_dict['context'])
    context_result = tools.transpose_list_of_dict(context_result)
    context_result = tools.modify_key_of_dict(context_result, lambda x: x)
    EAR_result = lst_tokenizer(data_dict['EAR_question'])
    EAR_result = tools.transpose_list_of_dict(EAR_result)
    EAR_result = tools.modify_key_of_dict(EAR_result, lambda x: 'EAR_' + x)
    ERR_result = lst_tokenizer(data_dict['ERR_question'])
    ERR_result = tools.transpose_list_of_dict(ERR_result)
    ERR_result = tools.modify_key_of_dict(ERR_result, lambda x: 'ERR_' + x)
    data_dict.update(context_result)
    data_dict.update(EAR_result)
    data_dict.update(ERR_result)
    data_dicts = tools.transpose_dict_of_list(data_dict)

    # 为label生成nparray
    data_dicts = tools.map_operation_to_list_elem(new_generate_EAR_target, data_dicts)
    data_dicts = tools.map_operation_to_list_elem(new_generate_ERR_target, data_dicts)


    # 最后，整理一下data_dicts中的数据，把不要的删掉
    # data_dict = tools.transpose_list_of_dict(data_dicts)
    # data_dicts = tools.transpose_dict_of_list(data_dict)

    cur_dataset = SimpleDataset(data_dict)
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
        argument_start_label, argument_end_label = torch.tensor(data_dict['argument_target_start'], dtype=torch.long), \
                                                   torch.tensor(data_dict['argument_target_end'], dtype=torch.long)
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


def new_val_dataset_factory(data_dicts: List[dict], dataset_type: str, stanza_nlp):
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
    # 去除content过长的sample
    data_dicts = tools.map_operation_to_list_elem(remove_illegal_length, data_dicts)
    # [content, events]

    # 按content-事件类型-触发词-进行划分
    data_dicts = tools.map_operation_to_list_elem(split_by_content_type_trigger, data_dicts)
    # [content, event_type, trigger_info, other_mentions]

    # 同时构造context，EAR问题与ERR问题。
    results = []
    for elem in data_dicts:
        results.extend(construct_EAR_ERR_context(elem, dataset_type, stanza_nlp))
    # [context, content, EAR_question, EAR_label, ERR_question, ERR_label, EAR_gt, ERR_gt]

    # tokenize
    data_dict = tools.transpose_list_of_dict(results)
    lst_tokenizer = tokenize_tools.bert_tokenizer()
    context_result = lst_tokenizer(data_dict['context'])
    context_result = tools.transpose_list_of_dict(context_result)
    context_result = tools.modify_key_of_dict(context_result, lambda x: x)
    EAR_result = lst_tokenizer(data_dict['EAR_question'])
    EAR_result = tools.transpose_list_of_dict(EAR_result)
    EAR_result = tools.modify_key_of_dict(EAR_result, lambda x: 'EAR_' + x)
    ERR_result = lst_tokenizer(data_dict['ERR_question'])
    ERR_result = tools.transpose_list_of_dict(ERR_result)
    ERR_result = tools.modify_key_of_dict(ERR_result, lambda x: 'ERR_' + x)
    data_dict.update(context_result)
    data_dict.update(EAR_result)
    data_dict.update(ERR_result)
    data_dicts = tools.transpose_dict_of_list(data_dict)

    cur_dataset = SimpleDataset(data_dict)
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
            'argument_gt': data_dict['EAR_gt'],
            'role_idx_gt': data_dict['ERR_gt']
        }
    val_dataloader = DataLoader(cur_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return val_dataloader


def dataset_factory(dataset_type: str):
    if dataset_type == 'FewFC':
        loaded = load_FewFC_ee('../../../data/NLP/EventExtraction/FewFC-main')
    elif dataset_type == 'Duee':
        loaded = load_Duee_ee_formated('../../../data/NLP/EventExtraction/duee')
    else:
        raise Exception(f'[dual_qa:dataset_factory]不存在{dataset_type}数据集！')

    stanza_nlp = stanza.Pipeline(lang='zh', processors='tokenize,ner')
    train_dataloader = new_train_dataset_factory(loaded['train'], dataset_type, stanza_nlp)
    val_dataloader = new_val_dataset_factory(loaded['val'], dataset_type)
    return train_dataloader, val_dataloader


model_registry = {
    "model": DualQA,
    'loss': DualQA_Loss,
    'Evaluator': DualQA_Evaluator,
    'train_val_data': dataset_factory
}