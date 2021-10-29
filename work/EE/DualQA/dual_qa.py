import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from utils import tools
from utils.data import SimpleDataset
from work.EE.EE_utils import *
from work.EE import EE_settings
import numpy as np
from torch.utils.data import DataLoader
from evaluate.evaluator import BaseEvaluator, F1Evaluator, CcksEvaluator


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
    def __init__(self, hidden: int):
        super(ArgumentClassifier, self).__init__()
        self.hidden = hidden
        self.start_classifier = nn.Linear(hidden, 1, bias=False)
        self.end_classifier = nn.Linear(hidden, 1, bias=False)

    def forward(self, G: torch.Tensor):
        """

        :param G: (bsz, C, hidden)
        :return:
        """
        start_digit = self.start_classifier(G)  # (bsz, C, 1)
        end_digit = self.end_classifier(G)  # (bsz, C, 1)

        start_prob = F.sigmoid(start_digit)  # (bsz, C, 1)
        end_prob = F.sigmoid(end_digit)  # (bsz, C, 1)

        start_prob, end_prob = start_prob.squeeze(), end_prob.squeeze()  # both (bsz, C) or (C) if bsz == 1
        if len(start_prob.shape) == 1:
            start_prob = start_prob.unsqueeze(dim=0)
            end_prob = end_prob.unsqueeze(dim=0)
        return start_prob, end_prob  # both (bsz, C)


class RoleClassifier(nn.Module):
    """
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
        原论文的实现有些疑问，所以我先按自己的想法实现一个
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

        self.role_classifier = RoleClassifier()

    def forward(self,
                input_ids: torch.Tensor,
                token_type_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                argument_input_ids: torch.Tensor = None,
                argument_token_type_ids: torch.Tensor = None,
                argument_attention_mask: torch.Tensor = None,
                role_input_ids: torch.Tensor = None,
                role_token_type_ids: torch.Tensor = None,
                role_attention_mask: torch.Tensor = None):
        """
        尽量遵循保持输入为纯tensor的原则，整个DualQA的输入分别是C，QA，QR的input_ids，token_type_ids，attention_mask

        在训练时，C，QA，QR均会作为输入，QA与QR都不能为None
        在预测时，如果QA部分为None而QR部分不为None，那么就只会进行ERR的预测，输出的EAR部分就为None
        如果QA不为None而QR为None，则对应的，进行EAR的预测，ERR输出None
        QA和QR不能同时为None
        :param input_ids: (bsz, C)
        :param token_type_ids:
        :param attention_mask:
        :param argument_input_ids: (bsz, QA), 一定与qa_token_type_ids和qa_attention_mask同时为/不为None
        :param argument_token_type_ids:
        :param argument_attention_mask:
        :param role_input_ids: (bsz, QR), 一定与qr_token_type_ids和qr_attention_mask同时为/不为None
        :param role_token_type_ids:
        :param role_attention_mask:
        :return:
        """
        if argument_input_ids is None and role_input_ids is None:
            raise Exception('[DualQA]QA与QR输入不能同时为None！')
        H = self.shared_encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)
        # (bsz, C, hidden)
        if argument_input_ids is not None:
            UA = self.shared_encoder(
                input_ids=argument_input_ids,
                token_type_ids=argument_token_type_ids,
                attention_mask=argument_attention_mask)
            # (bsz, QA, hidden)
            H_A_hat, UA_hat = self.flow_attention(H, UA)  # both (bsz, C, hidden)
            GA = self.shared_projection(H, H_A_hat, UA_hat)  # (bsz, C, hidden)
            start_probs, end_probs = self.arg_classifier(GA)  # both (bsz, C, hidden)
        if role_input_ids is not None:
            UR = self.shared_encoder(
                input_ids=role_input_ids,
                token_type_ids=role_token_type_ids,
                attention_mask=role_attention_mask)
            # (QR, hidden)
            H_R_hat, UR_hat = self.flow_attention(H, UR)  # both (bsz, C, hidden)
            GR = self.shared_projection(H, H_R_hat, UR_hat)  # (bsz, C, hidden)
            role_pred = self.role_classifier(GR)  # (bsz, role_cnt)


class DualQA_Loss(nn.Module):
    pass


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


def generate_EAR_questions_and_labels(data_dict: Dict[str, Any]):
    event_type = data_dict['event_type']
    other_mentions = data_dict['other_mentions']

    questions = []
    labels = []
    for elem_mention in other_mentions:
        role = elem_mention['role']
        word = elem_mention['word']
        question = f'在事件{event_type}中作为{role}角色的是哪个词？'
        label = elem_mention['span']
        questions.append(question)
        labels.append(label)

    data_dict.update({
        "argument_questions": questions,
        "argument_labels": labels
    })
    return [data_dict]


def generate_ERR_questions_and_labels(data_dict: Dict[str, Any]):
    event_type = data_dict['event_type']
    other_mentions = data_dict['other_mentions']

    questions = []
    labels = []
    for elem_mention in other_mentions:
        role = elem_mention['role']
        word = elem_mention['word']
        question = f'词语{word}在事件{event_type}中作为什么角色？'
        label = role
        questions.append(question)
        labels.append(label)

    data_dict.update({
        "role_questions": questions,
        "role_labels": labels
    })
    return [data_dict]


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

    for (elem_mention, elem_arg_q, elem_arg_label, elem_role_q, elem_role_label) in zip(other_mentions, arg_qs, arg_labels, role_qs, role_labels):
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


"""

"""
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
    data_dicts = tools.map_operation_to_list_elem(generate_EAR_questions_and_labels, data_dicts)
    data_dicts = tools.map_operation_to_list_elem(generate_ERR_questions_and_labels, data_dicts)
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
