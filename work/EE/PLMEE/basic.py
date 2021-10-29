import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluate.evaluator import *
from process.typed_processor_utils import *
from utils.batch_tool import find_matches
from torch.utils.data import IterableDataset, Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from torch.optim import AdamW
import copy
from itertools import chain
import numpy as np
import random
from utils.batch_tool import argument_span_determination
from utils.tools import *

filepath = '../../../data/NLP/EventExtraction/FewFC-main/train.json'
event_types = ['投资', '收购', '判决', '起诉', '中标', '股份股权转让', '担保', '签署合同', '质押', '减持']
event_types_idx = {x: i for (i, x) in enumerate(event_types)}
role_types = [
    'sub-org',
    'money',
    'amount',
    'obj-per',
    'obj-org',
    'sub',
    'share-per',
    'way',
    'title',
    'target-company',
    'collateral',
    'institution',
    'sub-per',
    'share-org',
    'number',
    'date',
    'obj',
    'proportion']
role_types_idx = {x: i for (i, x) in enumerate(role_types)}





def convert_to_tensor_brute_force(dict_of_things: Dict[str, Any]):
    """
    把一个不知道包含什么东西的dict，所有内容全部尝试转化为torch.Tensor。
    如
    :param dict_of_things:
    :return:
    """


class HybridDataset(IterableDataset):
    def __init__(self,
                 trigger_data: List[Dict[str, Any]],
                 trigger_label_start: List[np.ndarray],
                 trigger_label_end: List[np.ndarray],
                 argument_data: List[Dict[str, Any]],
                 argument_label_start: List[np.ndarray],
                 argument_label_end: List[np.ndarray],
                 bsz: int = 4,
                 shuffle: bool = True):
        """

        :param trigger_data: {input_ids: , token_type_ids: , attention_mask}
        :param trigger_label_start: ndarray (seq_l, event_cnt)
        :param trigger_label_end: ndarray (seq_l, event_cnt)
        :param argument_data: {arg_input_ids: , arg_token_type_ids: , arg_attention_mask}
        :param argument_label_start:
        :param argument_label_end
        :param bsz:
        """
        self.trigger_data = trigger_data
        self.trigger_label_start = trigger_label_start
        self.trigger_label_end = trigger_label_end
        self.argument_data = argument_data
        self.argument_label_start = argument_label_start
        self.argument_label_end = argument_label_end
        self.bsz = bsz
        self.shuffle = shuffle

        self.trigger_index = list(range(len(self.trigger_data)))
        self.argument_index = list(range(len(self.argument_data)))

    def __iter__(self):
        return self

    def __next__(self):
        trigger_index_sampled = random.sample(self.trigger_index, self.bsz)
        trigger_data_sampled = list(map(lambda x: self.trigger_data[x], trigger_index_sampled))
        trigger_label_start_sampled = list(map(lambda x: self.trigger_label_start[x], trigger_index_sampled))
        trigger_label_end_sampled = list(map(lambda x: self.trigger_label_end[x], trigger_index_sampled))
        batchified_trigger_data = batchify_dict(trigger_data_sampled)
        batchified_trigger_label_start = batchify_ndarray(trigger_label_start_sampled)
        batchified_trigger_label_end = batchify_ndarray(trigger_label_end_sampled)

        argument_index_sampled = random.sample(self.argument_index, self.bsz)
        argument_data_sampled = list(map(lambda x: self.argument_data[x], argument_index_sampled))
        argument_label_start_sampled = list(map(lambda x: self.argument_label_start[x], argument_index_sampled))
        argument_label_end_sampled = list(map(lambda x: self.argument_label_end[x], argument_index_sampled))
        batchified_argument_data = batchify_dict(argument_data_sampled)
        batchified_argument_label_start = batchify_ndarray(argument_label_start_sampled)
        batchified_argument_label_end = batchify_ndarray(argument_label_end_sampled)

        # convert to torch.Tensor
        for key, value in batchified_trigger_data.items():
            batchified_trigger_data[key] = torch.tensor(value)
        for key, value in batchified_argument_data.items():
            batchified_argument_data[key] = torch.tensor(value)
        batchified_trigger_label_start = torch.tensor(batchified_trigger_label_start)
        batchified_trigger_label_end = torch.tensor(batchified_trigger_label_end)
        batchified_argument_label_start = torch.tensor(batchified_argument_label_start)
        batchified_argument_label_end = torch.tensor(batchified_argument_label_end)

        # convert to cuda
        batchified_trigger_label_start = batchified_trigger_label_start.cuda()
        batchified_trigger_label_end = batchified_trigger_label_end.cuda()
        batchified_argument_label_start = batchified_argument_label_start.cuda()
        batchified_argument_label_end = batchified_argument_label_end.cuda()
        convert_dict_to_cuda(batchified_trigger_data)
        convert_dict_to_cuda(batchified_argument_data)

        return {
            "trigger_data": batchified_trigger_data,  # {input_ids: Tensor, token_type_ids: Tensor, attention_mask: Tensor}
            "trigger_label_start": batchified_trigger_label_start,
            "trigger_label_end": batchified_trigger_label_end,
            "argument_data": batchified_argument_data,
            "argument_label_start": batchified_argument_label_start,
            "argument_label_end": batchified_argument_label_end
        }


class EventExtractionEvaluateDataset(Dataset):
    """
    一个专门用于事件抽取模型的evaluate的dataset
    todo pytorch的DataSet是不是只需要实现__len__和__iter__?
    """
    def __init__(self, data: List[Dict[str, torch.Tensor]], origin_data: List[Dict[str, Any]]):
        """

        :param data: 能够直接输入模型的data
            {input_ids: , token_type_ids: , attention_mask: }
        :param origin_data: 源格式的数据
        """
        if len(data) != len(origin_data):
            raise Exception('[EventExtractionEvaluateDataset.__init__]data与origin的长度必须相同！二者应当一一对应')
        self.data = data
        self.origin_data = origin_data

        self.start = 0
        self.end = len(self.data)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.start == self.end:
            self.start = 1
        return self.data[self.start], self.origin_data[self.start]


def divide_by_type(json_dict: Dict[str, Any]):
    result_lst = []
    for elem in json_dict['events']:
        new_dict = json_dict.copy()
        new_dict['type'] = elem['type']
        new_dict['mentions'] = elem['mentions']
        new_dict.pop('events')
        result_lst.append(new_dict)
    return result_lst


def divide_by_type_and_trigger(data_dict: Dict[str, Any]):
    type_trigger_dict: Dict[Tuple[str, Tuple[int, int]], Any] = {}
    for elem in data_dict['events']:
        cur_type = elem['type']
        cur_mentions = elem['mentions']
        other_mentions = []
        trigger = {}
        for elem_mention in cur_mentions:
            if elem_mention['role'] == 'trigger':
                trigger.update(elem_mention)
            else:
                other_mentions.append(elem_mention)
        cur_trigger_span = tuple(trigger['span'])
        if (cur_type, cur_trigger_span) not in type_trigger_dict:
            type_trigger_dict[(cur_type, cur_trigger_span)] = [other_mentions, trigger]
    data_dict.pop('events')
    result_lst = []
    for key, value in type_trigger_dict.items():
        new_dict = {
            'trigger': value[1],
            "arguments": value[0],
            'type': key[0]
        }
        new_dict.update(copy.deepcopy(data_dict))
        result_lst.append(new_dict)
    return result_lst


def remove_illegal_chars(content_dict: Dict[str, Any]):
    content_dict['content'] = content_dict['content'].replace(' ', '_')
    return [content_dict]


class Matcher(TypedDataProcessor):
    def __init__(self):
        super(Matcher, self).__init__(['List[str]|token', 'str|content'], 'Tuple[dict,dict]|match', keep=True)

    def process(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        token = data_dict['List[str]|token']
        content = data_dict['str|content']
        result = find_matches(content, token)
        return {
            self.get_output_keys()[0]: result
        }


def clear_key(data_dict: Dict[str, Any]):
    remove_type(data_dict)
    return [data_dict]


def calculate_match(data_dict: Dict[str, Any]):
    content = data_dict['content']
    tokens = data_dict['token']
    match = find_matches(content, tokens)
    data_dict['token2origin'], data_dict['origin2token'] = match
    return [data_dict]


def get_arg_and_trigger_position(data_dict: Dict[str, Any]):
    trigger = data_dict['trigger']
    arguments = data_dict['arguments']
    token2origin, origin2token = data_dict['token2origin'], data_dict['origin2token']

    trig_span = trigger['span']
    trigger['span'] = (origin2token[trig_span[0]], origin2token[trig_span[1] - 1])

    for elem_arg in arguments:
        arg_span = elem_arg['span']
        try:
            elem_arg['span'] = (origin2token[arg_span[0]], origin2token[arg_span[1] - 1])
        except:
            print('')
    return [data_dict]


def generate_trigger_label(data_dict: Dict[str, Any]):
    input_len = len(data_dict['input_ids'])
    trigger_span = data_dict['trigger']['span']
    trigger_type = data_dict['type']
    type_idx = event_types_idx[trigger_type]
    trigger_label_start = np.zeros((input_len, len(event_types)))
    trigger_label_end = np.zeros((input_len, len(event_types)))
    trigger_label_start[trigger_span[0]][type_idx] = 1
    trigger_label_end[trigger_span[1]][type_idx] = 1
    data_dict['trigger_label_start'] = trigger_label_start
    data_dict['trigger_label_end'] = trigger_label_end
    return [data_dict]


def generate_argument_label(data_dict: Dict[str, Any]):
    input_len = len(data_dict['input_ids'])
    argument_label_start = np.zeros((input_len, len(role_types)))
    argument_label_end = np.zeros((input_len, len(role_types)))
    for elem_arg in data_dict['arguments']:
        arg_span = elem_arg['span']
        arg_type = elem_arg['role']
        arg_type_idx = role_types_idx[arg_type]
        argument_label_start[arg_span[0]][arg_type_idx] = 1
        argument_label_end[arg_span[1]][arg_type_idx] = 1
    data_dict['argument_label_start'] = argument_label_start
    data_dict['argument_label_end'] = argument_label_end
    return [data_dict]


def generate_argument_input(data_dict: Dict[str, Any]):
    origin_tti = data_dict['token_type_ids']
    trigger_info = data_dict['trigger']
    trigger_span = trigger_info['span']
    new_tti = origin_tti.copy()
    for idx in range(trigger_span[0], trigger_span[1] + 1):
        new_tti[idx] = 1
    data_dict['arg_token_type_ids'] = new_tti
    data_dict['arg_input_ids'] = data_dict['input_ids'].copy()
    data_dict['arg_attention_mask'] = data_dict['attention_mask'].copy()
    return [data_dict]


get_event_extraction_essential = \
    Json_Reader() + \
    ListModifier(remove_illegal_chars) + ListOfDictTranspose() + ReleaseDict() + \
    KeyFilter(['content', 'id', 'events']) * (KeyFilter(['content']) + BERT_Tokenizer() + ListOfDictTranspose() + ReleaseDict())\
    + SqueezeDict(['content', 'id', 'events', 'input_ids', 'token_type_ids', 'attention_mask', 'token']) + \
    DictOfListTranspose() + \
    ListModifier(clear_key) + \
    ListModifier(calculate_match) + ListModifier(divide_by_type_and_trigger) + \
    ListModifier(get_arg_and_trigger_position)

generate_arg_input = ListModifier(generate_argument_input)
generate_label = ListModifier(generate_trigger_label) + ListModifier(generate_argument_label)


def insert_dict(o, i):
    o['tokenized'] = i
    return o


# expect input
# trigger extraction part
# {
#     "te_input_ids": <input_ids (bsz, max_seq_l)>,
#     "te_token_type_ids": <token_type_ids (bsz, max_seq_l)>
#     "te_attention_mask": <attention_mask (bsz, max_seq_l)>
# }, {
#     "te_label": <(bsz, max_seq_l, event_type) 其中触发词的start与end位置对应type标为1>
# }
#
# argument extraction part
# {
#     "ae_input_ids": <input_ids (bsz, max_seq_l)>,
#     "ae_token_type_ids": <token_type_ids (bsz, max_seq_l), 触发词的start与end位置为1>
#     "ae_attention_mask": <attention_mask (bsz, max_seq_l)>
# }, {
#     "ae_label": <(bsz, max_seq_l, argument_type)>
# }

# create a random combination dataset

class PLMEE_Hybrid(nn.Module):
    def __init__(self, plm_lr=3e-5, linear_lr=5e-3):
        super(PLMEE_Hybrid, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.hidden = self.bert.config.hidden_size
        self.trigger_linear_start = nn.Linear(self.hidden, len(event_types))
        self.trigger_linear_end = nn.Linear(self.hidden, len(event_types))
        self.argument_linear_start = nn.Linear(self.hidden, len(role_types))
        self.argument_linear_end = nn.Linear(self.hidden, len(role_types))
        self.plm_lr = plm_lr
        self.linear_lr = linear_lr

        # predict
        self.trigger_threshold = 0.5
        self.argument_threshold = 0.5

    def init_weights(self):
        # todo 能否模块化
        torch.nn.init.xavier_uniform_(self.trigger_linear_start.weight)
        self.trigger_linear_start.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.argument_linear_start.weight)
        self.argument_linear_start.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.trigger_linear_end.weight)
        self.trigger_linear_end.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.argument_linear_end.weight)
        self.argument_linear_end.bias.data.fill_(0)

    def forward(self,
                trigger_input: Dict[str, torch.Tensor] = None,
                argument_input: Dict[str, torch.Tensor] = None):
        """
        双模式
            train:
                使用混合模型。输入trigger_input用于预测触发词，输入触发词标记的argument_input用于预测论元
            eval:
                只使用trigger_input的输入。首先预测触发词。预测出触发词后直接用预测得到的触发词预测论元。
        :param trigger_input:
            {input_ids: Tensor (bsz, seq_l),
             token_type_ids: Tensor (bsz, seq_l),
             attention_mask: Tensor (bsz, seq_l)}
        :param argument_input:
            {arg_input_ids: Tensor (bsz, seq_l),
             token_type_ids: Tensor (bsz, seq_l),
             attention_mask: Tensor (bsz, seq_l)}
        :return:
        """
        if self.training:  # train mode
            trigger_output = self.bert(**trigger_input)  # (bsz, seq_l, hidden)
            trigger_output_start = self.trigger_linear_start(trigger_output)  # (bsz, seq_l, event_type)
            trigger_output_end = self.trigger_linear_end(trigger_output)  # (bsz, seq_l, event_type)
            trigger_output_start, trigger_output_end = F.sigmoid(trigger_output_start), F.sigmoid(trigger_output_end)
            # both (bsz, seq_l, event_type * 2)

            argument_output = self.bert(input_ids=argument_input['arg_input_ids'],
                                        token_type_ids=argument_input['arg_token_type_ids'],
                                        attention_mask=argument_input['arg_attention_mask'])
            # (bsz, seq_l, hidden)
            argument_output_start = self.argument_linear_start(argument_output)  # (bsz, seq_l, role_type)
            argument_output_end = self.argument_linear_end(argument_output)  # (bsz, seq_l, role_type)

            argument_output_start, argument_output_end = \
                F.sigmoid(argument_output_start), F.sigmoid(argument_output_end)  # (bsz, seq_l, role_type)

            return {
                "trigger_output_start": trigger_output_start,  # (bsz, seq_l, event_type)
                "trigger_output_end": trigger_output_end,  # (bsz, seq_l, event_type)
                "argument_output_start": argument_output_start,  # (bsz, seq_l, role_type)
                "argument_output_end": argument_output_end  # (bsz, seq_l, role_type)
            }
        else:  # evaluate mode
            # assert bsz = 1
            trigger_output = self.bert(**trigger_input)  # (bsz, seq_l, hidden)
            trigger_output_start = self.trigger_linear_start(trigger_output)  # (bsz, seq_l, event_type)
            trigger_output_end = self.trigger_linear_end(trigger_output)  # (bsz, seq_l, event_type)
            trigger_output_start, trigger_output_end = F.sigmoid(trigger_output_start), F.sigmoid(trigger_output_end)  # (bsz, seq_l, event_type)
            trigger_probs_start, trigger_probs_end = np.array(trigger_output_start.squeeze().T), np.array(trigger_output_end.squeeze().T)  # (event_type, seq_l)
            trigger_digits_start, trigger_digits_end = (trigger_probs_start > self.trigger_threshold).astype(int), (trigger_probs_end > self.trigger_threshold).astype(int)  # (event_type, seq_l)

            trigger_spans: List[List[Span]] = []
            for i in range(trigger_probs_start.shape[0]):  # event_types
                spans_of_cur_type = argument_span_determination(trigger_digits_start[i], trigger_digits_end[i], trigger_probs_start[i], trigger_probs_end[i])
                trigger_spans.append(spans_of_cur_type)


            argument_spans: List[List[List[List[Span]]]] = []
            for elem_etype in trigger_spans:
                # elem_etype: List[Span]
                argument_spans.append([])
                for elem_span in elem_etype:
                    argument_spans[-1].append([])
                    # elem_span: Span
                    token_type_ids = trigger_input['token_type_ids']
                    token_type_ids[elem_span[0]] = 1
                    token_type_ids[elem_span[1]] = 1
                    trigger_input['token_type_ids'] = token_type_ids
                    argument_output = self.bert(**trigger_input)  # (bsz, seq_l, hidden)
                    argument_output_start = self.argument_linear_start(argument_output)  # (bsz, seq_l, role_type)
                    argument_output_end = self.argument_linear_end(argument_output)  # (bsz, seq_l, role_type)
                    argument_output_start, argument_output_end = F.sigmoid(argument_output_start), F.sigmoid(argument_output_end)  # (bsz, seq_l, role_type)
                    argument_probs_start, argument_probs_end = np.array(argument_output_start.squeeze().T), np.array(argument_output_end.squeeze().T)  # (role_type, seq_l)
                    argument_digits_start, argument_digits_end = (argument_probs_start > self.argument_threshold).astype(int), (argument_probs_end > self.argument_threshold).astype(int)  # (role_type, seq_l)

                    for i in range(argument_probs_start.shape[0]):  # role_types
                        spans_of_cur_type = argument_span_determination(argument_digits_start[i], argument_digits_end[i], argument_probs_start[i], argument_probs_end[i])
                        argument_spans[-1][-1].append(spans_of_cur_type)

            predict_result = spans2events(event_types, trigger_spans, argument_spans, role_types)
            return {
                "pred": predict_result
            }

    def get_optimizers(self):
        bert_param = self.bert.parameters()
        trigger_linear_param = self.trigger_linear.parameters()
        argument_linear_param = self.argument_linear.parameters()
        plm_optim = AdamW(bert_param, lr=self.plm_lr)
        linear_optim = AdamW(chain(trigger_linear_param, argument_linear_param), lr=self.linear_lr)
        return [plm_optim, linear_optim]


class PLMEE_Hybrid_Loss(nn.Module):
    def __init__(self, lam=0.6):
        super(PLMEE_Hybrid_Loss, self).__init__()
        self.Lambda = lam

    def forward(self,
                trigger_output_start: torch.Tensor,
                trigger_output_end: torch.Tensor,
                trigger_label_start: torch.Tensor,
                trigger_label_end: torch.Tensor,
                argument_output_start: torch.Tensor,
                argument_output_end: torch.Tensor,
                argument_label_start: torch.Tensor,
                argument_label_end: torch.Tensor):
        trigger_loss_start = F.binary_cross_entropy(trigger_output_start, trigger_label_start)
        trigger_loss_end = F.binary_cross_entropy(trigger_output_end, trigger_label_end)
        trigger_loss = trigger_loss_start + trigger_loss_end

        argument_loss_start = F.binary_cross_entropy(argument_output_start, argument_label_start)
        argument_loss_end = F.binary_cross_entropy(argument_output_end, argument_label_end)
        argument_loss = argument_loss_start + argument_loss_end

        combined_loss = self.Lambda * trigger_loss + (1 - self.Lambda) * argument_loss
        return combined_loss


class PLMEE_Hybrid_Evaluator(BaseEvaluator):
    def __init__(self):
        super(PLMEE_Hybrid_Evaluator, self).__init__()
        self.ccks = CcksEvaluator()
        self.f1 = F1Evaluator()

    def eval_single(self, pred: SentenceWithEvent, gt: SentenceWithEvent):
        """

        :param trigger_output:
        :param trigger_gt: {type: , span: }
        :param argument_output:
        :param argument_gt: List[{role: , span: }]
        :return:
        """
        self.ccks.eval_single(pred, gt)
        self.f1.eval_single(pred, gt)

    def eval_step(self) -> str:
        ccks_result = self.ccks.eval_step()
        f1_result = self.f1.eval_step()
        return f'CCKS Result:\n' + ccks_result + '\nF1 Result:\n' + f1_result


def train_dataset_factory(filepath, bsz: int = 8, shuffle=True):
    combined = get_event_extraction_essential + generate_label + generate_arg_input + ListOfDictTranspose()
    result = list(combined(filepath).values())[0]
    trigger_data = {
            "input_ids": result['input_ids'],
            "token_type_ids": result['token_type_ids'],
            "attention_mask": result['attention_mask']
        }
    trigger_data_lst = transpose_dict_of_list(trigger_data)
    argument_data = {
            "arg_input_ids": result['arg_input_ids'],
            "arg_token_type_ids": result['arg_token_type_ids'],
            "arg_attention_mask": result['arg_attention_mask']
        }
    argument_data_lst = transpose_dict_of_list(argument_data)
    dataset = HybridDataset(
        trigger_data=trigger_data_lst,
        trigger_label_start=result['trigger_label_start'],
        trigger_label_end=result['trigger_label_end'],
        argument_data=argument_data_lst,
        argument_label_start=result['argument_label_start'],
        argument_label_end=result['argument_label_end'],
        bsz=bsz
    )

    def collate_fn_do_nothing(lst):
        # assert len(lst) == 1
        content = lst[0]
        model_data, loss_data = split_dict(content, ['trigger_data', 'argument_data'])
        return model_data, loss_data
    dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle, collate_fn=collate_fn_do_nothing)
    return dataloader


def val_dataset_factory(filepath):
    origin = read_json_lines(filepath)
    combined = get_event_extraction_essential + ListOfDictTranspose()
    result = list(combined(filepath).values())[0]
    trigger_data = {
            "input_ids": result['input_ids'],
            "token_type_ids": result['token_type_ids'],
            "attention_mask": result['attention_mask']
        }
    trigger_data_lst = transpose_dict_of_list(trigger_data)

    dataset = EventExtractionEvaluateDataset(trigger_data_lst, origin)
    def collate_fn_do_nothing(lst):
        input_data, origin_data = lst[0]
        return {"trigger_input": input_data}, {"gt": origin_data}
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_do_nothing)
    return dataloader


model_registry = {
    "model": PLMEE_Hybrid,
    "args": [
        {'name': "filepath", 'dest': 'filepath', 'type': str, 'help': '训练/测试数据文件的路径'},
        {'name': "plm_lr", 'dest': 'plm_lr', 'type': float, 'help': '预训练模型的学习率'},
        {'name': "linear_lr", 'dest': 'plm_lr', 'type': float, 'help': '线性的学习率'},
    ],
    "loss": PLMEE_Hybrid_Evaluator,
    "evaluator": PLMEE_Hybrid_Evaluator,
    "train_data": train_dataset_factory,
    "val_data": val_dataset_factory
}

