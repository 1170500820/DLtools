
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from type_def import *
from work.EE.PLMEE import EE_settings
import numpy as np
from itertools import chain
import pickle
import copy

from evaluate.evaluator import BaseEvaluator, tokenspans2events, F1Evaluator, CcksEvaluator
from work.EE.PLMEE.sentence_representation_layer import SentenceRepresentation, TriggeredSentenceRepresentation
from work.EE.PLMEE.trigger_extraction_model import TriggerExtractionLayer_woSyntactic
from work.EE.PLMEE.argument_extraction_model import ArgumentExtractionModel_woSyntactic
from work.EE.EE_utils import *
from utils import tools
from utils.data import SimpleDataset
from analysis.recorder import NaiveRecorder
from models.model_utils import get_init_params


class JointEE(nn.Module):
    def __init__(self,
                 plm_path=EE_settings.plm_path,
                 n_head=EE_settings.n_head,
                 d_head=EE_settings.d_head,
                 hidden_dropout_prob=0.3,
                 plm_lr=EE_settings.plm_lr,
                 others_lr=EE_settings.others_lr,
                 trigger_threshold=EE_settings.trigger_extraction_threshold,
                 argument_threshold=EE_settings.argument_extraction_threshold):
        super(JointEE, self).__init__()
        self.init_params = get_init_params(locals())  # 默认模型中包含这个东西。也许是个不好的设计
        # store init params
        self.plm_path = plm_path
        self.n_head = n_head
        self.d_head = d_head
        self.hidden_dropout_prob = hidden_dropout_prob
        self.plm_lr = plm_lr
        self.others_lr = others_lr
        self.trigger_threshold = trigger_threshold
        self.argument_threshold = argument_threshold

        # initiate network structures
        #   Sentence Representation
        self.sentence_representation = SentenceRepresentation(self.plm_path)
        self.hidden_size = self.sentence_representation.hidden_size
        #   Trigger Extraction
        self.tem = TriggerExtractionLayer_woSyntactic(
            num_heads=self.n_head,
            hidden_size=self.hidden_size,
            d_head=self.d_head,
            dropout_prob=self.hidden_dropout_prob)
        #   Triggered Sentence Representation
        self.trigger_sentence_representation = TriggeredSentenceRepresentation(self.hidden_size)
        #   Argument Extraction
        self.aem = ArgumentExtractionModel_woSyntactic(
            n_head=self.n_head,
            d_head=self.d_head,
            hidden_size=self.hidden_size,
            dropout_prob=self.hidden_dropout_prob)

        self.trigger_spanses = []
        self.argument_spanses = []

    def forward(self,
                sentences: List[str],
                event_types: Union[List[str], List[StrList]],
                triggers: SpanList = None,
                token2origin_lst: List[dict] = None):
        """
        jointEE对于train和eval模式有着不同的行为

        train：
            event_types为List[str]，与sentences中的句子一一对应，同时与triggers也一一对应
            先通过sentences与event_types获得句子的表示[BERT + Pooling + CLN] (bsz, seq_l, hidden)
            然后用TEM预测出trigger，再用triggers中的Span(gt)通过AEM预测出argument。这里的Span和TEM的预测结果完全无关
        eval：
            event_types为List[StrList]，是sentences中每一个对应的句子的所有包含的事件类型
            先通过sentences与event_types获得句子在每一个事件类型下的表示。
            然后用TEM对每一个类型预测出trigger，再用AEM对每一个trigger预测出argument。所有结果会被转化为源数据格式
            triggers参数完全不会被用到
        :param sentences:
        :param event_types:
        :param triggers:
        :return:
        """
        if self.training:
            self.trigger_spanses = []
            self.argument_spanses = []
            # print('-----------')
            # print(f'input sentences:{sentences}')
            # sentence_types StrList during training
            H_styp = self.sentence_representation(sentences, event_types)  # (bsz, max_seq_l, hidden)
            # print(f'sentence representation output: {H_styp.shape}')
            trigger_start, trigger_end = self.tem(H_styp)  # (bsz, max_seq_l, 1)
            # print(f'trigger extraction output: start:{trigger_start.shape}, end:{trigger_end.shape}')
            arg_H_styps, RPEs = self.trigger_sentence_representation(H_styp, triggers)
            # print(f'triger representation output: arg_H_styps:{arg_H_styps.shape}, RPE:{RPEs.shape}')
            # arg_H_styps: (bsz, max_seq_l, hidden)
            # PREs: (bsz, max_seq_l, 1)
            argument_start, argument_end = self.aem(arg_H_styps, RPEs)
            # print(f'argument extraction output: start:{argument_start.shape}, end:{argument_end.shape}')
            # both (bsz, max_seq_l, len(role_types))
            return {
                "trigger_start": trigger_start,  # (bsz, seq_l, 1)
                "trigger_end": trigger_end,  # (bsz. seq_l, 1)
                "argument_start": argument_start,  # (bsz, seq_l, role_cnt)
                "argument_end": argument_end,  # (bsz, seq_l, role_cnt)
            }
        else:  # eval mode
            # sentence_types: List[StrList] during evaluating
            if len(sentences) != 1:
                raise Exception('eval模式下一次只预测单个句子!')
            cur_spanses = []  # List[SpanList]
            arg_spanses = []  # List[List[List[SpanList]]]
            for elem_sentence_type in event_types[0]:
                # elem_sentence_type: str
                H_styp = self.sentence_representation(sentences, [elem_sentence_type])  # (bsz, max_seq_l, hidden)
                trigger_start_tensor, trigger_end_tensor = self.tem(H_styp)  # (bsz, max_seq_l, 1)
                trigger_start_tensor, trigger_end_tensor = trigger_start_tensor.squeeze(), trigger_end_tensor.squeeze()
                # (max_seq_l)

                trigger_start_result = (trigger_start_tensor > self.trigger_threshold).int().tolist()
                trigger_end_result = (trigger_end_tensor > self.trigger_threshold).int().tolist()
                cur_spans = tools.argument_span_determination(trigger_start_result, trigger_end_result, trigger_start_tensor, trigger_end_tensor)
                # cur_spans: SpanList, triggers extracted from current sentence

                arg_spans: List[List[SpanList]] = []
                for elem_trigger_span in cur_spans:
                    arg_H_styp, RPE = self.trigger_sentence_representation(H_styp, [elem_trigger_span])
                    argument_start_tensor, argument_end_tensor = self.aem(arg_H_styp, RPE)
                    # (1, max_seq_l, len(role_types))
                    argument_start_tensor = argument_start_tensor.squeeze().T  # (role cnt, max_seq_l)
                    argument_end_tensor = argument_end_tensor.squeeze().T  # (role cnt, max_seq_l)

                    argument_start_result = (argument_start_tensor > self.argument_threshold).int().tolist()
                    argument_end_result = (argument_end_tensor > self.argument_threshold).int().tolist()
                    argument_spans: List[SpanList] = []
                    for idx_role in range(len(EE_settings.role_types)):
                        cur_arg_spans: SpanList = tools.argument_span_determination(argument_start_result[idx_role], argument_end_result[idx_role], argument_start_tensor[idx_role].tolist(), argument_end_tensor[idx_role].tolist())
                        argument_spans.append(cur_arg_spans)
                    arg_spans.append(argument_spans)
                cur_spanses.append(cur_spans)
                arg_spanses.append(arg_spans)
            self.trigger_spanses.append(cur_spanses)
            self.argument_spanses.append(arg_spanses)
            result = tokenspans2events(event_types[0], cur_spanses, arg_spanses, EE_settings.role_types, sentences[0], token2origin_lst[0])
            return {"pred": result}

    def get_optimizers(self):
        repr_plm_params, repr_other_params = self.sentence_representation.PLM.parameters(), self.sentence_representation.CLN.parameters()
        trigger_repr_params = self.trigger_sentence_representation.parameters()
        aem_params = self.aem.parameters()
        tem_params = self.tem.parameters()
        plm_optimizer = AdamW(params=repr_plm_params, lr=self.plm_lr)
        others_optimizer = AdamW(params=chain(aem_params, tem_params, repr_other_params, trigger_repr_params), lr=self.others_lr)
        return [plm_optimizer, others_optimizer]


class JointEE_Loss(nn.Module):
    """
    需要加入的内容：
    1, focal loss
    2, weight
    3, RF·IEF weight
    """
    def __init__(self, lambd=0.6, alpha=0.3, gamma=2):
        """

        :param lambd: loss = lambd * trigger + (1 - lambd) * argument
        :param alpha: focal weight param
        :param gamma: focal weight param
        """
        super(JointEE_Loss, self).__init__()
        self.lambd = lambd
        self.focal = tools.FocalWeight(alpha, gamma)

        # record
        self.last_trigger_loss, self.last_argument_loss = 0., 0.
        self.last_trigger_preds, self.last_argument_preds = (0., 0.), (0., 0.)

    def forward(self,
                trigger_start: torch.Tensor,
                trigger_end: torch.Tensor,
                trigger_label_start: torch.Tensor,
                trigger_label_end: torch.Tensor,
                argument_start: torch.Tensor,
                argument_end: torch.Tensor,
                argument_label_start: torch.Tensor,
                argument_label_end: torch.Tensor):
        """

        :param trigger_start: (bsz, seq_l, 1)
        :param trigger_end: (bsz, seq_l, 1)
        :param trigger_label_start:
        :param trigger_label_end:
        :param argument_start: (bsz, seq_l, role_cnt)
        :param argument_end: (bsz, seq_l, role_cnt)
        :param argument_label_start:
        :param argument_label_end:
        :return: loss
        """
        if trigger_start.shape != trigger_label_start.shape:
            print('error')
            breakpoint()
        # calculate focal weight
        trigger_start_weight_focal = self.focal(trigger_start, trigger_label_start)
        trigger_end_weight_focal = self.focal(trigger_end, trigger_label_end)
        argument_start_weight_focal = self.focal(argument_start, argument_label_start)
        argument_end_weight_focal = self.focal(argument_end, argument_label_end)

        # combine weights
        trigger_start_weight = trigger_start_weight_focal
        trigger_end_weight = trigger_end_weight_focal
        argument_start_weight = argument_start_weight_focal
        argument_end_weight = argument_end_weight_focal

        trigger_start_loss = F.binary_cross_entropy(trigger_start, trigger_label_start, trigger_start_weight)
        trigger_end_loss = F.binary_cross_entropy(trigger_end, trigger_label_end, trigger_end_weight)
        trigger_loss = trigger_start_loss + trigger_end_loss

        argument_start_loss = F.binary_cross_entropy(argument_start, argument_label_start, argument_start_weight)
        argument_end_loss = F.binary_cross_entropy(argument_end, argument_label_end, argument_end_weight)
        argument_loss = argument_start_loss + argument_end_loss

        loss = self.lambd * trigger_loss + (1 - self.lambd) * argument_loss

        # for recorder
        self.last_trigger_loss = float(trigger_loss)
        self.last_argument_loss = float(argument_loss)
        trigger_start_np, trigger_end_np, argument_start_np, argument_end_np = \
            [trigger_start.cpu().detach().numpy(), trigger_end.cpu().detach().numpy(),
             argument_start.cpu().detach().numpy(), argument_end.cpu().detach().numpy()]
        self.last_trigger_preds = (trigger_start_np, trigger_end_np)
        self.last_argument_preds = (argument_start_np, argument_end_np)

        return loss


class JointEE_Evaluator(BaseEvaluator):
    """
    f1 and ccks
    """
    def __init__(self):
        super(JointEE_Evaluator, self).__init__()
        self.gt_lst = []
        self.pred_lst = []
        self.ccks_evaluator = CcksEvaluator()
        self.f1_evaluator = F1Evaluator()

    def eval_single(self, pred, gt):
        self.gt_lst.append(copy.deepcopy(gt))
        self.pred_lst.append(copy.deepcopy(pred))
        self.ccks_evaluator.eval_single(copy.deepcopy(gt), copy.deepcopy(pred))
        self.f1_evaluator.eval_single(copy.deepcopy(gt), copy.deepcopy(pred))

    def eval_step(self) -> Dict[str, Any]:
        ccks_result = self.ccks_evaluator.eval_step()
        f1_result = self.f1_evaluator.eval_step()
        ccks_result.update(f1_result)
        self.gt_lst = []
        self.pred_lst = []
        return ccks_result


class JointEE_Recorder(NaiveRecorder):
    """
    同时还会记录一些别的输出

    首先，为了了解loss的计算情况，我需要
    """
    def __init__(self, save_path):
        super(JointEE_Recorder, self).__init__(save_path='work/PLMEE')
        self.train_ckp_freq = 100

        self.record['loss']: List[Tuple[float, float, float]] = []  # total, trigger, argument
        self.loss_freq = 2
        self.record['trigger_preds']: List[Tuple[np.ndarray, np.ndarray]] = []  # trigger start and end
        self.record['argument_preds']: List[Tuple[np.ndarray, np.ndarray]] = []  # argument start and end
        self.record['trigger_spanses'] = []
        self.record['argument_spanses'] = []
        self.pred_freq = 20
        self.epoch, self.batch = 0, 0

    def train_checkin(self, step_info):
        self.epoch, self.batch = step_info

    def record_after_backward(self, loss_output: torch.Tensor, loss_func: nn.Module):
        """

        :param loss_output:
        :param loss_func:
        :return:
        """
        if self.batch % self.loss_freq == 0:
            self.record['loss'].append((float(loss_output), loss_func.last_trigger_loss, loss_func.last_argument_loss))

        if self.batch % self.pred_freq == 0:
            self.record['trigger_preds'].append(loss_func.last_trigger_preds)
            self.record['argument_preds'].append(loss_func.last_argument_preds)

    def record_after_evaluate(self, model: nn.Module, evaluator: BaseEvaluator, eval_result: Dict[str, Any]):
        super(JointEE_Recorder, self).record_after_evaluate(model, evaluator, eval_result)
        self.record['trigger_spanses'].append(model.trigger_spanses)
        self.record['argument_spanses'].append(model.argument_spanses)

    def train_checkpoint(self):
        if self.batch % self.train_ckp_freq == 0:
            pickle.dump(self.record, open(self.save_name, 'wb'))
"""
data process part
"""


def find_trigger_and_arguments_in_mentions(mentions: List[Dict[str, Any]]):
    """
    mentions中一定包含且仅包含一个role为trigger的mention，以及多个其他mention
    :param mentions:
    :return: Tuple[trigger, arguments list]
    """
    arguments = []
    triggers = []
    for elem_mention in mentions:
        if elem_mention['role'] == 'trigger':
            triggers.append(elem_mention)
        else:
            arguments.append(elem_mention)
    return triggers[0], arguments


def merge_content_and_events_to_origin(data_dict: Dict[str, Any]):
    content = data_dict['content']
    events = data_dict['events'].copy()
    origin = {'content': content, 'events': events}
    data_dict['origin'] = origin
    return [data_dict]


def wrapped_find_matches(data_dict: Dict[str, Any]):
    content = data_dict['content']
    tokens = data_dict['token']
    token2origin, origin2token = tools.find_matches(content, tokens)
    data_dict['token2origin'] = token2origin
    data_dict['origin2token'] = origin2token
    return [data_dict]


def merge_arguments_with_same_trigger(events: List[Dict[str, Any]]):
    """
    先转化为dict，再转化回来。
    event_dict: type->trigger->argument lst
    :param events:
    :return:
    """
    # convert to event dict
    event_dict = {}
    trigger_words = {}
    for elem_event in events:
        if elem_event['type'] not in event_dict:
            event_dict[elem_event['type']] = {}
        trigger = None
        arguments = []
        for elem_mention in elem_event['mentions']:
            if elem_mention['role'] == 'trigger':
                trigger = elem_mention
            else:
                arguments.append(elem_mention)
        trigger_words[tuple(trigger['span'])] = trigger['word']
        if trigger is None:
            raise Exception(f'no trigger found in current event:{elem_event}')
        if tuple(trigger['span']) not in event_dict[elem_event['type']]:
            event_dict[elem_event['type']][tuple(trigger['span'])] = arguments
        else:
            event_dict[elem_event['type']][tuple(trigger['span'])].extend(arguments)

    # convert to original dict
    merged_events = []
    for event_type_key, event_type_value in event_dict.items():
        cur_mentions = []
        for trigger_key, arguments_value in event_type_value.items():
            cur_mentions.append({
                "span": list(trigger_key),
                "word": trigger_words[trigger_key],
                "role": 'trigger'
            })
            cur_mentions.extend(arguments_value)
        merged_events.append({
            "type": event_type_key,
            "mentions": cur_mentions
        })
    return merged_events


def wrapped_merge_arguments_with_same_trigger(data_dict: Dict[str, Any]):
    data_dict['events'] = merge_arguments_with_same_trigger(data_dict['events'])
    return [data_dict]


def generate_label(data_dict: Dict[str, Any]):
    """
    tree format:
    event_type -> [triggers, trigger_label start and end, arguments] in which arguments -> [trigger, arguments, trigger label, argument label]
    {
    "event_type-1": {
                    "triggers": {
                                "trigger1 span tuple": trigger1,
                                'trigger2 span tuple': trigger2,
                                ...
                                },
                    "trigger_label_start": np.ndarray (seq_l),
                    'trigger_label_end': np.ndarray (seq_l),
                    "arguments": {
                                'trigger1 span tuple': {
                                                        'arguments': arguments,
                                                        'argument_label_start': np.ndarray (role_cnt, seq_l),
                                                        'argument_label_end': np.ndarray (role_cnt, seq_l)
                                                        }
                                }
                    }
    }
    :param data_dict:
    :return:
    """
    events = data_dict['events']
    token2origin, origin2token = data_dict['token2origin'], data_dict['origin2token']
    tokens = data_dict['token']
    token_length = len(tokens)
    token_length_without_placeholder = token_length - 2  # JointEE模型比较特殊，其输出是不包含<CLS>与<SEP>等占位符的

    # generate tree
    tree = {}
    for elem_event in events:
        event_type = elem_event['type']
        event_mentions = elem_event['mentions']
        trigger, arguments = find_trigger_and_arguments_in_mentions(event_mentions)
        if event_type not in tree:
            tree[event_type] = {
                "triggers": {tuple(trigger['span']): trigger},
                "trigger_label_start": None,
                "trigger_label_end": None,
                "arguments": {
                    tuple(trigger['span']): {
                        "arguments": arguments,
                        "argument_label_start": None,
                        "argument_label_end": None}
                }
            }
        else:
            tree[event_type]['triggers'][tuple(trigger['span'])] = trigger
            tree[event_type]['arguments'][tuple(trigger['span'])] = {
                "arguments": arguments,
                "argument_label_start": None,
                'argument_label_end': None
            }

    # calculate label
    #   calculate trigger labels: (seq_l)
    for event_type, event_type_content in tree.items():
        triggers = event_type_content['triggers']  # key: trigger tuple span, value: trigger
        trigger_span_tuples = list(triggers.keys())
        trigger_label_start = np.zeros(token_length_without_placeholder)
        trigger_label_end = np.zeros(token_length_without_placeholder)
        for elem_span in trigger_span_tuples:
            # token_start与token_end分别为对应到tokenizer生成的token中的字词的位置
            # 在origin2token dict转化后减1，是因为去除了<CLS>
            # token_end在内部要额外减1，是因为标注数据中的end是不包含word本身的，也就是外边界。而模型需要word的最后一个token的index作为
            # 边界，也就是需要内边界。
            token_start, token_end = origin2token[elem_span[0]] - 1, origin2token[elem_span[1] - 1] - 1
            trigger_label_start[token_start] = 1
            trigger_label_end[token_end] = 1
        tree[event_type]['trigger_label_start'] = trigger_label_start
        tree[event_type]['trigger_label_end'] = trigger_label_end
    #   calculate argument labels: (role_cnt, seq_l)
    for event_type, event_type_content in tree.items():
        arguments_of_cur_event = event_type_content['arguments']
        for trigger_span_tuple, argument_infos in arguments_of_cur_event.items():
            cur_arguments = argument_infos['arguments']
            argument_label_start = np.zeros((len(EE_settings.role_types), token_length_without_placeholder))
            argument_label_end = np.zeros((len(EE_settings.role_types), token_length_without_placeholder))
            for elem_arg in cur_arguments:
                cur_role_type, cur_span = elem_arg['role'], elem_arg['span']
                cur_role_index = EE_settings.role_index[cur_role_type]
                cur_arg_start, cur_arg_end = origin2token[cur_span[0]] - 1, origin2token[cur_span[1] - 1] - 1
                argument_label_start[cur_role_index][cur_arg_start] = 1
                argument_label_end[cur_role_index][cur_arg_end] = 1
            argument_infos['argument_label_start'] = argument_label_start
            argument_infos['argument_label_end'] = argument_label_end
    data_dict['structure_tree'] = tree
    return [data_dict]


def split_into_train_data(data_dict: Dict[str, Any]):
    """
    将包含了structure_tree的，处理好的事件抽取数据，划分为多条训练数据
    :param data_dict:
    :return:
    """
    train_data_lst = []

    # -- sentence
    sentence = data_dict['content']
    origin2token = data_dict['origin2token']
    for event_type, event_type_content in data_dict['structure_tree'].items():

        # -- trigger
        origin_span_list = list(event_type_content['triggers'].keys())
        token_span_list: List[Tuple[int, int]] = []
        for elem_origin_span in origin_span_list:
            token_start = origin2token[elem_origin_span[0]] - 1
            token_end = origin2token[elem_origin_span[1] - 1] - 1
            token_span_list.append((token_start, token_end))

        # -- trigger labels
        trigger_label_start = event_type_content['trigger_label_start']
        trigger_label_end = event_type_content['trigger_label_end']

        # -- argument_labels
        argument_label_start_lst, argument_label_end_lst = [], []
        for elem_origin_span in origin_span_list:
            arg_info = event_type_content['arguments'][elem_origin_span]
            arg_start, arg_end = arg_info['argument_label_start'], arg_info['argument_label_end']
            argument_label_start_lst.append(arg_start)
            argument_label_end_lst.append(arg_end)

        for (one_trigger, one_arg_label_start, one_arg_label_end) in zip(token_span_list, argument_label_start_lst, argument_label_end_lst):
            cur_train_data_dict = {
                "sentence": sentence,
                'event_type': event_type,
                "trigger_label_start": trigger_label_start,
                "trigger_label_end": trigger_label_end,
                "trigger": one_trigger,
                "argument_label_start": one_arg_label_start,
                "argument_label_end": one_arg_label_end
            }
            train_data_lst.append(cur_train_data_dict)
    return train_data_lst


def train_dataset_factory(train_filename: str, bsz: int = 4):
    json_lines = tools.read_json_lines(train_filename)
    json_dict_lines = [{'content': x['content'], 'events': x['events']} for x in json_lines]
    # [content, events]

    # remove illegal characters
    data_dicts = tools.map_operation_to_list_elem(remove_function, json_dict_lines)
    # get [content, events]

    # remove data with illegal length
    data_dicts = tools.map_operation_to_list_elem(remove_illegal_length, data_dicts)

    # tokenize
    data_dict = tools.transpose_list_of_dict(data_dicts)
    lst_tokenizer = tools.bert_tokenizer()
    result = lst_tokenizer(data_dict['content'])
    result = tools.transpose_list_of_dict(result)
    data_dict.update(result)
    data_dicts = tools.transpose_dict_of_list(data_dict)
    # [content, events, input_ids, token_type_ids, attention_mask, token]

    # calculate match
    data_dicts = tools.map_operation_to_list_elem(wrapped_find_matches, data_dicts)
    # [content, events, input_ids, token_type_ids, attention_mask, token, token2origin, origin2token]

    # merge arguments with same trigger
    data_dicts = tools.map_operation_to_list_elem(wrapped_merge_arguments_with_same_trigger, data_dicts)

    # generating trigger and argument labels
    data_dicts = tools.map_operation_to_list_elem(generate_label, data_dicts)

    # split into train data
    train_data = tools.map_operation_to_list_elem(split_into_train_data, data_dicts)
    train_dataset = SimpleDataset(train_data)

    def collate_fn(lst):
        """

        :param lst: List[Dict], dict contains:
            - sentence  str
            - event_type  str
            - trigger  Span
            - trigger_label_start  np.ndarray (seq_l)
            - trigger_label_end  np.ndarray (seq_l)
            - argument_label_start  np.array (role_cnt, seq_l)
            - argument_label_end  np.ndarray (role_cnt, seq_l)
        :return:
        """
        dict_of_data = tools.transpose_list_of_dict(lst)
        arg_start_lst = list(map(lambda x: x.T, dict_of_data['argument_label_start']))
        arg_end_lst = list(map(lambda x: x.T, dict_of_data['argument_label_end']))  # list of (seq_l, role_cnt)

        batchified_arg_label_start = torch.tensor(tools.batchify_ndarray(arg_start_lst), dtype=torch.float)  # (bsz, max_seq_l, role_cnt)
        batchified_arg_label_end = torch.tensor(tools.batchify_ndarray(arg_end_lst), dtype=torch.float)  # (bsz, max_seq_l, role_cnt)
        batchified_trig_label_start = torch.tensor(tools.batchify_ndarray(dict_of_data['trigger_label_start']), dtype=torch.float).unsqueeze(dim=-1)  # (bsz, seq_l, 1)
        batchified_trig_label_end = torch.tensor(tools.batchify_ndarray(dict_of_data['trigger_label_end']), dtype=torch.float).unsqueeze(dim=-1)  # (bsz, seq_l, 1)

        model_input = {
            "sentences": dict_of_data['sentence'],
            "event_types": dict_of_data['event_type'],
            "triggers": dict_of_data['trigger']
        }
        loss_input = {
            "trigger_label_start": batchified_trig_label_start,
            "trigger_label_end": batchified_trig_label_end,
            "argument_label_start": batchified_arg_label_start,
            "argument_label_end": batchified_arg_label_end
        }

        # convert to cuda
        loss_input = tools.convert_dict_to_cuda(loss_input)

        return model_input, loss_input

    train_dataloader = DataLoader(train_dataset, batch_size=bsz, shuffle=True, collate_fn=collate_fn)
    return train_dataloader


def val_dataset_factory(val_filename: str):
    json_lines = tools.read_json_lines(val_filename)
    json_dict_lines = [{'content': x['content'], 'events': x['events']} for x in json_lines]
    # [content, events]

    # remove illegal characters
    data_dicts = tools.map_operation_to_list_elem(remove_function, json_dict_lines)

    # extract event types
    data_dicts = tools.map_operation_to_list_elem(extract_event_types, data_dicts)

    # merge to origin format
    data_dicts = tools.map_operation_to_list_elem(merge_content_and_events_to_origin, data_dicts)

    # tokenize
    data_dict = tools.transpose_list_of_dict(data_dicts)
    lst_tokenizer = tools.bert_tokenizer()
    result = lst_tokenizer(data_dict['content'])
    result = tools.transpose_list_of_dict(result)
    data_dict.update(result)
    data_dicts = tools.transpose_dict_of_list(data_dict)
    # [content, events, input_ids, token_type_ids, attention_mask, token]

    # calculate match
    data_dicts = tools.map_operation_to_list_elem(wrapped_find_matches, data_dicts)

    val_dataset = SimpleDataset(data_dicts)

    def collate_fn(lst):
        """

        :param lst: List[dict]
            keys:
            - content
            - events
            - event_types
            - origin
        :return:
        """
        data_lst = tools.transpose_list_of_dict(lst)
        return {
            "sentences": data_lst['content'],
            "event_types": data_lst['event_types'],
            "token2origin_lst": data_lst['token2origin']
               }, {
            "gt": data_lst['origin'][0]
        }

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return val_dataloader


class UseModel:
    def __init__(self, state_dict_path: str, init_params_path):
        init_params = pickle.load(open(init_params_path, 'rb'))
        self.model = JointEE(**init_params)
        self.model.load_state_dict(torch.load(open(state_dict_path, 'rb'), map_location=torch.device('cpu')))
        self.model.eval()

        self.tokenizer = tools.bert_tokenizer()

    def __call__(self, sentence: str, event_types: List[str]) -> dict:
        # remove illegal chars
        sentence = sentence.replace(' ', '_')

        # tokenize
        tokenized = self.tokenizer([sentence])[0]
        input_ids = torch.tensor(tokenized['input_ids']).unsqueeze(dim=0)  # (1, seq_l)
        token_type_ids = torch.tensor(tokenized['token_type_ids']).unsqueeze(dim=0)  # (1, seq_l)
        attention_mask = torch.tensor(tokenized['attention_mask']).unsqueeze(dim=0)  # (1, seq_l)

        # calculate match
        tokens = tokenized['token']
        token2origin, origin2token = tools.find_matches(sentence, tokens)
        result = self.model([sentence], [event_types], triggers=None, token2origin_lst=[token2origin])
        return result['pred']


model_registry = {
    "model": JointEE,
    "loss": JointEE_Loss,
    "evaluator": JointEE_Evaluator,
    "train_data": train_dataset_factory,
    "val_data": val_dataset_factory,
    "args": [
        {'name': "--train_file", 'dest': 'train_filename', 'type': str, 'help': '训练/测试数据文件的路径'},
        {'name': "--val_file", 'dest': 'val_filename', 'type': str, 'help': '训练/测试数据文件的路径'},
    ],
    'recorder': JointEE_Recorder
}


if __name__ == '__main__':
    filepath = '../../../data/NLP/EventExtraction/FewFC-main/train.json'

    # train_loader = train_dataset_factory(filepath, 4)
    val_loader = val_dataset_factory(filepath)
    #
    train, val = [], []
    for i in range(5):
        # train.append(next(iter(train_loader)))
        val.append(next(iter(val_loader)))
