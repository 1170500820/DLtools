import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertModel
from itertools import chain
import pickle

from type_def import *
from utils import tools, batch_tool, tokenize_tools
from utils.tokenize_tools import OffsetMapping
from utils.data import SimpleDataset
from evaluate.evaluator import BaseEvaluator, EE_F1Evaluator, SentenceWithEvent, Events, Event, Mention, Mentions
from models.model_utils import get_init_params
from analysis.recorder import NaiveRecorder

from work.EE import EE_settings, EE_utils


class PLMEE(nn.Module):
    def __init__(self,
                 plm_lr: float = EE_settings.plm_lr,
                 linear_lr: float = EE_settings.others_lr,
                 plm_path: str = EE_settings.default_plm_path,
                 event_types: str = EE_settings.event_types_full,
                 role_types: str = EE_settings.role_types,
                 threshold: float = EE_settings.event_detection_threshold):
        super(PLMEE, self).__init__()
        self.init_params = get_init_params(locals())

        self.plm_lr = plm_lr
        self.linear_lr = linear_lr
        self.plm_path = plm_path
        self.event_types = event_types
        self.role_types = role_types

        self.threshold = threshold

        self.bert = BertModel.from_pretrained(self.plm_path)
        self.hidden = self.bert.config.hidden_size
        self.trigger_linear_start = nn.Linear(self.hidden, len(self.event_types))
        self.trigger_linear_end = nn.Linear(self.hidden, len(self.event_types))
        self.argument_linear_start = nn.Linear(self.hidden, len(self.role_types))
        self.argument_linear_end = nn.Linear(self.hidden, len(self.role_types))

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.trigger_linear_start.weight)
        self.trigger_linear_start.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.argument_linear_start.weight)
        self.argument_linear_start.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.trigger_linear_end.weight)
        self.trigger_linear_end.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.argument_linear_end.weight)
        self.argument_linear_end.bias.data.fill_(0)

    def get_optimizers(self):
        self.plm_params = self.bert.parameters()
        self.trigger_start_params = self.trigger_linear_start.parameters()
        self.trigger_end_params = self.trigger_linear_end.parameters()
        self.argument_start_params = self.argument_linear_start.parameters()
        self.argument_end_params = self.argument_linear_end.parameters()

        plm_optimizer = AdamW(params=self.plm_params, lr=self.plm_lr)
        linear_optimizer = AdamW(params=chain(self.trigger_start_params, self.trigger_end_params, self.argument_start_params, self.argument_end_params), lr=self.linear_lr)
        return [plm_optimizer, linear_optimizer]

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor, trigger_gt: list = None):
        """

        :param input_ids: (bsz, seq_l)
        :param token_type_ids: (bsz, seq_l)
        :param attention_mask: (bsz, seq_l)
        :param trigger_gt:
            len(trigger_gt) == bsz
            len(trigger_gt[*]) == 2, start and end
            如果为None，则根据trigger_output提取trigger信息。如果不为None，则使用该list中的下标信息对token_type_ids进行标记
        :return:
        """
        bsz, seq_l = input_ids.shape
        trigger_output = self.predict_trigger(input_ids, token_type_ids, attention_mask)  # dict
        trigger_start, trigger_end = trigger_output['trigger_start'], trigger_output['trigger_end']
        # both (bsz, seq_l, event_type_cnt)
        if trigger_gt is None:
            # 这种情况下，无反向传播
            trigger_start = trigger_start.permute([0, 2, 1])
            trigger_end = trigger_end.permute([0, 2, 1])
            # both (bsz, event_type_cnt, seq_l)

            trigger_start_digit = (trigger_start > self.threshold).int().tolist()
            trigger_end_digit = (trigger_end > self.threshold).int().tolist()
            # both (bsz, event_type_cnt, seq_l)
            predicts = []  # (batch, event_type, [trigger_span, arguments])

            for i_batch, (e_start_tensor, e_start_digit, e_end_tensor, e_end_digit) in enumerate(list(zip(trigger_start, trigger_start_digit, trigger_end, trigger_end_digit))):
                # 对每个batch考虑
                # all (event_type_cnt, seq_l)
                predicts.append([])
                # predicts[-1]存放当前句子的信息
                cur_input_ids = input_ids[i_batch].unsqueeze(0)  # (seq_l)
                cur_attention_mask = input_ids[i_batch].unsqueeze(0)  # (seq_l)
                for i_etype, (e_start_type_tensor, e_start_type_digit, e_end_type_tensor, e_end_type_digit) in enumerate(list(zip(e_start_tensor, e_start_digit, e_end_tensor, e_end_digit))):
                    # 对当前句子的每个事件类型考虑
                    # all (seq_l)
                    predicts[-1].append([])
                    # predicts[-1][-1]存放当前事件类型的信息
                    # 该句子当前类型的触发词都存放在span当中
                    spans = tools.argument_span_determination(e_start_type_digit, e_end_type_digit, e_start_type_tensor, e_end_type_tensor)
                    # 存放该句子在每个触发词下所对应的论元的列表
                    argument_preds: List[Tuple[int, List[Tuple[int, int]]]] = []  # list element: (role_type, role span list)
                    # len(spans) == len(argument_preds)
                    for e_span in spans:
                        # argument_preds[-1]就存放论元的列表
                        cur_token_type_ids = token_type_ids[i_batch].clone().detach().unsqueeze(0)  # (1, seq_l)
                        cur_token_type_ids[0][e_span[0]] = 1
                        cur_token_type_ids[0][e_span[1]] = 1
                        argument_output = self.predict_argument(cur_input_ids, cur_token_type_ids, cur_attention_mask)
                        a_start, a_end = argument_output['argument_start'].squeeze(0).T, argument_output['argument_end'].squeeze(0).T
                        # both (role_type_cnt, seq_l)
                        for i_rtype, (e_start, e_end) in enumerate(list(zip(a_start, a_end))):
                            # both (seq_l)
                            a_start_digit = (e_start > self.threshold).int().tolist()
                            a_end_digit = (e_end > self.threshold).int().tolist()
                            cur_span = tools.argument_span_determination(a_start_digit, a_end_digit, a_start, a_end)
                            if len(cur_span) != 0:
                                argument_preds.append((i_rtype, cur_span))
                        if len(argument_preds) != 0:
                            predicts[-1][-1].append([e_span, argument_preds])

                    # predicts[-1][-1].append(spans)
                    # predicts[-1][-1].append(argument_preds)
                    # predicts[-1][-1].append((spans, argument_preds))
            return {
                'predicts': predicts
            }
        else:
            token_label = torch.zeros(token_type_ids.shape, dtype=torch.long)
            if token_type_ids.is_cuda:
                token_label = token_label.cuda()
            for idx, elem in enumerate(trigger_gt):
                token_label[idx][elem[0]] = 1
                token_label[idx][elem[1]] = 1
            token_type_ids = token_type_ids + token_label
            argument_output = self.predict_argument(input_ids, token_type_ids, attention_mask)
            a_start, a_end = argument_output['argument_start'], argument_output['argument_end']
            # both (bsz, seq_l, role_type_cnt)
            return {
                'trigger_start_pred': trigger_start,
                'trigger_end_pred': trigger_end,
                'argument_start_pred': a_start,
                'argument_end_pred': a_end,
                'attention_mask': attention_mask
            }

    def predict_trigger(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor):
        output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        result = output[0]  # (bsz, seq_l, hidden)

        trigger_start = torch.sigmoid(self.trigger_linear_start(result))  # (bsz, seq_l, event_type_cnt)
        trigger_end = torch.sigmoid(self.trigger_linear_end(result))  # (bsz, seq_l, event_type_cnt)
        return {
            'trigger_start': trigger_start,
            'trigger_end': trigger_end
        }

    def predict_argument(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor):
        output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        result = output[0]  # (bsz, seq_l, hidden)

        argument_start = torch.sigmoid(self.argument_linear_start(result))  # (bsz, seq_l, event_type_cnt)
        argument_end = torch.sigmoid(self.argument_linear_end(result))  # (bsz, seq_l, event_type_cnt)
        return {
            'argument_start': argument_start,
            'argument_end': argument_end
        }


class PLMEE_Loss(nn.Module):
    def forward(self,
                trigger_start_pred: torch.Tensor,
                trigger_end_pred: torch.Tensor,
                argument_start_pred: torch.Tensor,
                argument_end_pred: torch.Tensor,
                attention_mask: torch.Tensor,
                trigger_start_label: torch.Tensor,
                trigger_end_label: torch.Tensor,
                argument_start_label: torch.Tensor,
                argument_end_label: torch.Tensor):
        """

        :param trigger_start_pred: (bsz, seq_l, event_type_cnt)
        :param trigger_end_pred:
        :param argument_start_pred:
        :param argument_end_pred:
        :param attention_mask: (bsz, seq_l)
        :param trigger_start_label:
        :param trigger_end_label:
        :param argument_start_label:
        :param argument_end_label:
        :return:
        """
        # mask = (1 - attention_mask).bool()
        #
        # # 计算mask
        # trigger_start_pred = trigger_start_pred.masked_fill(mask, value=torch.tensor(0))
        # trigger_end_pred = trigger_end_pred.masked_fill(mask, value=torch.tensor(0))
        # argument_start_pred = argument_start_pred.masked_fill(mask, value=torch.tensor(0))
        # argument_end_pred = argument_end_pred.masked_fill(mask, value=torch.tensor(0))

        attention_mask = attention_mask.unsqueeze(-1)  # (bsz, seq_l, 1)
        # 计算loss
        trigger_start_loss = F.binary_cross_entropy(trigger_start_pred, trigger_start_label, reduction='none')
        trigger_end_loss = F.binary_cross_entropy(trigger_end_pred, trigger_end_label, reduction='none')
        argument_start_loss = F.binary_cross_entropy(argument_start_pred, argument_start_label, reduction='none')
        argument_end_loss = F.binary_cross_entropy(argument_end_pred, argument_end_label, reduction='none')

        ts_loss = torch.sum(trigger_start_loss * attention_mask) / torch.sum(attention_mask)
        te_loss = torch.sum(trigger_end_loss * attention_mask) / torch.sum(attention_mask)
        as_loss = torch.sum(argument_start_loss * attention_mask) / torch.sum(attention_mask)
        ae_loss = torch.sum(argument_end_loss * attention_mask) / torch.sum(attention_mask)

        trigger_loss = ts_loss + te_loss
        argument_loss = as_loss + ae_loss

        loss = trigger_loss + argument_loss

        return loss


def convert_predicts_to_SentenceWithEvent(predicts: list, sentence: str, offset_mapping: OffsetMapping, event_types: list = EE_settings.event_types_full, role_types: list = EE_settings.role_types):
    """

    :param predicts: (batch, event_type, [trigger_span, arguments])
        其中trigger span长度为2
        arguments为(trigger_span role_type)
        在eval的条件下，默认batch_size=1
    :return:
    """
    if len(predicts) != 1:
        raise Exception('[convert_predicts_to_SentenceWithEvent]batch_size不为1！')
    predict = predicts[0]
    events = []
    for i_etype, e_etype in enumerate(predict):
        event_type_word = event_types[i_etype]
        mentions = []
        for i_ta, e_ta in enumerate(e_etype):
            trigger_span, arguments = e_ta
            trigger_word = tokenize_tools.tokenSpan_to_word(sentence, trigger_span, offset_mapping)
            trigger_char_span = list(tokenize_tools.tokenSpan_to_charSpan(trigger_span, offset_mapping))
            for elem_arg in arguments:
                rtype_idx, arg_spans = elem_arg
                role_type_word = role_types[rtype_idx]
                for elem_role_span in arg_spans:
                    role_word = tokenize_tools.tokenSpan_to_word(sentence, elem_role_span, offset_mapping)
                    role_charspan = tokenize_tools.tokenSpan_to_charSpan(elem_role_span, offset_mapping)
                    mentions.append({
                        'word': role_word,
                        'span': role_charspan,
                        'role': role_type_word
                    })
            mentions.append({
                'word': trigger_word,
                'span': trigger_char_span,
                'role': 'trigger'
            })

        events.append({
            'type': event_type_word,
            'mentions': mentions
        })

    sentencenwithevents = {
        'id': '',
        'content': sentence,
        'events': events
    }
    return sentencenwithevents


class PLMEE_Evaluator(BaseEvaluator):
    def __init__(self, event_types: list = EE_settings.event_types_full, role_types: list = EE_settings.role_types):
        super(PLMEE_Evaluator, self).__init__()
        self.event_types = event_types
        self.role_types = role_types
        self.f1evaluator = EE_F1Evaluator()
        self.pred_lst, self.gt_lst = [], []

    def eval_single(self, predicts: list, gt: SentenceWithEvent, sentence, offset_mapping):
        preds = convert_predicts_to_SentenceWithEvent(predicts, sentence, offset_mapping, self.event_types, self.role_types)
        self.f1evaluator.eval_single(gt, preds)
        self.pred_lst.append({
            'preds': predicts,
            'sentence': sentence
        })
        self.gt_lst.append(gt)


    def eval_step(self) -> Dict[str, Any]:
        result = self.f1evaluator.eval_step()
        self.pred_lst = []
        self.gt_lst = []
        return result


def train_dataset_factory(data_dicts: List[dict], bsz: int = EE_settings.default_bsz, shuffle: bool = EE_settings.default_shuffle, dataset_type: str = 'FewFC'):
    if dataset_type == 'FewFC':
        event_types = EE_settings.event_types_full
        role_types = EE_settings.role_types
    else:
        raise Exception(f'{dataset_type}数据集不存在！')
    train_dataset = SimpleDataset(data_dicts)


    def collate_fn(lst):
        """
        data_dict包含
        - content
        - input_ids
        - token_type_ids
        - attention_mask
        - trigger_gt
        - trigger_start_label
        - trigger_end_label
        - argument_start_label
        - argument_end_label
        - offset_mapping

        模型需要输入:
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

        # generate trigger gt
        trigger_gt = data_dict['trigger_gt']

        # generate trigger labels
        trigger_start_label_info, trigger_end_label_info = data_dict['trigger_start_label'], data_dict['trigger_end_label']
        trigger_start_label, trigger_end_label = torch.zeros((bsz, seq_l, len(event_types)), dtype=torch.float), torch.zeros((bsz, seq_l, len(event_types)), dtype=torch.float)
        for idx, e_trigger_label in enumerate(trigger_start_label_info):
            for i_etype, i_start in enumerate(e_trigger_label):
                i_cur_type, i_cur_index = i_start
                trigger_start_label[idx][i_cur_index][i_cur_type] = 1
        for idx, e_trigger_label in enumerate(trigger_end_label_info):
            for i_etype, i_end in enumerate(e_trigger_label):
                i_cur_type, i_cur_index = i_end
                trigger_end_label[idx][i_cur_index][i_cur_type] = 1

        # generate argument labels
        arg_start_info, arg_end_info = data_dict['argument_start_label'], data_dict['argument_end_label']
        argument_start_label, argument_end_label = torch.zeros((bsz, seq_l, len(role_types)), dtype=torch.float), torch.zeros((bsz, seq_l, len(role_types)), dtype=torch.float)
        for idx, e_arg_label in enumerate(arg_start_info):
            for i_rtype, i_start in enumerate(e_arg_label):
                i_cur_type, i_cur_index = i_start
                argument_start_label[idx][i_cur_index][i_cur_type] = 1
        for idx, e_arg_label in enumerate(arg_end_info):
            for i_rtype, i_end in enumerate(e_arg_label):
                i_cur_type, i_cur_index = i_end
                argument_end_label[idx][i_cur_index][i_cur_type] = 1

        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'trigger_gt': trigger_gt
               }, {
            'trigger_start_label': trigger_start_label,
            'trigger_end_label': trigger_end_label,
            'argument_start_label': argument_start_label,
            'argument_end_label': argument_end_label
        }

    train_dataloader = DataLoader(train_dataset, batch_size=bsz, shuffle=shuffle, collate_fn=collate_fn)

    return train_dataloader


def dev_dataset_factory(data_dicts: List[dict], dataset_type: str, valid_type: str = 'total'):
    """

    :param data_dicts:
    :param dataset_type:
    :param valid_type: trigger, argument, total
    :return:
    """
    valid_dataset = SimpleDataset(data_dicts)

    def trigger_collate_fn(lst):
        data_dict = tools.transpose_list_of_dict(lst)
        bsz = len(lst)

        # generate basic input
        input_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['input_ids']), dtype=torch.long)
        token_type_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['token_type_ids']), dtype=torch.long)
        attention_mask = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['attention_mask']), dtype=torch.long)

    def arg_collate_fn(lst):
        data_dict = tools.transpose_list_of_dict(lst)
        bsz = len(lst)

        # generate basic input
        input_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['input_ids']), dtype=torch.long)
        token_type_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['token_type_ids']), dtype=torch.long)
        attention_mask = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['attention_mask']), dtype=torch.long)

    def total_collate_fn(lst):
        data_dict = tools.transpose_list_of_dict(lst)
        bsz = len(lst)

        # generate basic input
        input_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['input_ids']), dtype=torch.long)
        token_type_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['token_type_ids']), dtype=torch.long)
        attention_mask = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['attention_mask']), dtype=torch.long)

        content = data_dict['content'][0]
        events = data_dict['events'][0]
        offset_mapping = data_dict['offset_mapping'][0]
        sentence_with_event = {
            'content': content,
            'events': events,
            'id': ''
        }
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
               }, {
            'gt': sentence_with_event,
            'sentence': content,
            'offset_mapping': offset_mapping
        }

    if valid_type == 'trigger':
        collate_fn = trigger_collate_fn
    elif valid_type == 'argument':
        collate_fn = arg_collate_fn
    elif valid_type == 'total':
        collate_fn = total_collate_fn
    else:
        raise Exception(f'{valid_type}为错误的评价方法')

    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return valid_dataloader


def dataset_factory(train_file: str, valid_file: str, bsz: int = EE_settings.default_bsz, shuffle: bool = EE_settings.default_shuffle, dataset_type: str = 'FewFC'):
    train_data_dicts = pickle.load(open(train_file, 'rb'))
    valid_data_dicts = pickle.load(open(valid_file, 'rb'))

    train_dataloader = train_dataset_factory(train_data_dicts, bsz=bsz, shuffle=shuffle, dataset_type=dataset_type)
    valid_dataloader = dev_dataset_factory(valid_data_dicts, dataset_type=dataset_type)

    return train_dataloader, valid_dataloader



model_registry = {
    'model': PLMEE,
    'loss': PLMEE_Loss,
    'evaluator': PLMEE_Evaluator,
    'train_val_data': dataset_factory,
    'recorder': NaiveRecorder
}


if __name__ == '__main__':
    train_file = 'temp_data/train.FewFC.labeled.pk'
    valid_file = 'temp_data/valid.FewFC.gt.pk'
    bsz = 4
    shuffle = False
    dataset_type = 'FewFC'

    train_data_dicts = pickle.load(open(train_file, 'rb'))
    valid_data_dicts = pickle.load(open(valid_file, 'rb'))

    train_dataloader = train_dataset_factory(train_data_dicts, bsz=bsz, shuffle=shuffle, dataset_type=dataset_type)
    valid_dataloader = dev_dataset_factory(valid_data_dicts, dataset_type=dataset_type)

    limit = 5
    train_data, valid_data = [], []
    for idx, (train_sample, valid_sample) in enumerate(list(zip(train_dataloader, valid_dataloader))):
        train_data.append(train_sample)
        valid_data.append(valid_sample)

