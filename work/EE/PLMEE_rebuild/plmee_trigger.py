import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertModel, AutoTokenizer
from itertools import chain
import pickle
from tqdm import tqdm

from type_def import *
from utils import tools, batch_tool, tokenize_tools
from utils.tokenize_tools import OffsetMapping
from utils.data import SimpleDataset
from evaluate.evaluator import BaseEvaluator, EE_F1Evaluator, SentenceWithEvent, Events, Event, Mention, Mentions, F1_Evaluator
from models.model_utils import get_init_params
from analysis.recorder import NaiveRecorder
from work.EE.EE_utils import load_jsonl, dump_jsonl

from work.EE import EE_settings, EE_utils


class PLMEE_Trigger(nn.Module):
    def __init__(self, plm_lr: float = EE_settings.plm_lr, linear_lr: float = EE_settings.others_lr,
                 plm_path: str = EE_settings.default_plm_path, dataset_type: str = 'FewFC'):
        super(PLMEE_Trigger, self).__init__()
        self.init_params = get_init_params(locals())

        self.plm_lr = plm_lr
        self.linear_lr = linear_lr
        self.plm_path = plm_path

        self.dataset_type = dataset_type
        if self.dataset_type == 'FewFC':
            event_types = EE_settings.event_types_full
        elif self.dataset_type == 'Duee':
            event_types = EE_settings.duee_event_types
        else:
            raise Exception(f'{dataset_type}数据集不存在！')
        self.event_types = event_types

        self.bert = BertModel.from_pretrained(self.plm_path)
        self.hidden = self.bert.config.hidden_size

        # 分别用于预测触发词的开始与结束位置
        self.start_classifiers = nn.ModuleList(nn.Linear(self.hidden, 2) for i in range(len(self.event_types)))
        self.end_classifiers = nn.ModuleList(nn.Linear(self.hidden, 2) for i in range(len(self.event_types)))

        self.init_weights()

    def init_weights(self):
        for elem_cls in self.start_classifiers:
            torch.nn.init.xavier_uniform_(elem_cls.weight)
            elem_cls.bias.data.fill_(0)
        for elem_cls in self.end_classifiers:
            torch.nn.init.xavier_uniform_(elem_cls.weight)
            elem_cls.bias.data.fill_(0)

    def get_optimizers(self):
        linear_start_params = self.start_classifiers.parameters()
        linear_end_params = self.end_classifiers.parameters()
        bert_params = self.bert.parameters()
        linear_optimizer = AdamW(params=chain(linear_start_params, linear_end_params), lr=self.linear_lr)
        bert_optimizer = AdamW(params=bert_params, lr=self.plm_lr)
        return [linear_optimizer, bert_optimizer]

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor):
        """

        :param input_ids: (bsz, seq_l)
        :param token_type_ids: (bsz, seq_l)
        :param attention_mask: (bsz, seq_l)
        :return:
        """
        output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        embed = output[0]  # (bsz, seq_l, hidden)

        pred_starts = []
        pred_ends = []
        for cls in self.start_classifiers:
            cls_output = cls(embed)  # (bsz, seq_l, 2)
            cls_output = F.softmax(cls_output, dim=2)
            pred_starts.append(cls_output)  # (bsz, seq_l, 2)
        for cls in self.end_classifiers:
            cls_output = cls(embed)  # (bsz, seq_l, 2)
            cls_output = F.softmax(cls_output, dim=2)
            pred_ends.append(cls_output)  # (bsz, seq_l, 2)

        return {
            'pred_starts': pred_starts,
            'pred_ends': pred_ends,
            'mask': attention_mask
        }


class PLMEE_Trigger_T(nn.Module):
    def __init__(self, plm_lr: float = EE_settings.plm_lr, linear_lr: float = EE_settings.others_lr,
                 plm_path: str = EE_settings.default_plm_path, dataset_type: str = 'FewFC'):
        super(PLMEE_Trigger_T, self).__init__()
        self.init_params = get_init_params(locals())

        self.plm_lr = plm_lr
        self.linear_lr = linear_lr
        self.plm_path = plm_path

        self.dataset_type = dataset_type
        if self.dataset_type == 'FewFC':
            event_types = EE_settings.event_types_full
        elif self.dataset_type == 'Duee':
            event_types = EE_settings.duee_event_types
        else:
            raise Exception(f'{dataset_type}数据集不存在！')
        self.event_types = event_types

        self.bert = BertModel.from_pretrained(self.plm_path)
        self.hidden = self.bert.config.hidden_size

        # 分别用于预测触发词的开始与结束位置
        self.start_classifier = nn.Linear(self.hidden, len(self.event_types))
        self.end_classifier = nn.Linear(self.hidden, len(self.event_types))
        # self.start_classifiers = nn.ModuleList(nn.Linear(self.hidden, 2) for i in range(len(self.event_types)))
        # self.end_classifiers = nn.ModuleList(nn.Linear(self.hidden, 2) for i in range(len(self.event_types)))

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.start_classifier.weight)
        self.start_classifier.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.end_classifier.weight)
        self.end_classifier.bias.data.fill_(0)

    def get_optimizers(self):
        plm_params = self.bert.parameters()
        linear_start_params = self.start_classifier.parameters()
        linear_end_params = self.end_classifier.parameters()

        plm_optimizer = AdamW(params=plm_params, lr=self.plm_lr)
        linear_optimizer = AdamW(params=chain(linear_start_params, linear_end_params), lr=self.linear_lr)

        return [plm_optimizer, linear_optimizer]

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor):
        """

        :param input_ids: (bsz, seq_l)
        :param token_type_ids: (bsz, seq_l)
        :param attention_mask: (bsz, seq_l)
        :return:
        """
        bsz, seq_l = input_ids.shape

        output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        embed = output[0]  # (bsz, seq_l, hidden)

        trigger_start = torch.sigmoid(self.start_classifier(embed))  # (bsz, seq_l, event_type_cnt)
        trigger_end = torch.sigmoid(self.end_classifier(embed))  # (bsz, seq_l, event_type_cnt)

        return {
            'trigger_start': trigger_start,
            'trigger_end': trigger_end,
            'attention_mask': attention_mask
        }


class PLMEE_Trigger_Loss(nn.Module):
    def forward(
            self,
            pred_starts: List[torch.Tensor],
            pred_ends: List[torch.Tensor],
            mask: torch.Tensor,
            start_labels: List[torch.Tensor],
            end_labels: List[torch.Tensor]):
        """

        :param pred_starts:
        :param pred_ends: both list of (bsz, seq_l, 2)

        :param start_labels:
        :param end_labels: both list of (bsz, seq_l, 2)

        :param mask: (bsz, seq_l) batch中每一个句子的mask
        :return:
        """
        bsz = pred_starts[0].shape[0]
        event_loss = []

        for (pstart, pend, lstart, lend) in zip(pred_starts, pred_ends, start_labels, end_labels):
            start_losses, end_losses = [], []
            lstart, lend = lstart.cuda(), lend.cuda()
            for i_batch in range(bsz):
                start_loss = F.cross_entropy(pstart[i_batch], lstart[i_batch], reduction='none')  # (bsz, seq_l)
                end_loss = F.cross_entropy(pend[i_batch], lend[i_batch], reduction='none')  # (bsz, seq_l)
                start_loss = torch.sum(start_loss * mask[i_batch]) / torch.sum(mask[i_batch])
                end_loss = torch.sum(end_loss * mask[i_batch]) / torch.sum(mask[i_batch])
                start_losses.append(start_loss)
                end_losses.append(end_loss)
            start_loss = sum(start_losses)
            end_loss = sum(end_losses)
            combined_loss = start_loss + end_loss
            event_loss.append(combined_loss)

        loss = sum(event_loss) / len(event_loss)
        return loss


class PLMEE_Trigger_T_Loss(nn.Module):
    def forward(self,
                trigger_start: torch.Tensor,
                trigger_end: torch.Tensor,
                attention_mask: torch.Tensor,
                trigger_start_label: torch.Tensor,
                trigger_end_label: torch.Tensor,):
        """

        :param trigger_start_pred: (bsz, seq_l, event_type_cnt)
        :param trigger_end_pred:
        :param attention_mask: (bsz, seq_l)
        :param trigger_start_label:
        :param trigger_end_label:
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
        trigger_start_loss = F.binary_cross_entropy(trigger_start, trigger_start_label, reduction='none')
        trigger_end_loss = F.binary_cross_entropy(trigger_end, trigger_end_label, reduction='none')

        ts_loss = torch.sum(trigger_start_loss * attention_mask) / torch.sum(attention_mask)
        te_loss = torch.sum(trigger_end_loss * attention_mask) / torch.sum(attention_mask)

        trigger_loss = ts_loss + te_loss

        loss = trigger_loss

        return loss


def convert_output_to_evaluate_format(pred_starts: List[torch.Tensor], pred_ends: List[torch.Tensor], mask: torch.Tensor, sentence: str, offset_mapping: OffsetMapping, event_types: list = EE_settings.event_types_full):
    """
    将模型的输出转换为evaluator能够直接判定的格式

    :param pred_starts:
    :param pred_ends: list of (bsz, seq_l, 2) bsz==1
    :param mask: 不需要
    :param sentence: 原句
    :param offset_mapping: FastTokenizer的输出
    :return:
    """
    result = {}
    detailed = {}
    for i_type, (e_start, e_end) in enumerate(list(zip(pred_starts, pred_ends))):
        cur_type = event_types[i_type]
        e_start = e_start.squeeze(dim=0)  # (seq_l, 2)
        e_end = e_end.squeeze(dim=0)  # (seq_l, 2)

        start_prob, end_prob = e_start.T[0].T.tolist(), e_end.T[0].T.tolist()  # 取预测的第一维作为prob
        start_digit, end_digit = e_start.sort(dim=1).indices.T[0].T.tolist(), e_end.sort(dim=1).indices.T[0].T.tolist()

        spans = tools.argument_span_determination(start_digit, end_digit, start_prob, end_prob)

        words = []
        detailed_words = []
        for elem_span in spans:
            word = tokenize_tools.tokenSpan_to_word(sentence, elem_span, offset_mapping)
            words.append(word)
            detailed_words.append({
                'word': word,
                'token_span': elem_span
            })

        if len(words) != 0:
            result[cur_type] = words
            detailed[cur_type] = detailed_words
    return result, detailed


def convert_output_to_evaluate_format_T(trigger_start: torch.Tensor, trigger_end: torch.Tensor, mask: torch.Tensor, sentence: str, offset_mapping: OffsetMapping, event_types: list = EE_settings.event_types_full, threshold: float = 0.5):
    """
    将模型的输出转换为evaluator能够直接判定的格式

    :param trigger_start:
    :param trigger_end:  (1, seq_l, event_type_cnt)
    :param mask: 不需要
    :param sentence: 原句
    :param offset_mapping: FastTokenizer的输出
    :return:
    """
    result = {}
    detailed = {}
    trigger_start = trigger_start.squeeze(0).T
    trigger_end = trigger_end.squeeze(0).T  # both (event_type_cnt, seq_l)
    for i_type, (e_start, e_end) in enumerate(list(zip(trigger_start, trigger_end))):
        # e_start (seq_l); e_end (seq_l)
        cur_type = event_types[i_type]

        start_digit = (e_start > threshold).int().tolist()
        end_digit = (e_end > threshold).int().tolist()
        spans = tools.argument_span_determination(start_digit, end_digit, e_start, e_end)

        words = []
        detailed_words = []
        for elem_span in spans:
            word = tokenize_tools.tokenSpan_to_word(sentence, elem_span, offset_mapping)
            words.append(word)
            detailed_words.append({
                'word': word,
                'token_span': elem_span
            })

        if len(words) != 0:
            result[cur_type] = words
            detailed[cur_type] = detailed_words
    return result, detailed


class PLMEE_Trigger_Evaluator(BaseEvaluator):
    def __init__(self, dataset_type: str = 'FewFC'):
        super(PLMEE_Trigger_Evaluator, self).__init__()
        self.dataset_type = dataset_type
        self.f1_eval = F1_Evaluator()
        self.pred_lst = []
        self.gt_lst = []

    def eval_single(self, pred_starts: List[torch.Tensor], pred_ends: List[torch.Tensor], mask: torch.Tensor, sentence: str, offset_mapping: OffsetMapping, gt: dict):
        if self.dataset_type == 'FewFC':
            event_types = EE_settings.event_types_full
        elif self.dataset_type == 'Duee':
            event_types = EE_settings.duee_event_types
        else:
            raise Exception(f'{self.dataset_type}数据集不存在！')

        converted_preds, detailed = convert_output_to_evaluate_format(pred_starts, pred_ends, mask, sentence, offset_mapping, event_types)

        preds, gts = [], []

        for key, value in converted_preds.items():
            for elem_v in value:
                preds.append(key + ':' + elem_v)

        for key, value in gt.items():
            for elem_v in value:
                gts.append(key + ':' + elem_v['word'])

        gts = list(set(gts))
        self.f1_eval.eval_single(preds, gts)
        self.pred_lst.append({
            'results': preds,
            'detailed': detailed,
            'sentence': sentence,
            'offset_mapping': offset_mapping
        })
        self.gt_lst.append(gts)

    def eval_step(self) -> Dict[str, Any]:
        result = self.f1_eval.eval_step()
        self.pred_lst = []
        self.gt_lst = []
        return result


class PLMEE_Trigger_T_Evaluator(BaseEvaluator):
    def __init__(self, dataset_type: str = 'FewFC'):
        super(PLMEE_Trigger_T_Evaluator, self).__init__()
        self.dataset_type = dataset_type
        self.f1_eval = F1_Evaluator()
        self.pred_lst = []
        self.gt_lst = []

    def eval_single(self, trigger_start: torch.Tensor, trigger_end: torch.Tensor, attention_mask: torch.Tensor, sentence: str, offset_mapping: OffsetMapping, gt: dict):
        if self.dataset_type == 'FewFC':
            event_types = EE_settings.event_types_full
        elif self.dataset_type == 'Duee':
            event_types = EE_settings.duee_event_types
        else:
            raise Exception(f'{self.dataset_type}数据集不存在！')

        converted_preds, detailed = convert_output_to_evaluate_format_T(trigger_start, trigger_end, attention_mask, sentence, offset_mapping, event_types)

        preds, gts = [], []

        for key, value in converted_preds.items():
            for elem_v in value:
                preds.append(key + ':' + elem_v)

        for key, value in gt.items():
            for elem_v in value:
                gts.append(key + ':' + elem_v['word'])

        gts = list(set(gts))
        self.f1_eval.eval_single(preds, gts)
        self.pred_lst.append({
            'results': preds,
            'detailed': detailed,
            'sentence': sentence,
            'offset_mapping': offset_mapping
        })
        self.gt_lst.append(gts)

    def eval_step(self) -> Dict[str, Any]:
        result = self.f1_eval.eval_step()
        self.pred_lst = []
        self.gt_lst = []
        return result


def train_dataset_factory(data_dicts: List[dict], bsz: int = EE_settings.default_bsz, shuffle: bool = EE_settings.default_shuffle, dataset_type: str = 'FewFC', model_type: str = 'M'):
    if dataset_type == 'FewFC':
        event_types = EE_settings.event_types_full
    elif dataset_type == 'Duee':
        event_types = EE_settings.duee_event_types
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
        - labels
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

        # generate label tensor
        labels = data_dict['labels']
        start_labels, end_labels = [], []
        for idx, elem_type in enumerate(event_types):
            start_label = torch.zeros((bsz, seq_l), dtype=torch.long)
            end_label = torch.zeros((bsz, seq_l), dtype=torch.long)
            for i_batch in range(bsz):
                if elem_type in labels[i_batch]:
                    for elem_trigger in labels[i_batch][elem_type]:
                        start_label[i_batch][elem_trigger['span'][0]] = 1
                        end_label[i_batch][elem_trigger['span'][1]] = 1
            start_labels.append(start_label)
            end_labels.append(end_label)

        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
               }, {
            'start_labels': start_labels,
            'end_labels': end_labels
        }

    def collate_fn_T(lst):
        """
        - content
        - input_ids
        - token_type_ids
        - attention_mask
        - labels
        """
        data_dict = tools.transpose_list_of_dict(lst)
        bsz = len(lst)

        # generate basic input
        input_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['input_ids']), dtype=torch.long)
        token_type_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['token_type_ids']), dtype=torch.long)
        attention_mask = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['attention_mask']), dtype=torch.long)
        seq_l = input_ids.shape[1]
        # all (bsz, max_seq_l)

        # generate label tensor
        labels = data_dict['labels']
        start_label, end_label = torch.zeros((bsz, seq_l, len(event_types))), torch.zeros((bsz, seq_l, len(event_types)))
        for idx, elem_type in enumerate(event_types):
            for i_batch in range(bsz):
                if elem_type in labels[i_batch]:
                    for elem_trigger in labels[i_batch][elem_type]:
                        start_label[i_batch][elem_trigger['span'][0]][idx] = 1
                        end_label[i_batch][elem_trigger['span'][1]][idx] = 1

        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
               }, {
            'trigger_start_label': start_label,
            'trigger_end_label': end_label
        }

    if model_type == 'M':
        train_dataloader = DataLoader(train_dataset, batch_size=bsz, shuffle=shuffle, collate_fn=collate_fn)
    elif model_type == 'T':
        train_dataloader = DataLoader(train_dataset, batch_size=bsz, shuffle=shuffle, collate_fn=collate_fn_T)
    else:
        raise Exception(f'{model_type}模型不存在！')

    return train_dataloader


def valid_dataset_factory(data_dicts: List[dict], dataset_type: str = 'FewFC'):
    valid_dataset = SimpleDataset(data_dicts)

    def collate_fn(lst):
        """
        - input_ids
        - token_type_ids
        - attention_mask
        - gts
        :param lst:
        :return:
        """
        data_dict = tools.transpose_list_of_dict(lst)
        bsz = len(lst)

        # generate basic input
        input_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['input_ids']), dtype=torch.long)
        token_type_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['token_type_ids']), dtype=torch.long)
        attention_mask = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['attention_mask']), dtype=torch.long)

        gts = data_dict['gts'][0]
        sentence = gts['sentence']
        offset_mapping = gts['offset_mapping']
        gt = gts['gt']

        return {
           'input_ids': input_ids,
           'token_type_ids': token_type_ids,
           'attention_mask': attention_mask
               }, {
            'sentence': sentence,
            'offset_mapping': offset_mapping,
            'gt': gt
        }

    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return valid_dataloader


def dataset_factory(train_file: str, valid_file: str, bsz: int = EE_settings.default_bsz, shuffle: bool = EE_settings.default_shuffle, dataset_type: str = 'FewFC', model_type: str = 'M'):
    train_data_dicts = pickle.load(open(train_file, 'rb'))
    valid_data_dicts = pickle.load(open(valid_file, 'rb'))
    print(f'dataset_type: {dataset_type}')

    train_dataloader = train_dataset_factory(train_data_dicts, bsz=bsz, shuffle=shuffle, dataset_type=dataset_type, model_type=model_type)
    valid_dataloader = valid_dataset_factory(valid_data_dicts, dataset_type=dataset_type)

    return train_dataloader, valid_dataloader


class UseModel:
    def __init__(self, state_dict_path: str, init_params_path: str, use_gpu: bool = False, plm_path: str = EE_settings.default_plm_path, dataset_type: str = 'FewFC'):
        # 首先加载初始化模型所使用的参数
        init_params = pickle.load(open(init_params_path, 'rb'))
        self.model = PLMEE_Trigger(**init_params)
        if not use_gpu:
            self.model.load_state_dict(torch.load(open(state_dict_path, 'rb'), map_location=torch.device('cpu')))
        else:
            self.model.load_state_dict(torch.load(open(state_dict_path, 'rb'), map_location=torch.device('cuda')))

        # 初始化 raw2input, output2read所使用的工具
        # 该部分的参数直接从__init__传入就好

        self.tokenizer = AutoTokenizer.from_pretrained(plm_path)

        if dataset_type == 'FewFC':
            self.event_types = EE_settings.event_types_full
        elif dataset_type == 'Duee':
            self.event_types = EE_settings.duee_event_types
        else:
            raise Exception(f'{dataset_type}数据集不存在！')


    def __call__(self, sentence: str):
        """
        从sentence中抽取触发词以及对应的类型。
        :param sentence:
        :return:
        """

        tokenized = self.tokenizer(sentence, return_offsets_mapping=True, max_length=256)
        input_ids = torch.tensor(tokenized['input_ids']).unsqueeze(dim=0)  # (1, seq_l)
        token_type_ids = torch.tensor(tokenized['token_type_ids']).unsqueeze(dim=0)  # (1, seq_l)
        attention_mask = torch.tensor(tokenized['attention_mask']).unsqueeze(dim=0)  # (1, seq_l)
        offset_mapping = tokenized['offset_mapping']
        tokens = self.tokenizer.convert_ids_to_tokens(tokenized['input_ids'])

        result = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pred_starts, pred_ends, mask = result['pred_starts'], result['pred_ends'], result['mask']
        converted = convert_output_to_evaluate_format(pred_starts, pred_ends, mask, sentence, offset_mapping, self.event_types)
        return converted


def output_result(use_model: ..., input_filename: str, output_filename: str):
    """
    读取input文件，使用use_model抽取其中所有句子的触发词，然后写入output_filename文件当中
    :param use_model:
    :param input_filename: jsonl格式，{'content': 需要抽取的句子}
    :param output_filename: jsonl格式, {'content': 需要抽取的句子, 'event': }
    :return:
    """
    inp = load_jsonl(input_filename)
    for idx in tqdm(range(len(inp))):
        elem = inp[idx]
        elem['events'] = use_model(elem['content'])
    dump_jsonl(inp, output_filename)


model_registry = {
    'model': PLMEE_Trigger_T,
    'loss': PLMEE_Trigger_T_Loss,
    'evaluator': PLMEE_Trigger_T_Evaluator,
    'train_val_data': dataset_factory,
    'recorder': NaiveRecorder,
    'func': output_result,
    'use_model': UseModel
}


"""
用于测试的一些函数
"""

def generate_trial_data(dataset_type: str):
    if dataset_type == 'Duee':
        train_file = 'temp_data/train.PLMEE_Trigger.Duee.labeled.pk'
        valid_file = 'temp_data/valid.PLMEE_Trigger.Duee.gt.pk'
    elif dataset_type == 'FewFC':
        train_file = 'temp_data/train.PLMEE_Trigger.FewFC.labeled.pk'
        valid_file = 'temp_data/valid.PLMEE_Trigger.FewFC.gt.pk'
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


def predict_duee_test():
    input_filename = '../../../data/NLP/EventExtraction/duee/duee_test2.json/duee_test2.json'
    output_filename = '../../../checkpoint/duee_trigger_result.json'



if __name__ == '__main__':
    # predict_duee_test()
    train_dataloader, train_data, valid_dataloader, valid_data = generate_trial_data('Duee')
