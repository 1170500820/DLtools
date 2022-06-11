import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertModel, AutoTokenizer
from itertools import chain
import pickle

from type_def import *
from utils import tools, batch_tool, tokenize_tools
from utils.tokenize_tools import OffsetMapping
from utils.data import SimpleDataset
from evaluate.evaluator import BaseEvaluator, F1_Evaluator, SentenceWithEvent, Events, Event, Mention, Mentions
from models.model_utils import get_init_params
from analysis.recorder import NaiveRecorder

from work.EE import EE_settings, EE_utils


class PLMEE_Argument(nn.Module):
    """
    与PLMEE_Trigger的结构相似，
    """
    def __init__(self, plm_lr: float = EE_settings.plm_lr, linear_lr: float = EE_settings.others_lr,
                 plm_path: str = EE_settings.default_plm_path, dataset_type: str = 'FewFC'):
        super(PLMEE_Argument, self).__init__()
        self.init_params = get_init_params(locals())

        self.plm_lr = plm_lr
        self.linear_lr = linear_lr
        self.plm_path = plm_path
        self.dataset_type = dataset_type
        if self.dataset_type == 'FewFC':
            self.role_types = EE_settings.role_types
        elif self.dataset_type == 'Duee':
            self.role_types = EE_settings.duee_role_types
        else:
            raise Exception(f'{dataset_type}数据集不存在！')

        self.bert = BertModel.from_pretrained(self.plm_path)
        self.hidden = self.bert.config.hidden_size

        # 分别用于预测论元的开始与结束位置
        self.start_classifiers = nn.ModuleList(nn.Linear(self.hidden, 2) for i in range(len(self.role_types)))
        self.end_classifiers = nn.ModuleList(nn.Linear(self.hidden, 2) for i in range(len(self.role_types)))

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
            pred_starts.append(cls(embed))  # (bsz, seq_l, 2)
        for cls in self.end_classifiers:
            pred_ends.append(cls(embed))  # (bsz, seq_l, 2)

        return {
            'pred_starts': pred_starts,
            'pred_ends': pred_ends,
            'mask': attention_mask
        }


class PLMEE_Argument_Loss(nn.Module):
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
        role_loss = []
        for (p_start, p_end, l_start, l_end) in zip(pred_starts, pred_ends, start_labels, end_labels):
            start_losses, end_losses = [], []
            l_start, l_end = l_start.cuda(), l_end.cuda()
            for i_batch in range(bsz):
                start_loss = F.cross_entropy(p_start[i_batch], l_start[i_batch], reduction='none')  # (bsz, seq_l)
                end_loss = F.cross_entropy(p_end[i_batch], l_end[i_batch], reduction='none')  # (bsz, seq_l)
                start_loss = torch.sum(start_loss * mask[i_batch]) / torch.sum(mask[i_batch])
                end_loss = torch.sum(end_loss * mask[i_batch]) / torch.sum(mask[i_batch])
            start_loss = sum(start_losses) / len(start_losses)
            end_loss = sum(end_losses) / len(end_losses)
            combines_loss = start_loss + end_loss
            role_loss.append(combines_loss)

        loss = sum(role_loss)
        return loss

def convert_output_to_evaluate_format(pred_starts: List[torch.Tensor], pred_ends: List[torch.Tensor], mask: torch.Tensor, sentence: str, offset_mapping: OffsetMapping, role_types: list = EE_settings.role_types):
    """

    :param pred_starts:
    :param pred_ends:
    :param mask:
    :param sentence:
    :param offset_mapping:
    :param role_types:
    :return:
    """
    result = {}
    for i_type, (e_start, e_end) in enumerate(list(zip(pred_starts, pred_ends))):
        cur_type = role_types[i_type]
        e_start = e_start.squeeze(dim=0)  # (seq_l, 2)
        e_end = e_end.squeeze(dim=0)  # (seq_l, 2)

        start_prob, end_prob = e_start.T[1].T.tolist(), e_end.T[1].T.tolist()  # 取预测的第二维作为prob
        start_digit, end_digit = e_start.sort(dim=1).indices.T[1].T.tolist(), e_end.sort(dim=1).indices.T[1].T.tolist()

        spans = tools.argument_span_determination(start_digit, end_digit, start_prob, end_prob)

        words = []
        for elem_span in spans:
            words.append(tokenize_tools.tokenSpan_to_word(sentence, elem_span, offset_mapping))

        if len(words) != 0:
            result[cur_type] = words
    return result


class PLMEE_Argument_Evaluator(BaseEvaluator):
    def __init__(self, dataset_type: str = 'FewFC'):
        super(PLMEE_Argument_Evaluator, self).__init__()
        self.dataset_type = dataset_type
        self.f1_eval = F1_Evaluator()
        self.pred_lst = []
        self.gt_lst = []

    def eval_single(self, pred_starts: List[torch.Tensor], pred_ends: List[torch.Tensor], mask: torch.Tensor, sentence: str, offset_mapping: OffsetMapping, gt: dict):
        if self.dataset_type == 'FewFC':
            role_types = EE_settings.role_types
        elif self.dataset_type == 'Duee':
            role_types = EE_settings.duee_role_types
        else:
            raise Exception(f'{self.dataset_type}数据集不存在！')

        converted_preds = convert_output_to_evaluate_format(pred_starts, pred_ends, mask, sentence, offset_mapping, role_types)

        preds, gts = [], []

        for key, value in converted_preds.items():
            for elem_v in value:
                preds.append(key + '-' +  elem_v)

        for key, value in gt.items():
            for elem_v in value:
                gts.append(key + '-' +  elem_v['word'])

        gts = list(set(gts))
        self.f1_eval.eval_single(preds, gts)
        self.pred_lst.append(preds)
        self.gt_lst.append(gts)

    def eval_step(self) -> Dict[str, Any]:
        result = self.f1_eval.eval_step()
        self.pred_lst = []
        self.gt_lst = []
        return result


def train_dataset_factory(data_dicts: List[dict], bsz: int = EE_settings.default_bsz, shuffle: bool = EE_settings.default_shuffle, dataset_type: str = 'FewFC'):
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
        data_dict包括
        - input_ids
        - token_type_ids
        - attention_mask
        - label
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

        label = data_dict['label']
        start_labels, end_labels = [], []
        for idx, elem_role in enumerate(role_types):
            start_label = torch.zeros((bsz, seq_l), dtype=torch.long)
            end_label = torch.zeros((bsz, seq_l), dtype=torch.long)
            for i_batch in range(bsz):
                if elem_role in label[i_batch]:
                    for elem_arg_span in label[i_batch][elem_role]:
                        start_label[i_batch][elem_arg_span[0]] = 1
                        end_label[i_batch][elem_arg_span[1]] = 1
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


    train_dataloader = DataLoader(train_dataset, batch_size=bsz, shuffle=shuffle, collate_fn=collate_fn)

    return train_dataloader



def valid_dataset_factory(data_dicts: List[dict], dataset_type: str = 'FewFC'):
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
        - input_ids
        - token_type_ids
        - attention_mask
        - sentence
        - offset_mapping
        - gt
        :param lst:
        :return:
        """
        data_dict = tools.transpose_list_of_dict(lst)
        bsz = len(lst)

        # generate basic input
        input_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['input_ids']), dtype=torch.long)
        token_type_ids = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['token_type_ids']), dtype=torch.long)
        attention_mask = torch.tensor(batch_tool.batchify_ndarray1d(data_dict['attention_mask']), dtype=torch.long)

        gt = data_dict['gt'][0]

        return {
           'input_ids': input_ids,
           'token_type_ids': token_type_ids,
           'attention_mask': attention_mask
               }, {
            'sentence': data_dict['sentence'],
            'offset_mapping': data_dict['offset_mapping'],
            'gt': gt
        }

    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return valid_dataloader


def dataset_factory(train_file: str, valid_file: str, bsz: int = EE_settings.default_bsz, shuffle: bool = EE_settings.default_shuffle, dataset_type: str = 'FewFC'):
    train_data_dicts = pickle.load(open(train_file, 'rb'))
    valid_data_dicts = pickle.load(open(valid_file, 'rb'))
    print(f'dataset_type: {dataset_type}')

    train_dataloader = train_dataset_factory(train_data_dicts, bsz=bsz, shuffle=shuffle, dataset_type=dataset_type)
    valid_dataloader = valid_dataset_factory(valid_data_dicts, dataset_type=dataset_type)

    return train_dataloader, valid_dataloader


class UseModel:
    def __init__(self, state_dict_path: str, init_params_path: str, use_gpu: bool = False, plm_path: str = EE_settings.default_plm_path, dataset_type: str = 'FewFC'):
        # 首先加载初始化模型所使用的参数
        init_params = pickle.load(open(init_params_path, 'rb'))
        self.model = PLMEE_Argument(**init_params)
        if not use_gpu:
            self.model.load_state_dict(torch.load(open(state_dict_path, 'rb'), map_location=torch.device('cpu')))
        else:
            self.model.load_state_dict(torch.load(open(state_dict_path, 'rb'), map_location=torch.device('gpu')))

        # 初始化raw2input, output2read所使用的工具

        self.tokenizer = AutoTokenizer.from_pretrained(plm_path)
        if dataset_type == 'FewFC':
            self.role_types = EE_settings.role_types
        elif dataset_type == 'Duee':
            self.role_types = EE_settings.duee_role_types
        else:
            raise Exception(f'{dataset_type}数据集不存在！')

    def __call__(self, sentence: str, trigger_span: Tuple[int, int]):
        pass


def output_result(use_model: ..., input_filename: str, output_filename: str):
    pass


model_registry = {
    'model': PLMEE_Argument,
    'loss': PLMEE_Argument_Loss,
    'evaluator': PLMEE_Argument_Evaluator,
    'train_val_data': dataset_factory,
    'recorder': NaiveRecorder,
    'func': output_result,
    'use_model': UseModel
}


if __name__ == '__main__':
    train_file = 'temp_data/train.PLMEE_Argument.Duee.labeled.pk'
    valid_file = 'temp_data/valid.PLMEE_Argument.Duee.gt.pk'

    bsz = 4
    shuffle = False
    dataset_type = 'Duee'

    train_data_dicts = pickle.load(open(train_file, 'rb'))
    valid_data_dicts = pickle.load(open(valid_file, 'rb'))

    train_dataloader = train_dataset_factory(train_data_dicts, bsz=bsz, shuffle=shuffle, dataset_type=dataset_type)
    valid_dataloader = valid_dataset_factory(valid_data_dicts, dataset_type=dataset_type)

    limit = 5
    train_data, valid_data = [], []
    for idx, (train_sample, valid_sample) in enumerate(list(zip(train_dataloader, valid_dataloader))):
        train_data.append(train_sample)
        valid_data.append(valid_sample)
        if idx >= limit:
            break
