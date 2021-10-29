import copy
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import work.EE.PLMEE.EE_settings as EE_settings
from transformers import BertModel
from torch.optim import AdamW
from evaluate.evaluator import BaseEvaluator, MultiLabelClsEvaluator
from utils import tools
from utils.data import SimpleDataset
from models.model_utils import get_init_params


class EventDetection(nn.Module):
    def __init__(self,
                 plm_path='bert-base-chinese',
                 hidden_dropout_prob=0.3,
                 n_labels=len(EE_settings.event_types_full),
                 plm_lr=EE_settings.plm_lr,
                 others_lr=EE_settings.others_lr,
                 threshold=EE_settings.event_detection_threshold,
                 event_types_lst=EE_settings.event_types_full):
        """

        :param plm_path:
        :param hidden_dropout_prob:
        :param n_labels:
        :param plm_lr:
        :param others_lr:
        :param threshold:
        :param event_types_lst:
        """
        super(EventDetection, self).__init__()
        self.init_params = get_init_params(locals())
        self.bert = BertModel.from_pretrained(plm_path)
        self.hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.plm_lr = plm_lr
        self.others_lr = others_lr
        self.classifier = nn.Linear(self.hidden_size, n_labels)
        self.threshold = threshold
        self.event_types_lst = event_types_lst
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0)

    def get_optimizers(self):
        plm_parameters = self.bert.parameters()
        linear_parameters = self.classifier.parameters()
        plm_opt = AdamW(params=plm_parameters, lr=self.plm_lr)
        others_opt = AdamW(params=linear_parameters, lr=self.others_lr)
        return [plm_opt, others_opt]

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        #  Bert output
        #   Tuple[embed (bsz, seq_l, hidden), pooled embed (bsz, hidden)]
        pooled_output = output[1]   # (bsz, hidden_size)
        pooled_output = self.dropout(pooled_output)
        #  (bsz, hidden)
        logits = self.classifier(pooled_output)  # (bsz, num_labels)
        probs = F.sigmoid(logits)
        if self.training:
            return {
                "logits": probs  # (bsz, num_labels)
            }
        else:
            pred = (probs > self.threshold).squeeze().int().tolist()  # list of len num_labels
            pred_types = []
            for idx in range(len(pred)):
                if pred[idx] == 1:
                    pred_types.append(self.event_types_lst[idx])
            return {
                "types": pred_types
            }


class EventDetectionLoss(nn.Module):
    def __init__(self, pref=2):
        super(EventDetectionLoss, self).__init__()
        self.pos_pref_weight = tools.PosPrefWeight(pref)

    def forward(self, logits=None, labels=None):
        """

        :param logits: model output logits (bsz, num_labels)
        :param gt: (bsz, num_labels):
        :return:
        """
        reshaped_result = logits.squeeze()  # （bsz, num_labels）
        pos_weight = self.pos_pref_weight(labels)
        loss = F.binary_cross_entropy(reshaped_result, labels.cuda(), pos_weight.cuda())
        return loss

# 垃圾代码应该埋了
#
# class EventDetectionEvaluator(BaseEvaluator):
#     def __init__(self, threshold=EE_settings.event_detection_threshold):
#         super(EventDetectionEvaluator, self).__init__()
#         self.threshold = threshold
#         self.totals, self.predicts, self.corrects = [], [], []
#
#     def evaluate(self, logits_dicts, gt_dicts):
#         """
#
#         :param logits_dicts: list of logits dicts. A logits dict be like {"logits": Tensor}
#         :param gt_dicts:
#         :return:
#         """
#         assert len(logits_dicts) == len(gt_dicts)
#         total, predict, correct = 0, 0, 0
#         for i in range(len(logits_dicts)):
#             logits_dict, gt_dict = logits_dicts[i], gt_dicts[i]
#             i_total, i_predict, i_correct = self.eval_single(**logits_dict, **gt_dict)
#             total += i_total
#             predict += i_predict
#             correct += i_correct
#         recall = correct / total if total != 0 else 0
#         precision = correct / predict if predict != 0 else 0
#         f_measure = (2 * recall * precision) / (recall + precision) if recall + precision != 0 else 0
#         return precision, recall, f_measure
#
#     def eval_single(self, logits=None, gt=None):
#         """
#         todo 需要一次输入所有的结果与gt
#         因为是evaluate阶段，所以默认下面的bsz全部为1
#         :param logits: model output logits (bsz, 1, num_labels)
#         :param gt: (bsz, num_labels)
#         :return:
#         """
#         assert logits.shape[0] == gt.shape[0] == 1, f'evaluate阶段，bsz应该都为0:logits.size:{logits.shape}, gt.size:{gt.shape}'
#         reshaped_result = paddle.cast(logits.squeeze() > self.threshold, paddle.int64).tolist()  # (num_labels)
#         reshaped_gt = paddle.cast(gt.squeeze(), paddle.int64).tolist()  # (num_labels)
#         total, predict, correct = 0, 0, 0
#         for i_gt, c_gt in enumerate(reshaped_gt):
#             c_rt = reshaped_result[i_gt]
#             if c_gt == 1:
#                 total += 1
#             if c_rt == 1:
#                 predict += 1
#             if c_rt == c_gt == 1:
#                 correct += 1
#         self.totals.append(total)
#         self.predicts.append(predict)
#         self.corrects.append(correct)
#
#         predicted_types = []
#         for idx_event, elem_pred in enumerate(reshaped_result):
#             if elem_pred == 1:
#                 predicted_types.append(pd_settings.event_types_real[idx_event])
#         return predicted_types
#
#     def eval_step(self):
#         total = sum(self.totals)
#         predict = sum(self.predicts)
#         correct = sum(self.corrects)
#         self.totals, self.predicts, self.corrects = [], [], []
#         recall = correct / total if total != 0 else 0
#         precision = correct / predict if predict != 0 else 0
#         f_measure = (2 * recall * precision) / (recall + precision) if recall + precision != 0 else 0
#         return f'|| f1: {f_measure:<7.5f} || precision: {precision:<7.5f} || recall: {recall:<7.5f} ||' \
#                f'\n|| total: {total} || predict: {predict} || correct: {correct}||'


class AssembledEvaluator(BaseEvaluator):
    def __init__(self):
        super(AssembledEvaluator, self).__init__()
        self.multi_label_evaluator = MultiLabelClsEvaluator()
        self.total_types, self.total_gt = [], []

    def eval_single(self, types: List[str], gts: List[str]):
        self.total_types.append(copy.deepcopy(types))
        self.total_gt.append(copy.deepcopy(gts))
        self.multi_label_evaluator.eval_single(types, gts)

    def eval_step(self) -> Dict[str, Any]:
        result = self.multi_label_evaluator.eval_step()
        self.total_types, self.total_gt = [], []
        return result


def generate_event_type_label(data_dict: Dict[str, Any], event_type_idx: Dict[str, int]= EE_settings.event_types_full_index):
    event_types = data_dict['event_types']
    label = np.zeros(len(event_type_idx))
    for elem_type in event_types:
        label[event_type_idx[elem_type]] = 1
    data_dict['label'] = label
    return [data_dict]


def train_dataset_factory(train_filename: str, bsz: int):
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

    # extract event types
    data_dicts = tools.map_operation_to_list_elem(extract_event_types, data_dicts)
    # [content, events, event_types, input_ids, token_type_ids, attention_mask, token]

    # generate event_type label
    data_dicts = tools.map_operation_to_list_elem(generate_event_type_label, data_dicts)
    # [][content, events, event_types, input_ids, token_type_ids, attention_mask, token, label]

    train_dataset = SimpleDataset(data_dicts)

    def collate_fn(lst):
        """

        :param lst: List[dict], key contains:
            input_ids
            token_type_ids
            attention_mask
            label (event_cnt)
        :return:
        """
        dict_of_data = tools.transpose_list_of_dict(lst)
        input_ids = list(map(np.array, dict_of_data['input_ids']))
        token_type_ids = list(map(np.array, dict_of_data['token_type_ids']))
        attention_mask = list(map(np.array, dict_of_data['attention_mask']))

        batchified_input_ids = torch.tensor(tools.batchify_ndarray(input_ids))  # (bsz, max_seq_l)
        batchified_token_type_ids = torch.tensor(tools.batchify_ndarray(token_type_ids))  # (bsz, max_seq_l)
        batchified_attention_mask = torch.tensor(tools.batchify_ndarray(attention_mask))  # (bsz, max_seq_l)

        batchified_labels = torch.tensor(np.array(dict_of_data['label']), dtype=torch.float)  # (bsz, event_cnt)

        # convert to cuda
        batchified_input_ids = batchified_input_ids.cuda()
        batchified_token_type_ids = batchified_token_type_ids.cuda()
        batchified_attention_mask = batchified_attention_mask.cuda()
        batchified_labels = batchified_labels.cuda()

        return {
            "input_ids": batchified_input_ids,
            "token_type_ids": batchified_token_type_ids,
            "attention_mask": batchified_attention_mask
               }, {
            "labels": batchified_labels
        }

    train_dataloader = DataLoader(train_dataset, batch_size=bsz, shuffle=True, collate_fn=collate_fn)
    return train_dataloader


def val_dataset_factory(val_filename: str):
    json_lines = tools.read_json_lines(val_filename)
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

    # extract event types
    data_dicts = tools.map_operation_to_list_elem(extract_event_types, data_dicts)
    # [content, events, event_types, input_ids, token_type_ids, attention_mask, token]

    val_dataset = SimpleDataset(data_dicts)

    def collate_fn(lst):
        """

        :param lst: List[dict] keys:
            input_ids, token_type_ids, attention_mask, event_types
        :return:
        """
        dict_of_data = tools.transpose_list_of_dict(lst)
        input_ids = list(map(np.array, dict_of_data['input_ids']))
        token_type_ids = list(map(np.array, dict_of_data['token_type_ids']))
        attention_mask = list(map(np.array, dict_of_data['attention_mask']))

        batchified_input_ids = torch.tensor(tools.batchify_ndarray(input_ids))  # (bsz, max_seq_l)
        batchified_token_type_ids = torch.tensor(tools.batchify_ndarray(token_type_ids))  # (bsz, max_seq_l)
        batchified_attention_mask = torch.tensor(tools.batchify_ndarray(attention_mask))  # (bsz, max_seq_l)

        # convert to cuda
        batchified_input_ids = batchified_input_ids.cuda()
        batchified_token_type_ids = batchified_token_type_ids.cuda()
        batchified_attention_mask = batchified_attention_mask.cuda()

        gts = dict_of_data['event_types']

        return {
            "input_ids": batchified_input_ids,
            "token_type_ids": batchified_token_type_ids,
            "attention_mask": batchified_attention_mask
               }, {
            "gts": gts[0]  # List[str]
        }

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return val_dataloader


class UseModel:
    def __init__(self, state_dict_path: str, init_params_path):
        init_params = pickle.load(open(init_params_path, 'rb'))
        self.model = EventDetection(**init_params)
        self.model.load_state_dict(torch.load(open(state_dict_path, 'rb'), map_location=torch.device('cpu')))
        self.model.eval()

        # convert部分
        self.tokenizer = tools.bert_tokenizer()

    def __call__(self, sentence: str) -> dict:
        sentence = sentence.replace(' ', '_')
        tokenized = self.tokenizer([sentence])[0]
        input_ids = torch.tensor(tokenized['input_ids']).unsqueeze(dim=0)  # (1, seq_l)
        token_type_ids = torch.tensor(tokenized['token_type_ids']).unsqueeze(dim=0)  # (1, seq_l)
        attention_mask = torch.tensor(tokenized['attention_mask']).unsqueeze(dim=0)  # (1, seq_l)
        print(f'tokenized:{tokenized}')
        print(f'input_ids.shape:{input_ids.shape}')
        result = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return result['types']


model_registry = {
    "model": EventDetection,
    "loss": EventDetectionLoss,
    "evaluator": AssembledEvaluator,
    "train_data": train_dataset_factory,
    "val_data": val_dataset_factory,
    "args": [
        {'name': "--train_file", 'dest': 'train_filename', 'type': str, 'help': '训练/测试数据文件的路径'},
        {'name': "--val_file", 'dest': 'val_filename', 'type': str, 'help': '训练/测试数据文件的路径'},
    ],
    "use": UseModel
}


if __name__ == '__main__':
    pass
