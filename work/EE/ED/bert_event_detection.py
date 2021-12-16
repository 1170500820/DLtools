import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertModel
import copy

from type_def import *
from models.model_utils import get_init_params
from evaluate.evaluator import BaseEvaluator, MultiLabelClsEvaluator
from work.EE import EE_settings
from utils import tools, batch_tool
from dataset import ee_dataset
from utils.data import SimpleDataset


"""
Model
    input:
        - input_ids    (bsz, seq_l)
        - token_type_ids        (bsz, seq_l)
        - attention_mask        (bsz, seq_l)
    output(train):
        - logits        (bsz, num_labels)
    output(eval):
        - probs     (bsz=1, num_labels)
Loss
    input:
        - logits        (bsz, num_labels)
        - labels        (bsz, num_labels)
    output:
        - loss
Evaluator
    input:
        - types     List[str]
        - gts       List[str]
    output:
        result
"""


# model
class EventDetection(nn.Module):
    def __init__(self,
                 plm_path=EE_settings.default_plm_path,
                 hidden_dropout_prob=EE_settings.default_dropout_prob,
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
            # pred = (probs > self.threshold).squeeze().int().tolist()  # list of len num_labels
            # pred_types = []
            # for idx in range(len(pred)):
            #     if pred[idx] == 1:
            #         pred_types.append(self.event_types_lst[idx])
            # return {
            #     "types": pred_types
            # }
            return {
                "probs": probs
            }



# loss
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


# evaluator
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


# dataset
def dataset_factory(data_dir: str, data_type: str, bsz: int = EE_settings.default_bsz):
    result = ee_dataset.building_event_detection_dataset(data_dir, data_type, EE_settings.default_plm_path)

    def train_collate_fn(lst):
        dict_of_list = tools.transpose_list_of_dict(lst)

        input_ids = torch.tensor(tools.batchify_ndarray(dict_of_list['input_ids']))  # (bsz, seq_l)
        token_type_ids = torch.tensor(tools.batchify_ndarray(dict_of_list['token_type_ids']))  # (bsz, seq_l)
        attention_mask = torch.tensor(tools.batchify_ndarray(dict_of_list['attention_mask']))  # (bsz, seq_l)
        labels = torch.tensor(tools.batchify_ndarray(dict_of_list['label']))  # (bsz, label_cnt)

        return {
            'input_ids': input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask
               }, {
            "labels": labels
        }

    def valid_collate_fn(lst):
        dict_of_list = tools.transpose_list_of_dict(lst)

        input_ids = torch.tensor(tools.batchify_ndarray(dict_of_list['input_ids']))  # (bsz=1, seq_l)
        token_type_ids = torch.tensor(tools.batchify_ndarray(dict_of_list['token_type_ids']))  # (bsz=1, seq_l)
        attention_mask = torch.tensor(tools.batchify_ndarray(dict_of_list['attention_mask']))  # (bsz=1, seq_l)
        gts = dict_of_list['event_types']

        return {
            'input_ids': input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask
               }, {
            "gts": gts
        }

    train, valid = result['train'], result['valid']
    train_dataset = SimpleDataset(train)
    train_dataloader = DataLoader(train_dataset, batch_size=bsz, shuffle=True, collate_fn=train_collate_fn)
    valid_dataset = SimpleDataset(valid)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=valid_collate_fn)
    return train_dataloader, valid_dataloader


# sample_to_eval_format
def sample_to_eval_format(sentence):
    data_dict = {"content": sentence}


# eval_output_to_read_format
def eval_output_to_read_format(probs: torch.Tensor):
    """

    :param probs: (bsz=1, label_cnt)
    :return:
    """
    pred = (probs > EE_settings.event_detection_threshold).squeeze().int().tolist()
    pred_types = []
    for idx in range(len(pred)):
        if pred[idx] == 1:
            pred_types.append(EE_settings.duee_event_types[idx])
    return {
        "gts": pred_types
    }


# train_output_to_loss_format
def train_output_to_loss_format(logits: torch.Tensor):
    return logits


# UseModel


model_registry = {
    'model': EventDetection,
    "loss": EventDetectionLoss,
    "evaluator": AssembledEvaluator,
    'dataset': dataset_factory,
    "sample_to_eval_format": None,
    'eval_output_to_read_format': None,
    "train_output_to_loss_format": None,
    "UseModel": None,
    "args": [
        {'name': '--data_type', 'dest': 'data_type', 'type': str, 'help': '使用的训练集类型'},
        {'name': '--data_dir', 'dest': 'data_dir', 'type': str, 'help': '训练集的路径'}
    ],
    "recorder": None,
}
