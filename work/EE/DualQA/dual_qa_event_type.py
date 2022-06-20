from type_def import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import random
import json
import pickle
from itertools import chain

from utils import tools, tokenize_tools, batch_tool
from utils.data import SimpleDataset
from work.EE.EE_utils import *
from evaluate.evaluator import BaseEvaluator, F1_Evaluator, PrecisionEvaluator
from dataset.ee_dataset import load_FewFC_ee, load_Duee_ee_formated
from analysis.recorder import NaiveRecorder
from work.EE.DualQA import dualqa_utils, dualqa_settings
from work.EE.DualQA.dual_qa import SharedEncoder, SimilarityModel, FlowAttention, SharedProjection
from work.EE.DualQA.dualqa_utils import concat_token_for_evaluate
from models.model_utils import get_init_params


class EventTypeClassifier(nn.Module):
    """
    判断一个句子中有哪些事件类型
    """
    def __init__(self, hidden: int, max_seq_len: int = 256, dataset_type: str = 'FewFC'):
        super(EventTypeClassifier, self).__init__()

        self.max_seq_len = max_seq_len
        self.hidden = hidden

        if dataset_type == 'FewFC':
            self.event_types = EE_settings.event_types_full
        elif dataset_type == 'Duee':
            self.event_types = EE_settings.duee_event_types
        else:
            raise Exception(f'{dataset_type}数据集不存在！')
        event_types_cnt = len(self.event_types)

        self.conv3 = nn.Conv2d(1, 32, (3, hidden))
        self.conv4 = nn.Conv2d(1, 32, (4, hidden))
        self.conv5 = nn.Conv2d(1, 32, (5, hidden))
        self.conv6 = nn.Conv2d(1, 32, (6, hidden))

        self.classifier = nn.Linear(128, event_types_cnt)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0)

    def forward(self, G: torch.Tensor):
        """

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

        G_digits = self.classifier(G_max)  # (bsz, event types cnt)

        return G_digits


class EventTypeIdentifier(nn.Module):
    """
    判断某一个事件类型是否存在于句子中
    """
    def __init__(self, hidden: int, max_seq_len: int = 256):
        super(EventTypeIdentifier, self).__init__()

        self.max_seq_len = max_seq_len
        self.hidden = hidden

        self.conv3 = nn.Conv2d(1, 32, (3, hidden))
        self.conv4 = nn.Conv2d(1, 32, (4, hidden))
        self.conv5 = nn.Conv2d(1, 32, (5, hidden))
        self.conv6 = nn.Conv2d(1, 32, (6, hidden))

        self.classifier = nn.Linear(128, 2)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0)

    def forward(self, G: torch.Tensor):
        """

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

        G_digits = self.classifier(G_max)  # (bsz, 2)

        return G_digits


class DualQA_EventType(nn.Module):
    def __init__(self,
                 plm_path: str = dualqa_settings.plm_path,
                 plm_lr: float = dualqa_settings.plm_lr,
                 linear_lr: float = dualqa_settings.linear_lr,
                 dataset_type: str = 'FewFC'):
        super(DualQA_EventType, self).__init__()
        self.init_params = get_init_params(locals())

        self.plm_path = plm_path
        self.dataset_type = dataset_type
        self.shared_encoder = SharedEncoder(plm_path)
        self.hidden = self.shared_encoder.bert.config.hidden_size
        self.flow_attention = FlowAttention(self.hidden)
        self.shared_projection = SharedProjection(self.hidden)

        self.event_type_classifier = EventTypeClassifier(self.hidden, dataset_type=self.dataset_type)
        self.event_type_identifier = EventTypeIdentifier(self.hidden)

        self.linear_lr = linear_lr
        self.plm_lr = plm_lr

    def forward(self,
                context_input_ids: torch.Tensor,
                context_token_type_ids: torch.Tensor,
                context_attention_mask: torch.Tensor,
                CLS_input_ids: torch.Tensor = None,
                CLS_token_type_ids: torch.Tensor = None,
                CLS_attention_mask: torch.Tensor = None,
                IDF_input_ids: torch.Tensor = None,
                IDF_token_type_ids: torch.Tensor = None,
                IDF_attention_mask: torch.Tensor = None):
        """
        C - 上下文（context）
        CLS - 用于判断句子包含哪些事件类型的询问句
        IDF - 用于判断句子是否包含某个事件类型的询问句

        训练时，同时使用CLS和IDF
        预测时，会根据None判断使用哪一个：
            如果CLS为None，则任务为预测句子中包含哪些事件类型
            如果IDF为None，则任务为预测某个事件类型是否属于句子
        :param context_input_ids:
        :param context_token_type_ids:
        :param context_attention_mask:
        :param CLS_input_ids:
        :param CLS_token_type_ids:
        :param CLS_attention_mask:
        :param IDF_input_ids:
        :param IDF_token_type_ids:
        :param IDF_attention_mask:
        :return:
        """
        if CLS_input_ids is None and IDF_input_ids is None:
            raise Exception('[DualQA_EventType]CLS与IDF不能同时为None')

        event_types_pred, event_types_identified = None, None

        H = self.shared_encoder(
            input_ids=context_input_ids,
            token_type_ids=context_token_type_ids,
            attention_mask=context_attention_mask)

        if CLS_input_ids is not None:
            U_CLS = self.shared_encoder(
                input_ids=CLS_input_ids,
                token_type_ids=CLS_token_type_ids,
                attention_mask=CLS_attention_mask
            )
            # (bsz, Q_CLS, hidden)
            H_CLS_hat, U_CLS_hat = self.flow_attention(H, U_CLS, context_attention_mask, CLS_attention_mask)
            # both (bsz, C, hidden)
            G_CLS = self.shared_projection(H, H_CLS_hat, U_CLS_hat)
            # (bsz, C, hidden)
            event_types_pred = self.event_type_classifier(G_CLS)
            # (bsz, event_types_cnt)

        if IDF_input_ids is not None:
            U_IDF = self.shared_encoder(
                input_ids=IDF_input_ids,
                token_type_ids=IDF_token_type_ids,
                attention_mask=IDF_attention_mask
            )
            # (Q_IDF, hidden)
            H_IDF_hat, U_IDF_hat = self.flow_attention(H, U_IDF, context_attention_mask, IDF_attention_mask)
            # both (bsz, C, hidden)
            G_IDF = self.shared_projection(H, H_IDF_hat, U_IDF_hat)
            # (bsz, C, hidden)
            event_types_identified = self.event_type_identifier(G_IDF)
            # (bsz, 2)

        return {
            "event_types_pred": event_types_pred,  # (bsz, event_types_cnt)
            "event_types_identified": event_types_identified  # (bsz, 2)
        }

    def get_optimizers(self):
        # plm param
        plm_params = self.shared_encoder.parameters()

        # linear param
        flow_params = self.flow_attention.parameters()
        projection_params = self.shared_projection.parameters()

        cls_params = self.event_type_classifier.parameters()
        idf_params = self.event_type_identifier.parameters()

        # optimizers
        plm_optimizer = AdamW(params=plm_params, lr=self.plm_lr)
        linear_optimizer = AdamW(params=chain(flow_params, projection_params, cls_params, idf_params))

        return [plm_optimizer, linear_optimizer]


class DualQA_EventType_Loss(nn.Module):
    CLS_loss_value = 0
    IDF_loss_value = 0

    def __init__(self, lambd: float = 0.5):
        super(DualQA_EventType_Loss, self).__init__()
        self.lambd = lambd

    def forward(self,
                event_types_pred: torch.Tensor,
                event_types_identified: torch.Tensor,
                pred_label: torch.Tensor,
                identified_label: torch.Tensor):
        """

        :param event_types_pred: (bsz, event types cnt)
        :param event_types_identified: (bsz, 2)
        :param pred_label: (bsz, event_types cnt)
        :param identified_label: (bsz)
        :return:
        """
        pred_loss = F.binary_cross_entropy(event_types_pred, pred_label)

        identify_loss = F.cross_entropy(event_types_identified, identified_label)

        loss = self.lambd * pred_loss + (1 - self.lambd) * identify_loss

        return loss


class DualQA_EventType_Evaluator(BaseEvaluator):
    def __init__(self, dataset_type: str = 'FewFC'):
        super(DualQA_EventType_Evaluator, self).__init__()
        self.dataset_type = dataset_type
        if dataset_type == 'FewFC':
            self.event_types = EE_settings.event_types_full
        elif dataset_type == 'Duee':
            self.event_types = EE_settings.duee_event_types
        else:
            raise Exception(f'{dataset_type}数据集不存在！')

        self.classifier_evaluator = F1_Evaluator()
        self.identifier_evaluator = PrecisionEvaluator()

        self.pred_lst = []
        self.gt_lst = []

        self.info_dict = {
            'main': 'f1-measure'
        }

    def eval_single(self, event_types_pred, event_types_identified, cls_gt: StrList, idf_gt: bool):
        """

        :param event_types_pred: (1, event_types_cnt)
        :param event_types_identified: (1, 2)
        :param cls_gt: StrList 包含着当前句子所存在的事件类
        :param idf_gt: bool
        :return:
        """

    def eval_step(self) -> Dict[str, Any]:
        pass


def train_dataset_factory(data_dicts: List[dict], bsz: int = dualqa_settings.default_bsz, shuffle: bool = True, dataset_type: str = 'FewFC'):
    pass


def val_dataset_factory(data_dicts: List[dict], dataset_tyep: str = 'FewFC'):
    pass


def dataset_factory(dataset_type: str, train_file: str, valid_file: str, bsz: int):
    train_data_dicts = pickle.load(open(train_file, 'rb'))
    valid_data_dicts = pickle.load(open(valid_file, 'rb'))
    # valid_data_dicts = list(json.loads(x) for x in open(valid_file, 'r', encoding='utf-8').read().strip().split('\n'))

    train_dataloader = train_dataset_factory(train_data_dicts, bsz=bsz)
    val_dataloader = val_dataset_factory(valid_data_dicts)
    return train_dataloader, val_dataloader


model_registry = {
    'model': DualQA_EventType,
    'loss': DualQA_EventType_Loss,
    'evaluator': DualQA_EventType_Evaluator,
    'train_val_data': dataset_factory,
    'recorder': NaiveRecorder,

}


if __name__ == '__main__':
    pass