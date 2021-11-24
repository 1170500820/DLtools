import torch
import torch.nn as nn
import torch.nn.functional as F

from work.NER import NER_settings

from type_def import *
from evaluate.evaluator import BaseEvaluator, F1_Evaluator
from work.NER import NER_utils, NER_settings


class BiLSTM_NER(nn.Module):
    def __init__(self,
                 ner_cnt: int = len(NER_settings.msra_ner_tags),
                 lr: float = NER_settings.others_lr,
                 input_size: int = NER_settings.embedding_dim,
                 hidden_size: int = NER_settings.hidden_size,
                 num_layers: int = NER_settings.num_layers):
        super(BiLSTM_NER, self).__init__()

        # 保存初始化参数
        self.ner_cnt = ner_cnt
        self.lr = lr
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 初始化Bi-LSTM结构
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True)
        self.linear = nn.Linear(2 * hidden_size, ner_cnt)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0)

    def forward(self, input: torch.Tensor):
        """

        :param input: (bsz, seq_l, embedding_dim)
        :return:
        """
        output, (_, _) = self.bilstm(input)  # (bsz, seq_l, 2 * hidden_size)
        output = F.relu(output)  # (bsz, seq_l, 2 * hidden_size)
        output = self.linear(output)  # (bsz, seq_l, ner_cnt)
        output = torch.softmax(output, dim=-1)  # (bsz, seq_l, ner_cnt)
        return {
            "ner_result": output  # (bsz, seq_l, ner_cnt)
        }


class BiLSTM_NER_Loss(nn.Module):
    def __init__(self):
        super(BiLSTM_NER_Loss, self).__init__()

    def forward(self, ner_result: torch.Tensor, ner_target: torch.Tensor):
        """

        :param ner_result: (bsz, seq_l, ner_cnt)
        :param ner_target: (bsz, seq_l)
        :return:
        """
        bsz, seq_l, ner_cnt = ner_result.shape
        ner_result = ner_result.view(bsz * seq_l, ner_cnt)
        ner_target = ner_target.view(bsz * seq_l)
        ner_loss = F.cross_entropy(ner_result, ner_target)

        return ner_loss


class BiLSTM_NER_Evaluator(BaseEvaluator):
    def __init__(self):
        super(BiLSTM_NER_Evaluator, self).__init__()
        self.f1_evaluator = F1_Evaluator()

    def eval_single(self,
                    ner_result: torch.Tensor,
                    ner_gt: List[str]):
        """

        :param ner_result:
        :param ner_gt:
        :return:
        """
        ner_result = ner_result.squeeze(dim=0)  # (seq_l, ner_cnt)
        ner_tag_result = NER_utils.tensor_to_ner_label(ner_result)  # List[str]
        self.f1_evaluator.eval_single(ner_tag_result, ner_gt)

    def eval_step(self) -> Dict[str, Any]:
        sf1 = self.f1_evaluator.eval_step()
        return sf1



