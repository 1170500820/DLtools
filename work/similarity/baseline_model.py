import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel
from evaluate.evaluator import BaseEvaluator
from process.typed_processor_utils import *
from utils.data import SimpleDataset
from type_def import *
from torch.utils.data import DataLoader


def dataset_factory_train(filepath: str, bsz: int = 4, shuffle: bool = True):
    p = TSV_Reader() + (ListMapper(
        lambda x: x[:2]) + BERT_Tokenizer_double() + ListOfDictTranspose() + ReleaseDict()) * ListMapper(
        lambda x: int(x[-1])) + Dict2List(['mapped', 'input_ids', 'token_type_ids', 'attention_mask'])
    result = p(filepath)
    result = list(map(remove_type, list(result.values())[0]))
    dataset = SimpleDataset(result)

    def collate_fn(lst):
        collate_result = simple_collate_fn(lst)
        labels = torch.tensor(collate_result['mapped'], dtype=torch.float).unsqueeze(dim=1)
        input_ids_lst = collate_result['input_ids']
        input_ids_tensors = list(map(torch.tensor, input_ids_lst))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_tensors, batch_first=True, padding_value=0)
        token_type_ids_lst = collate_result['token_type_ids']
        token_type_ids_tensors = list(map(torch.tensor, token_type_ids_lst))
        token_type_ids = torch.nn.utils.rnn.pad_sequence(token_type_ids_tensors, batch_first=True, padding_value=0)
        attention_mask_lst = collate_result['attention_mask']
        attention_mask_tensors = list(map(torch.tensor, attention_mask_lst))
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_tensors, batch_first=True, padding_value=0)
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask
        }, {
            "label": labels
               }
    return DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)


def dataset_factory_val(filepath: str):
    return dataset_factory_train(filepath, bsz=1, shuffle=False)


class BertSimilarity(nn.Module):
    def __init__(self):
        super(BertSimilarity, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.hidden = self.bert.config.hidden_size
        self.linear = nn.Linear(self.hidden, 1)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0)

    def get_optimizers(self):
        return [AdamW(self.parameters(), lr=3e-5)]

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        """

        :param input_ids: (bsz, d_seq_l)
        :param token_type_ids: (bsz, d_seq_l)
        :param attention_mask: (bsz, d_seq_l)
        :return:
        """
        result = self.bert(input_ids, token_type_ids, attention_mask)[1]  # (1, hidden)
        result = self.linear(result.squeeze())  # (1)
        result = F.sigmoid(result)  # (1)
        return {
            "output": result  # (1)
        }


class BertSimilarityLoss(nn.Module):
    def __init__(self):
        super(BertSimilarityLoss, self).__init__()

    def forward(self, output: torch.Tensor = None, label: torch.Tensor = None):
        """

        :param output:
        :param label:
        :return:
        """
        loss = F.binary_cross_entropy(output, label)
        return loss


class BertSimilarityEvaluator(BaseEvaluator):
    def __init__(self):
        super(BertSimilarityEvaluator, self).__init__()
        self.output_lst, self.label_lst = [], []

    def eval_single(self, output: torch.Tensor = None, label: torch.Tensor = None):
        self.output_lst.append(float(output))
        self.label_lst.append(float(label))

    def eval_step(self) -> str:
        outputs = np.array(self.output_lst)
        labels = np.array(self.label_lst)
        acc = sum(outputs == labels) / len(outputs)
        return f'acc:{acc:<7.5f}'


model_registry = {
    "model": BertSimilarity,
    "args": [],  # List[Args]
    "loss": BertSimilarityLoss,
    "evaluator": BertSimilarityEvaluator,
    "train_data": dataset_factory_train,
    "val_data": dataset_factory_val
}

# dataloader = dataset_factory_train('../../data/NLP/Similarity/bq_corpus/train.tsv')
