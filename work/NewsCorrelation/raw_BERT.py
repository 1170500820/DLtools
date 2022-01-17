import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertModel, AutoTokenizer, AutoModel

from type_def import *
from utils.data import SimpleDataset
import newsco_settings
import newsco_utils
from models.Loss.regular_loss import Scalar_Loss
from evaluate.evaluator import Pearson_Evaluator
from utils import batch_tool, tools



class BERT_for_Sentence_Similarity(nn.Module):
    def __init__(self, pretrained_path: str):
        """

        :param pretrained_path:
        """
        super(BERT_for_Sentence_Similarity, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_path)
        self.hidden = self.bert.config.hidden_size
        self.linear = nn.Linear(self.hidden, 1)

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0)

    def forward(self,
                input_ids: torch.Tensor,
                token_type_ids: torch.Tensor,
                attention_mask: torch.Tensor):
        """

        :param input_ids: (bsz, seq_l)
        :param token_type_ids: (bsz, seq_l)
        :param attention_mask: (bsz, seq_l)
        :return:
        """
        bert_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = bert_output[1]  # (bsz, hidden_size)
        linear_output = self.linear(pooled_output)  # (bsz, 1)
        relu_output = F.relu(linear_output)  # (bsz, 1)
        return {
            "output": relu_output
        }


def dataset_factory(crawl_dir: str = newsco_settings.crawl_file_dir, newspair_file: str = newsco_settings.newspair_file, pretrain_path: str = newsco_settings.pretrain_path):
    samples = newsco_utils.build_news_samples(crawl_dir, newspair_file)
    # language_pair, id1, id2, crawl1, crawl2, Overall, ...

    # 划分训练、评价
    train_samples, val_samples = batch_tool.train_val_split(samples, newsco_settings.train_ratio)

    train_dataloader = train_dataset_factory(train_samples, pretrain_path)
    val_dataloader = eval_dataset_factory(val_samples, pretrain_path)


def train_dataset_factory(samples: List[Dict[str, Any]], pretrain_path: str):
    """

    :param samples:
    :return:
    """
    # 首先提取出train所需的数据
    sample_dict = tools.transpose_list_of_dict(samples)
    crawl1s, crawl2s = sample_dict['crawl1'], sample_dict['crawl2']
    crawl1_dict, crawl2_dict = tools.transpose_list_of_dict(crawl1s), tools.transpose_list_of_dict(crawl2s)

    scores = list(float(x) for x in sample_dict['Overall'])  # List[float]
    title1_lst, title2_lst, text1_lst, text2_lst, des1_lst, des2_lst = \
        crawl1_dict['title'], crawl2_dict['title'], crawl1_dict['text'], crawl2_dict['text'], crawl1_dict['meta_description'], crawl2_dict['meta_description']

    tokenizer = AutoTokenizer.from_pretrained(pretrain_path)


    def collate_fn(lst):
        pass


def eval_dataset_factory(samples: List[Dict[str, Any]], pretrain_path: str):
    """

    :param samples:
    :return:
    """

    def collate_fn(lst):
        pass


model_registry = {
    "model": BERT_for_Sentence_Similarity,
    "evaluator": Pearson_Evaluator,
    "loss": Scalar_Loss,
    "train_data": train_dataset_factory,
    "val_data": eval_dataset_factory
}

