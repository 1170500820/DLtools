import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertModel, AutoTokenizer, AutoModel

from type_def import *
from utils.data import SimpleDataset
from work.NewsCorrelation import newsco_settings
from work.NewsCorrelation import newsco_utils
from models.Loss.regular_loss import Scalar_Loss
from evaluate.evaluator import Pearson_Evaluator
from utils import batch_tool, tools, tokenize_tools



class BERT_for_Sentence_Similarity(nn.Module):
    def __init__(self, pretrain_path: str = newsco_settings.pretrain_path, linear_lr: float = newsco_settings.linear_lr, plm_lr: float = newsco_settings.plm_lr):
        """

        :param pretraine_path:
        """
        super(BERT_for_Sentence_Similarity, self).__init__()
        self.linear_lr = linear_lr
        self.plm_lr = plm_lr
        self.bert = AutoModel.from_pretrained(pretrain_path)
        self.hidden = self.bert.config.hidden_size
        self.linear = nn.Linear(self.hidden, 1)

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0)

    def get_optimizers(self):
        bert_params = self.bert.parameters()
        linear_params = self.linear.parameters()
        bert_optimizer = AdamW(params=bert_params, lr=self.plm_lr)
        linear_optimizer = AdamW(params=linear_params, lr=self.linear_lr)
        return [bert_optimizer, linear_optimizer]

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor):
        """

        :param input_ids: (bsz, seq_l)
        :param attention_mask: (bsz, seq_l)
        :return:
        """
        bert_output = self.bert(input_ids, attention_mask)
        pooled_output = bert_output[1]  # (bsz, hidden_size)
        linear_output = self.linear(pooled_output)  # (bsz, 1)
        relu_output = F.relu(linear_output)  # (bsz, 1)
        if self.training:
            return {
                "output": relu_output
            }
        else:
            relu_output = float(relu_output)
            return {
                "pred": relu_output
            }


def dataset_factory(
        crawl_dir: str = newsco_settings.crawl_file_dir,
        newspair_file: str = newsco_settings.newspair_file,
        pretrain_path: str = newsco_settings.pretrain_path,
        bsz: int = newsco_settings.bsz):
    print('[dataset_factory]building samples...', end=' ... ')
    samples = newsco_utils.build_news_samples(crawl_dir, newspair_file)
    # language_pair, id1, id2, crawl1, crawl2, Overall, ...

    # 划分训练、评价
    train_samples, val_samples = batch_tool.train_val_split(samples, newsco_settings.train_ratio)
    print('finish')

    print('[dataset_factory]preparing data for training...', end=' ... ')
    train_dataloader = train_dataset_factory(train_samples, pretrain_path, bsz)
    print('finish')

    print('[dataset_factory]preparing data for evaluating...', end=' ... ')
    val_dataloader = eval_dataset_factory(val_samples, pretrain_path)
    print('finish')

    return train_dataloader, val_dataloader


def convert_parsed_sample_to_model_sample(samples, plm_path) -> List[Dict[str, Any]]:
    """
    将utils中从文件读取并合并的样本，转化为适合模型输入处理的样本
    :param samples:
    :param plm_path:
    :return:
    """
    # 首先提取出train所需的数据
    sample_dict = tools.transpose_list_of_dict(samples)
    crawl1s, crawl2s = sample_dict['crawl1'], sample_dict['crawl2']
    crawl1_dict, crawl2_dict = tools.transpose_list_of_dict(crawl1s), tools.transpose_list_of_dict(crawl2s)

    scores = list(float(x) for x in sample_dict['Overall'])  # List[float]
    title1_lst, title2_lst, text1_lst, text2_lst = \
        crawl1_dict['title'], crawl2_dict['title'], crawl1_dict['text'], crawl2_dict['text']

    title_tokenizer = tokenize_tools.xlmr_tokenizer(max_len=newsco_settings.title_len, plm_path=plm_path)
    text_tokenizer = tokenize_tools.xlmr_tokenizer(max_len=newsco_settings.text_len, plm_path=plm_path)
    title1_result = tools.transpose_list_of_dict(title_tokenizer(title1_lst))  # Dict[str, list]
    title2_result = tools.transpose_list_of_dict(title_tokenizer(title2_lst))  # Dict[str, list]
    text1_result = tools.transpose_list_of_dict(text_tokenizer(text1_lst))  # Dict[str, list]
    text2_result = tools.transpose_list_of_dict(text_tokenizer(text2_lst))  # Dict[str, list]

    tokenize_result = {
        'title1_input_ids': title1_result['input_ids'],
        'title2_input_ids': title2_result['input_ids'],
        'text1_input_ids': text1_result['input_ids'],
        'text2_input_ids': text2_result['input_ids'],
    }
    tokenize_result_dicts = tools.transpose_dict_of_list(tokenize_result)  # List[dict]
    concat_results = list(newsco_utils.xlmr_sentence_concat_cza(
        [x['title1_input_ids'], x['text1_input_ids'], x['title2_input_ids'], x['text2_input_ids']]
    ) for x in tokenize_result_dicts)  # List[IntList] list of concatenated input_ids
    attention_masks = list(newsco_utils.generate_attention_mask(x, newsco_settings.max_seq_length) for x in concat_results)
    padded_input_ids = list(newsco_utils.pad_input_ids(x, newsco_settings.max_seq_length) for x in concat_results)

    train_samples = {
        "label": scores,  # List[float]
        "input_ids": padded_input_ids,  # List[IntList]
        "attention_mask": attention_masks  # List[IntList]
    }
    train_samples = tools.transpose_dict_of_list(train_samples)
    return train_samples


def train_dataset_factory(samples: List[Dict[str, Any]], pretrain_path: str, bsz: int):
    """

    :param samples:
    :return:
    """
    # 首先提取出train所需的数据
    train_samples = convert_parsed_sample_to_model_sample(samples, pretrain_path)
    train_dataset = SimpleDataset(train_samples)

    def collate_fn(lst):
        """
        :param lst: [
            {
            "label",
            "input_ids",
            "attention_mask",
            }
        ]
        :return:
        """
        label = torch.Tensor(list(x['label'] for x in lst))  # (bsz, 1)
        input_ids = torch.tensor(list(x['input_ids'] for x in lst), dtype=torch.int)  # (bsz, max_seq_l)
        attention_mask = torch.tensor(list(x['attention_mask'] for x in lst), dtype=torch.int)  # (bsz, max_seq_l)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }, {
            "label": label
        }
    train_dataloader = DataLoader(train_dataset, batch_size=bsz, shuffle=True, collate_fn=collate_fn)
    return train_dataloader


def eval_dataset_factory(samples: List[Dict[str, Any]], pretrain_path: str):
    """

    :param samples:
    :return:
    """
    val_samples = convert_parsed_sample_to_model_sample(samples, pretrain_path)
    # label, input_ids, attention_mask
    val_dataset = SimpleDataset(val_samples)
    def collate_fn(lst):
        target = lst[0]['label']
        input_ids = torch.tensor(list(x['input_ids'] for x in lst), dtype=torch.int)  # (1, max_seq_l)
        attention_mask = torch.tensor(list(x['attention_mask'] for x in lst), dtype=torch.int)  # (1, max_seq_l)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
               }, {
            "gt": target
        }
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return val_dataloader


model_registry = {
    "model": BERT_for_Sentence_Similarity,
    "evaluator": Pearson_Evaluator,
    "loss": Scalar_Loss,
    "train_data": train_dataset_factory,
    "val_data": eval_dataset_factory,
    'train_val_data': dataset_factory
}

