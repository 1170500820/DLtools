import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from evaluate.evaluator import Pearson_Evaluator

"""
todo 改造
模型基本结构：
    - 使用新闻标题、新闻正文以及新闻描述
    - 分别通过词向量转化为表示矩阵 (seq_l, embedding)
"""


class cnnModel(nn.Module):
    def __init__(self, config):
        """

        :param config:
            外部
            - word_embbed_path? word_embedding_path 词向量文件的路径

            输入
            - title_len
            - text_len
            - meta_descrip_len

            网络结构
            - conv_output_channels
            - att_key_dim
            - layer_size
        todo 改为直接传入
        """
        super(cnnModel, self).__init__()
        self.config = config
        wordEmb = np.load(self.config['word_embbed_path'])
        # self.wordEmbeddingLayer = nn.Embedding(wordEmb.shape[0], wordEmb.shape[1])  # TODO 初始化

        wordEmb = torch.FloatTensor(wordEmb)
        self.wordEmbeddingLayer = nn.Embedding.from_pretrained(wordEmb)
        # self.wordEmbeddingLayer.requires_grad_(requires_grad=False)

        self.newstitleEncoder = cnnNewsTextEncoder(config, self.config['title_len'])
        self.newstextEncoder = cnnNewsTextEncoder(config, self.config['text_len'])
        self.newsdesEncoder = cnnNewsTextEncoder(config, self.config['meta_descrip_len'])

        self.MutiViewAttention = AttentionBase(self.config['conv_output_channels'] * 3, self.config['att_key_dim'])

        self.title_emb_size = 6 * self.config['conv_output_channels']
        self.fc1 = torch.nn.Linear(in_features=self.title_emb_size, out_features=self.config['layer_size'], bias=True)
        self.fc2 = torch.nn.Linear(in_features=self.config['layer_size'], out_features=1, bias=True)

    def forward(self, x):
        """
        news1_title_batch: (bsz, title_len)
        news2_title_batch: (bsz, title_len)
        news1_text_batch:  (bsz, text_len)
        news2_text_batch: (bsz, text_len)
        news1_description_batch:  (bsz, descrip_len)
        news2_description_batch:
        label: (batch_size,)
               [0,1,...]
        :param x:
        :return:
        """
        news1_title_batch, news2_title_batch, news1_text_batch, news2_text_batch, \
            news1_description_batch, news2_description_batch = x

        # 经过词向量层，转变为
        # (batch_size, title_len, word_embedding)
        news1_title_batch = self.wordEmbeddingLayer(news1_title_batch)
        news2_title_batch = self.wordEmbeddingLayer(news2_title_batch)

        news1_text_batch = self.wordEmbeddingLayer(news1_text_batch)
        news2_text_batch = self.wordEmbeddingLayer(news2_text_batch)
        news1_description_batch = self.wordEmbeddingLayer(news1_description_batch)
        news2_description_batch = self.wordEmbeddingLayer(news2_description_batch)

        # (batch_size, conv_output_channels * 3)
        news1titlerep = self.newstitleEncoder(news1_title_batch)
        news1textrep = self.newstextEncoder(news1_text_batch)
        news1descriptionbatch = self.newsdesEncoder(news1_description_batch)

        news1_rep = torch.cat((news1titlerep, news1textrep, news1descriptionbatch), dim=-1)
        news1_rep = torch.reshape(news1_rep, shape=(news1_rep.shape[0], 3, self.config['conv_output_channels'] * 3))
        news1_rep = self.MutiViewAttention(news1_rep)

        news2titlerep = self.newstitleEncoder(news2_title_batch)
        news2textrep = self.newstextEncoder(news2_text_batch)
        news2descriptionbatch = self.newsdesEncoder(news2_description_batch)

        news2_rep = torch.cat((news2titlerep, news2textrep, news2descriptionbatch), dim=-1)
        news2_rep = torch.reshape(news2_rep, shape=(news2_rep.shape[0], 3, self.config['conv_output_channels'] * 3))
        news2_rep = self.MutiViewAttention(news2_rep)

        # (batch_size, conv_output_channels * 6)
        news_concat = torch.cat((news1_rep, news2_rep), dim=1)

        # (batch_size, layer_size)
        news_concat = self.fc1(news_concat)
        news_concat = torch.nn.Sigmoid()(news_concat)
        news_similarity = self.fc2(news_concat)

        # # (batch_size, layer_size)
        # news_concat = self.fc1(news_concat)
        # news_concat = torch.nn.LeakyReLU()(news_concat)
        # news_similarity = self.fc2(news_concat)
        # news_similarity = torch.nn.Sigmoid()(news_similarity)
        return news_similarity


# 注意力机制层
class AttentionBase(nn.Module):
    # 在__init__构造函数中会执行变量的初始化、参数初始化、子网络初始化的操作
    def __init__(self, in_dim, att_dim):
        super(AttentionBase, self).__init__()
        self.in_dim = in_dim
        self.att_dim = att_dim

        self.dense = torch.nn.Linear(in_features=self.in_dim, out_features=self.att_dim, bias=True)
        nn.init.xavier_uniform_(self.dense.weight)
        self.query_dense = torch.nn.Linear(in_features=att_dim, out_features=1, bias=False)
        nn.init.xavier_uniform_(self.query_dense.weight)

        # 保存中间结果
        self.attention_weight = None

    # forward函数实现了 TitleEncoder 网络的执行逻辑
    def forward(self, inputs):
        """
        使用论文中的attention机制，对word_number加权求和
        输入：[batch_size, word_number, word_dim]
        输出：[batch_size, word_dim]
        """
        attention = self.dense(inputs)
        attention = torch.tanh(attention)
        attention = self.query_dense(attention)
        attention = torch.exp(attention)
        attention = torch.squeeze(attention, 2)  # TODO
        attention_weight = attention / torch.sum(attention, dim=-1, keepdim=True)
        self.attention_weight = attention_weight  # 保存中间结果用于可视化
        attention_weight = torch.unsqueeze(attention_weight, 2)
        weighted_input = torch.multiply(inputs, attention_weight)
        attention_output = torch.sum(weighted_input, 1)
        return attention_output


class cnnAndpool(nn.Module):

    def __init__(self, config, filter_height, len_):
        super().__init__()
        self.config = config
        self.conv = nn.Conv2d(in_channels=1, out_channels=self.config['conv_output_channels'],
                              kernel_size=(filter_height, self.config['word_embedd_size']), stride=1,
                              padding=0)  # 输入为N,C,H,W
        self.maxpool = nn.MaxPool2d(kernel_size=(len_ - filter_height + 1, 1), stride=1, padding=0)

    def forward(self, x):
        """
        x (news_title_batch): (batch_size, title_len, word_embed_dim)
                           [[[word11_embedding],[word12_embedding],[word13_embedding],..],
                            [[word21_embedding],[word22_embedding],[word23_embedding],..],
                            ...]
        :return:
        """
        # (batch_size, title_len, word_embed_dim)

        # 转变为(batch_size, 1, title_len, word_embed_dim)
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))

        # 经过卷积
        # (batch_size, conv_output_channels, title_len - filter_height + 1, 1)
        x = self.conv(x)

        # relu为正值
        x = nn.ReLU()(x)

        # 经过最大池化
        # (batch_size, conv_output_channels,1,1)
        x = self.maxpool(x)

        # (batch_size, conv_output_channels)
        return torch.reshape(x, (x.shape[0], x.shape[1]))


class cnnNewsTextEncoder(nn.Module):
    def __init__(self, config, textlen):
        super().__init__()
        self.config = config
        self.convAndpool1 = cnnAndpool(config, 1, textlen)
        self.convAndpool2 = cnnAndpool(config, 2, textlen)
        self.convAndpool3 = cnnAndpool(config, 3, textlen)

    def forward(self, x):
        """
        x (news_title_batch): (batch_size, title_len, word_embed_dim)
                           [[[word11_embedding],[word12_embedding],[word13_embedding],..],
                            [[word21_embedding],[word22_embedding],[word23_embedding],..],
                            ...]
        :return:
        """
        # (batch_size, title_len, word_embed_dim)

        # (batch_size, conv_output_channels)
        newsemb1 = self.convAndpool1(x)
        newsemb2 = self.convAndpool2(x)
        newsemb3 = self.convAndpool3(x)

        # (batch_size, conv_output_channels * 3)
        return torch.cat((newsemb1, newsemb2, newsemb3), dim=1)


model_registry = {
    "model": cnnModel,
    'evaluator': Pearson_Evaluator
}