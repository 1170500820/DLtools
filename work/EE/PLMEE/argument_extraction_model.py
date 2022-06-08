import torch
import torch.nn as nn
import torch.nn.functional as F

from . import EE_settings


class ArgumentExtractionModel_woSyntactic(nn.Module):
    def __init__(self, n_head, d_head, hidden_size, dropout_prob, dataset_type: str = 'FewFC'):
        """

        :param n_head:
        :param hidden_size:
        :param d_head:True
        :param dropout_prob:
        """
        super(ArgumentExtractionModel_woSyntactic, self).__init__()

        if dataset_type == 'FewFC':
            self.role_types = EE_settings.role_types
        elif dataset_type == 'Duee':
            self.role_types = EE_settings.duee_role_types
        else:
            raise Exception(f'{dataset_type}数据集不存在！')
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.d_head = d_head
        self.dropout_prob = dropout_prob

        # initiate network structures
        #   self-attention
        self.self_attn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.n_head, dropout=self.dropout_prob)
        #   FCN for finding triggers
        self.fcn_start = nn.Linear(self.hidden_size, len(self.role_types))
        self.fcn_end = nn.Linear(self.hidden_size, len(self.role_types))
        #   LSTM
        self.lstm = nn.LSTM(self.hidden_size * 2 + 1, self.hidden_size//2,
                            batch_first=True, dropout=self.dropout_prob, bidirectional=True)

        self.init_weights()

        # code for not adding LSTM layer
        # FCN for trigger finding
        # origin + attn(origin) + syntactic + RPE
        # self.fcn_start = nn.Linear(self.hidden_size * 2 + ltp_embedding_dim + 1, len(role_types))
        # self.fcn_end = nn.Linear(self.hidden_size * 2 + ltp_embedding_dim + 1, len(role_types))

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.fcn_end.weight)
        self.fcn_end.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.fcn_start.weight)
        self.fcn_start.bias.data.fill_(0)


    def forward(self, cln_embeds, relative_positional_encoding):
        """

        :param cln_embeds: (bsz, seq_l, hidden_size)
        :param relative_positional_encoding: (bsz, seq_l, 1) todo 无效区域的距离设为inf还是0
        :return:
        """
        # self attention (multihead attention)
        attn_out, attn_out_weights = self.self_attn(cln_embeds, cln_embeds, cln_embeds)
        # attn_out: (bsz, seq_l, hidden)

        # concatenation
        final_repr = torch.cat((cln_embeds, attn_out, relative_positional_encoding), dim=-1)
        # final_repr: (bsz, seq_l, 2 * hidden + 1)

        lstm_out, (_, __) = self.lstm(final_repr)
        # lstm_out: (bsz, seq_l, hidden)

        start_logits, end_logits = self.fcn_start(lstm_out), self.fcn_end(lstm_out)
        # start_logits and end_logits: (bsz, seq_l, len(role_types))

        starts, ends = F.sigmoid(start_logits), F.sigmoid(end_logits)
        # starts and ends: (bsz, seq_l, len(role_types))
        return starts, ends


class ArgumentDetectionModel_woSyntactic(nn.Module):
    """
    只预测论元存在，不预测具体类型
    只是更改了线性层的输出纬度
    """
    def __init__(self, n_head, d_head, hidden_size, dropout_prob):
        """

        :param n_head:
        :param hidden_size:
        :param d_head:True
        :param dropout_prob:
        """
        super(ArgumentDetectionModel_woSyntactic, self).__init__()

        self.n_head = n_head
        self.hidden_size = hidden_size
        self.d_head = d_head
        self.dropout_prob = dropout_prob

        # initiate network structures
        #   self-attention
        self.self_attn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.n_head, dropout=self.dropout_prob)
        #   FCN for finding triggers
        self.fcn_start = nn.Linear(self.hidden_size, 1)
        self.fcn_end = nn.Linear(self.hidden_size, 1)
        #   LSTM
        self.lstm = nn.LSTM(self.hidden_size * 2 + 1, self.hidden_size//2,
                            batch_first=True, dropout=self.dropout_prob, bidirectional=True)

        self.init_weights()

        # code for not adding LSTM layer
        # FCN for trigger finding
        # origin + attn(origin) + syntactic + RPE
        # self.fcn_start = nn.Linear(self.hidden_size * 2 + ltp_embedding_dim + 1, len(role_types))
        # self.fcn_end = nn.Linear(self.hidden_size * 2 + ltp_embedding_dim + 1, len(role_types))

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.fcn_end.weight)
        self.fcn_end.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.fcn_start.weight)
        self.fcn_start.bias.data.fill_(0)

    def forward(self, cln_embeds, relative_positional_encoding):
        """

        :param cln_embeds: (bsz, seq_l, hidden_size)
        :param relative_positional_encoding: (bsz, seq_l, 1) todo 无效区域的距离设为inf还是0
        :return:
        """
        # self attention (multihead attention)
        attn_out, attn_out_weights = self.self_attn(cln_embeds, cln_embeds, cln_embeds)
        # attn_out: (bsz, seq_l, hidden)

        # concatenation
        final_repr = torch.cat((cln_embeds, attn_out, relative_positional_encoding), dim=-1)
        # final_repr: (bsz, seq_l, 2 * hidden + 1)

        lstm_out, (_, __) = self.lstm(final_repr)
        # lstm_out: (bsz, seq_l, hidden)

        start_logits, end_logits = self.fcn_start(lstm_out), self.fcn_end(lstm_out)
        # start_logits and end_logits: (bsz, seq_l, 1)

        starts, ends = F.sigmoid(start_logits), F.sigmoid(end_logits)
        # starts and ends: (bsz, seq_l, 1)
        return starts, ends