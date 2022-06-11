import torch
import torch.nn as nn
import torch.nn.functional as F


from . import EE_settings
import itertools


class TriggerExtractionLayer_woSyntactic(nn.Module):
    """
    初始化参数:
        - n_heads
        - hidden_size
        - d_head
        - dropout_prob
        - syntactic_size

    输入:
        cln_embeds: tensor (bsz, seq_l, hidden_size)

    输出:
        starts: tensor (bsz, seq_l)
        ends: tensor (bsz, seq_l)
    """
    def __init__(self, num_heads, hidden_size, d_head, dropout_prob):
        super(TriggerExtractionLayer_woSyntactic, self).__init__()
        # store params
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.d_head = d_head
        self.dropout_prob = dropout_prob

        # initiate network structures
        #   self-attention
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout_prob, batch_first=True)
        #   lstm
        self.lstm = nn.LSTM(self.hidden_size * 2, self.hidden_size // 2,
                            dropout=self.dropout_prob, bidirectional=True, batch_first=True)
        # FCN for finding triggers
        self.fcn_start = nn.Linear(self.hidden_size, 1)
        self.fcn_end = nn.Linear(self.hidden_size, 1)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.fcn_end.weight)
        self.fcn_end.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.fcn_start.weight)
        self.fcn_start.bias.data.fill_(0)

    def forward(self, cln_embeds: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        注意bsz=1的情况
        :param cln_embeds: (bsz, seq_l, hidden_size)， 经过CLN处理的句子embeddings
        :return:
        """
        # self attention (multihead attention)
        attn_out, attn_out_weights = self.self_attn(cln_embeds, cln_embeds, cln_embeds)
        # todo attn_out: (bsz, seq_l, hidden) ?

        # concatenation
        final_repr = torch.cat((cln_embeds, attn_out), dim=-1)
        # final_repr: (bsz, seq_l, hidden * 2)

        lstm_out, (_, __) = self.lstm(final_repr)
        # lstm_out: (bsz, seq_l, hidden)

        # linear
        start_logits, end_logits = self.fcn_start(lstm_out).squeeze(), self.fcn_end(lstm_out).squeeze()
        # ot both (bsz, seq_l, 1), convert to (bsz, seq_l)

        # 需要保证batch维存在
        if len(start_logits.shape) == 1:
            start_logits = start_logits.unsqueeze(dim=0)
            end_logits = end_logits.unsqueeze(dim=0)
        # sigmoid
        starts, ends = F.sigmoid(start_logits).unsqueeze(dim=-1), F.sigmoid(end_logits).unsqueeze(dim=-1)  # got both (bsz, seq_l)
        return starts, ends
