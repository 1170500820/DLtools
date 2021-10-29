import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalLayerNormalization(nn.Module):
    def __init__(self, hidden_size, eps=1e-15):
        """
        todo 把paper或者网站的地址放在这里。要参考的。
        :param hidden_size:
        :param eps:
        """
        super(ConditionalLayerNormalization, self).__init__()
        self.weight_map = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias_map = nn.Linear(hidden_size, hidden_size, bias=False)
        self.t = nn.Parameter(torch.tensor(hidden_size, dtype=torch.float))
        self.bias = nn.Parameter(torch.tensor(hidden_size, dtype=torch.float))

        self.hidden_size = hidden_size
        self.eps = eps
        self.reset_weight_and_bias()

    def reset_weight_and_bias(self):
        """
        初始化的作用是在训练的开始阶段不让CLN起作用
        :return:
        """
        nn.init.ones_(self.t)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.weight_map.weight)
        nn.init.zeros_(self.bias_map.weight)

    def forward(self, representation, condition):
        """
        稍微修改了一下repr_mean和repr_var的计算方法，现在对于非batch的，即rep (seq_l, hidden), cond (1, hidden)的情况，也能正常算
        :param representation: (bsz, seq_l, hidden_size), sentence representation that does not contain <CLS> and <SEP>
        :param condition: (bsz, 1, hidden_size), pooled type embedding
        :return: denormed_repr (bsz, seq_l, hidden_size)
        """
        weight = self.weight_map(condition) + self.t  # (bsz, 1, hidden_size)
        bias = self.bias_map(condition) + self.bias  # (bsz, 1, hidden_size)
        # (bsz, 1, hidden) or (bsz, hidden)
        # normed_repr = F.layer_norm(representation, [self.hidden_size], weight, bias) # RuntimeError
        repr_mean = torch.mean(representation, dim=-1, keepdim=True)   # (bsz, 1, hidden)
        repr_var = torch.var(representation, dim=-1, unbiased=False, keepdim=True)  # (bsz, 1, hidden)
        normed_repr = (representation - repr_mean) / torch.sqrt(repr_var + self.eps)  # (bsz, seq_l, hidden)
        denormed_repr = torch.multiply(weight, normed_repr) + bias # weight and bias (bsz, 1, hidden_size)

        # 这个CLN可以手动实现吗
        return denormed_repr
