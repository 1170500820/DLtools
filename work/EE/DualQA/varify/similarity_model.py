import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarityModel(nn.Module):
    def __init__(self, hidden: int):
        super(SimilarityModel, self).__init__()
        self.hidden = hidden

    def forward(self, h: torch.Tensor, u: torch.Tensor):
        """

        :param h: (bsz, |C|, hidden)
        :param u: (bsz, |Q|, hidden)
        :return:
        """
        C, Q = h.shape[1], u.shape[1]
        h1 = h.unsqueeze(dim=2)  # (bsz, |C|, 1, hidden)
        u1 = u.unsqueeze(dim=1)  # (bsz, 1, |Q|, hidden)
        h2 = h1.expand(-1, -1, Q, -1)  # (bsz, |C|, |Q|, hidden)
        u2 = u1.expand(-1, C, -1, -1)  # (bsz, |C|, |Q|, hidden)
        concatenated = torch.cat([h2, u2, h2 * u2], dim=-1)  # (bsz, |C|, |Q|, 3 * hidden)
        return concatenated  # (bsz, |C|, |Q|)
        # concatenated = torch.cat([h2, u2, h2 * u2], dim=-1)  # (bsz, |C|, |Q|, 3 * hidden)
        # l1_output = F.relu(self.l1_sim(concatenated))  # (bsz, |C|, |Q|, hidden)
        # l2_output = F.relu(self.l2_sim(l1_output))  # (bsz, |C|, |Q|, 1)
        # l2_output = l2_output.squeeze()  # (bsz, |C|, |Q|) or (|C|, |Q|) if bsz == 1
        # if len(l2_output.shape) == 2:
        #     l2_output = l2_output.unsqueeze(dim=0)  # (1, |C|, |Q|)
        # return l2_output  # (bsz, |C|, |Q|)


if __name__ == '__main__':
    C = torch.tensor([[1, 1], [2, 2]])
    H = torch.tensor([[3, 3], [4, 4]])
    C = C.unsqueeze(0)
    H = H.unsqueeze(0)
    smodel = SimilarityModel(2)
    concatenated = smodel(C, H)
