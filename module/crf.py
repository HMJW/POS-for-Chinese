import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import *


class CRFlayer(torch.nn.Module):
    def __init__(self, labels_num):
        super(CRFlayer, self).__init__()
        self.labels_num = labels_num
        self.transitions = torch.nn.Parameter(torch.randn(labels_num, labels_num))
              # (i,j)->p(tag[i]->tag[j])
        # 句首迁移
        self.strans = torch.nn.Parameter(torch.randn(labels_num))
        # 句尾迁移
        self.etrans = torch.nn.Parameter(torch.randn(labels_num))
        self.reset_parameters()

    def reset_parameters(self):
        # self.transitions.data.zero_()
        # self.etrans.data.zero_()
        # self.strans.data.zero_()
        init.normal_(self.transitions.data, 0, 1 / self.labels_num ** 0.5)
        init.normal_(self.strans.data, 0, 1 / self.labels_num ** 0.5)
        init.normal_(self.etrans.data, 0, 1 / self.labels_num ** 0.5)
        # bias = (6. / self.labels_num) ** 0.5
        # nn.init.uniform_(self.transitions, -bias, bias)
        # nn.init.uniform_(self.strans, -bias, bias)
        # nn.init.uniform_(self.etrans, -bias, bias)

    def get_logZ(self, emit, mask):
        T, B, N = emit.shape
        alpha = emit[0] + self.strans  # [B, N]
        for i in range(1, T):
            trans_i = self.transitions.unsqueeze(0)  # [1, N, N]
            emit_i = emit[i].unsqueeze(1)  # [B, 1, N]
            mask_i = mask[i].unsqueeze(1).expand_as(alpha)  # [B, N]
            scores = trans_i + emit_i + alpha.unsqueeze(2)  # [B, N, N]
            scores = torch.logsumexp(scores, dim=1)  # [B, N]
            alpha[mask_i] = scores[mask_i]

        return torch.logsumexp(alpha+self.etrans, dim=1).sum()

    def score(self, emit, target, mask):
        'written by zhangyu'
        T, B, N = emit.shape
        scores = torch.zeros_like(target, dtype=torch.float)

        # 加上句间迁移分数
        scores[1:] += self.transitions[target[:-1], target[1:]]
        # 加上发射分数
        scores += emit.gather(dim=2, index=target.unsqueeze(2)).squeeze(2)
        # 通过掩码过滤分数
        score = scores.masked_select(mask).sum()

        # 获取序列最后的词性的索引
        ends = mask.sum(dim=0).view(1, -1) - 1
        # 加上句首迁移分数
        score += self.strans[target[0]].sum()
        # 加上句尾迁移分数
        score += self.etrans[target.gather(dim=0, index=ends)].sum()
        return score

    def forward(self, emit_matrixs, labels, mask):
        logZ = self.get_logZ(emit_matrixs, mask)
        scores = self.score(emit_matrixs, labels, mask)
        # return logZ - scores
        return (logZ - scores) / emit_matrixs.size()[1]