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
        self.init_trans()

    def init_trans(self):
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
        T, B, N = emit.shape
        scores = torch.zeros(T, B)

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

    def forward(self, emit_matrix, lengths):
        pass

    def viterbi_decode(self, emit_matrix):
        length = emit_matrix.size()[0]
        max_score = torch.zeros((length, self.labels_num))
        paths = torch.zeros((length, self.labels_num), dtype=torch.long)

        max_score[0] = emit_matrix[0] + self.strans
        for i in range(1, length):
            emit_scores = emit_matrix[i]
            scores = emit_scores + self.transitions + \
                max_score[i - 1].view(-1, 1).expand(-1, self.labels_num)
            max_score[i], paths[i] = torch.max(scores, 0)

        max_score[-1] += self.etrans
        prev = torch.argmax(max_score[-1])
        predict = [prev.item()]
        for i in range(length - 1, 0, -1):
            prev = paths[i][prev.item()]
            predict.insert(0, prev.item())
        return predict
