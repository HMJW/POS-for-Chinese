import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import *

from module import CharLSTM,CRFlayer


class Char_LSTM_CRF(torch.nn.Module):
    def __init__(self, n_char, char_dim, char_hidden, n_word, word_dim,
                 n_layers, word_hidden, n_target, drop=0.5):
        super(Char_LSTM_CRF, self).__init__()

        self.embedding_dim = word_dim
        self.drop1 = torch.nn.Dropout(drop)
        self.embedding = torch.nn.Embedding(n_word, word_dim, padding_idx=0)
        self.char_lstm = CharLSTM(n_char, char_dim, char_hidden)

        if n_layers > 1:
            self.lstm_layer = torch.nn.LSTM(
                input_size=self.embedding_dim + char_hidden,
                hidden_size=word_hidden//2,
                batch_first=True,
                bidirectional=True,
                num_layers=n_layers,
                dropout=0.2
            )
        else:
            self.lstm_layer = torch.nn.LSTM(
                input_size=self.embedding_dim + char_hidden,
                hidden_size=word_hidden//2,
                batch_first=True,
                bidirectional=True,
                num_layers=1,
            )
        self.hidden = nn.Linear(word_hidden, word_hidden//2)
        self.out = torch.nn.Linear(word_hidden//2, n_target)
        self.CRFlayer = CRFlayer(n_target)

        self.init_embedding()
        self.init_linear()

    def init_linear(self):
        init.xavier_uniform_(self.out.weight)
        init.xavier_uniform_(self.hidden.weight)

    def init_embedding(self):
        bias = (3.0 / self.embedding.weight.size(1)) ** 0.5
        init.uniform_(self.embedding.weight, -bias, bias)

    def load_pretrained_embedding(self, pre_embeddings):
        assert (pre_embeddings.size()[1] == self.embedding_dim)
        self.embedding.weight = nn.Parameter(pre_embeddings)

    def forward(self, x, lens, word_chars, word_lengths):
        mask = torch.arange(x.size()[1]) < lens.unsqueeze(-1)
        h = self.char_lstm.forward(word_chars[mask], word_lengths[mask])
        h = pad_sequence(torch.split(
            h, lens.tolist()), True, padding_value=0)

        x = self.embedding(x)
        x = self.drop1(torch.cat((x, h), 2))

        sorted_lens, sorted_index = torch.sort(lens, dim=0, descending=True)
        raw_index = torch.sort(sorted_index, dim=0)[1]
        x = x[sorted_index]
        x = pack_padded_sequence(x, sorted_lens, batch_first=True)

        r_out, state = self.lstm_layer(x, None)
        out, _ = pad_packed_sequence(r_out, batch_first=True, padding_value=0)
        out = out[raw_index]
        out = torch.tanh(self.hidden(out))
        out = self.out(out)
        return out

    def get_loss(self, emit_matrixs, labels, mask):
        logZ = self.CRFlayer.get_logZ(emit_matrixs, mask)
        scores = self.CRFlayer.score(emit_matrixs, labels, mask)
        return (logZ - scores) / emit_matrixs.size()[1]

    def predict(self, emit_matrix):
        return self.CRFlayer.viterbi_decode(emit_matrix)
