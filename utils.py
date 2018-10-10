import pickle

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as Data
from torch.nn.utils.rnn import *

from loader import TensorDataSet


def save_pkl(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_pkl(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

    
class DataSet(object):
    def __init__(self, filename=None):
        self.filename = filename
        self.sentence_num = 0
        self.word_num = 0
        self.word_seqs = []
        self.label_seqs = []
        sentence = []
        sequence = []
        with open(self.filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if line == '\n':
                    self.word_seqs.append(sentence)
                    self.label_seqs.append(sequence)
                    self.sentence_num += 1
                    sentence = []
                    sequence = []
                else:
                    conll = line.split()
                    sentence.append(conll[1])
                    sequence.append(conll[3])
                    self.word_num += 1
        print('%s : sentences:%d，words:%d' % (filename, self.sentence_num, self.word_num))


class Vocab():
    def __init__(self, words, labels, chars):
        self.UNK = '<UNK>'
        self.PAD = '<PAD>'

        self._words = [self.PAD] + words + [self.UNK]
        self._chars = [self.PAD] + chars + [self.UNK]
        self._labels = labels

        self._word2id = {w: i for i, w in enumerate(self._words)}
        self._char2id = {c: i for i, c in enumerate(self._chars)}
        self._label2id = {l: i for i, l in enumerate(self._labels)}

        self.num_words = len(self._words)
        self.num_chars = len(self._chars)
        self.num_labels = len(self._labels)

        self.UNK_word_index = self._word2id[self.UNK]
        self.UNK_char_index = self._char2id[self.UNK]
        self.PAD_word_index = self._word2id[self.PAD]
        self.PAD_char_index = self._char2id[self.PAD]

    def read_embedding(self, embedding_file):
        with open(embedding_file, 'r') as f:
            lines = f.readlines()
        splits = [line.split() for line in lines]
        # read pretrained embedding file
        words, vectors = zip(*[
            (split[0], list(map(float, split[1:]))) for split in splits
        ])

        pretrained = {w: torch.tensor(v) for w, v in zip(words, vectors)}
        unk_words = [w for w in words if w not in self.word2id]
        unk_chars = [c for c in ''.join(unk_words) if c not in self.char2id]

        # extend words and chars
        # ensure the <PAD> token at the first position
        self._words =[self.PAD] + sorted(set(self._words + unk_words) - {self.PAD})
        self._chars =[self.PAD] + sorted(set(self._chars + unk_chars) - {self.PAD})

        # update the words,chars dictionary
        self._word2id = {w: i for i, w in enumerate(self._words)}
        self._char2id = {c: i for i, c in enumerate(self._chars)}
        self.UNK_word_index = self._word2id[self.UNK]
        self.UNK_char_index = self._char2id[self.UNK]
        self.PAD_word_index = self._word2id[self.PAD]
        self.PAD_char_index = self._char2id[self.PAD]
        
        # update the numbers of words and chars
        self.num_words = len(self._words)
        self.num_chars = len(self._chars)

        # initial the extended embedding table
        embdim = len(vectors[0])
        
        extended_embed = torch.randn(self.num_words, embdim)
        init.normal_(extended_embed, 0, 1 / embdim ** 0.5)

        # the word in pretrained file use pretrained vector
        # the word not in pretrained file but in training data use random initialized vector
        for i, w in enumerate(self.words):
            if w in pretrained:
                extended_embed[i] = pretrained[w]
        return extended_embed

    def word2id(self, word):
        assert (isinstance(word, str) or isinstance(word, list))
        if isinstance(word, str):
            return self._word2id.get(word, self.UNK_word_index)
        elif isinstance(word, list):
            return [self._word2id.get(w, self.UNK_word_index) for w in word]

    def label2id(self, label):
        assert (isinstance(label, str) or isinstance(label, list))
        if isinstance(label, str):
            return self._label2id.get(label, 0) # if label not in training data, index to 0 ?
        elif isinstance(label, list):
            return [self._label2id.get(l, 0) for l in label]

    def char2id(self, char, max_len=20):
        assert (isinstance(char, str) or isinstance(char, list))
        if isinstance(char, str):
            return self._char2id.get(char, self.UNK_char_index)
        elif isinstance(char, list):
            return [[self._char2id.get(c, self.UNK_char_index) for c in w[:max_len]] + 
                    [0] * (max_len - len(w)) for w in char]


class Evaluator(object):
    def __init__(self, vocab):
        self.pred_num = 0
        self.gold_num = 0
        self.correct_num = 0
        self.vocab = vocab

    def clear_num(self):
        self.pred_num = 0
        self.gold_num = 0
        self.correct_num = 0

    def eval_tag(self, network, data_loader):
        network.eval()
        total_loss = 0.0

        for x, lens, chars_x, chars_lens, y in data_loader:
            batch_size = x.size(0)
            # mask = torch.arange(y.size(1)) < lens.unsqueeze(-1)
            mask = x.gt(0)
            
            out = network.forward(x, lens, chars_x, chars_lens)

            batch_loss = network.get_loss(out.transpose(0, 1), y.t(), mask.t()) * batch_size
            total_loss += batch_loss

            predicts = network.CRFlayer.viterbi_batch(out.transpose(0, 1), mask.t())
            predicts = pad_sequence(predicts, padding_value=-1, batch_first=True)

            correct_num = torch.sum(predicts==y)
            self.correct_num += correct_num
            self.pred_num += sum(lens)
            self.gold_num += sum(lens)

        precision = self.correct_num.float()/self.pred_num.float()
        self.clear_num()
        return total_loss, precision
