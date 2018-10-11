import pickle

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as Data
from torch.nn.utils.rnn import *


def save_pkl(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_pkl(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


def collate_fn(data):
    batch = zip(*data)
    return tuple([torch.tensor(x) if len(x[0].size()) < 1 else pad_sequence(x, True) for x in batch])


def collate_fn_cuda(data):
    batch = zip(*data)
    return tuple([torch.tensor(x).cuda() if len(x[0].size()) < 1 else pad_sequence(x, True).cuda() for x in batch])


class Corpus(object):
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


class TensorDataSet(Data.Dataset):
    def __init__(self, *data):
        super(TensorDataSet, self).__init__()
        self.items = list(zip(*data))

    def __getitem__(self, index):
        return self.items[index]
    
    def __len__(self):
        return len(self.items)
    

class Vocab(object):
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
        unk_words = [w for w in words if w not in self._word2id]
        unk_chars = [c for c in ''.join(unk_words) if c not in self._char2id]

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
        for i, w in enumerate(self._words):
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

    def id2label(self, id):
        assert (isinstance(id, int) or isinstance(id, list))
        if isinstance(id, int):
            assert (id >= self.num_labels)
            return self._labels[id] # if label not in training data, index to 0 ?
        elif isinstance(id, list):
            return [self._labels[i] for i in id]


class Decoder(object):
    def __init__(self):
        pass

    @staticmethod
    def viterbi(crf, emit_matrix):
        length = emit_matrix.size()[0]
        max_score = torch.zeros_like(emit_matrix)
        paths = torch.zeros_like(emit_matrix, dtype=torch.long)

        max_score[0] = emit_matrix[0] + crf.strans
        for i in range(1, length):
            emit_scores = emit_matrix[i]
            scores = emit_scores + crf.transitions + \
                max_score[i - 1].view(-1, 1).expand(-1, crf.labels_num)
            max_score[i], paths[i] = torch.max(scores, 0)

        max_score[-1] += crf.etrans
        prev = torch.argmax(max_score[-1])
        predict = [prev.item()]
        for i in range(length - 1, 0, -1):
            prev = paths[i][prev.item()]
            predict.insert(0, prev.item())
        return torch.tensor(predict)
    
    @staticmethod
    def viterbi_batch(crf, emits, masks):
        'optimized by zhangyu'
        T, B, N = emits.shape
        lens = masks.sum(dim=0)
        delta = torch.zeros_like(emits)
        paths = torch.zeros_like(emits, dtype=torch.long)

        delta[0] = crf.strans + emits[0]  # [B, N]

        for i in range(1, T):
            trans_i = crf.transitions.unsqueeze(0)  # [1, N, N]
            emit_i = emits[i].unsqueeze(1)  # [B, 1, N]
            scores = trans_i + emit_i + delta[i - 1].unsqueeze(2)  # [1, N, N]+[B, 1, N]+[B, N, 1]->[B, N, N]
            delta[i], paths[i] = torch.max(scores, dim=1)

        predicts = []
        for i, length in enumerate(lens):
            prev = torch.argmax(delta[length - 1, i] + crf.etrans)

            predict = [prev]
            for j in reversed(range(1, length)):
                prev = paths[j, i, prev]
                predict.append(prev)

            predicts.append(torch.tensor(predict).flip(0))

        return predicts