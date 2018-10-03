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


class Corpus():
    def __init__(self, words, labels, chars):
        self.UNK = '<UNK>'
        self.PAD = '<PAD>'

        self.words = [self.PAD] + words + [self.UNK]
        self.chars = [self.PAD] + chars + [self.UNK]
        self.labels = labels

        self.word2id = {w: i for i, w in enumerate(self.words)}
        self.char2id = {c: i for i, c in enumerate(self.chars)}
        self.label2id = {l: i for i, l in enumerate(self.labels)}

        self.num_words = len(self.words)
        self.num_chars = len(self.chars)
        self.num_labels = len(self.labels)

        self.UNK_word_index = self.word2id[self.UNK]
        self.UNK_char_index = self.char2id[self.UNK]
        self.PAD_word_index = self.word2id[self.PAD]
        self.PAD_char_index = self.char2id[self.PAD]

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
        self.words =[self.PAD] + sorted(set(self.words + unk_words) - {self.PAD})
        self.chars =[self.PAD] + sorted(set(self.chars + unk_chars) - {self.PAD})

        # update the words,chars dictionary
        self.word2id = {w: i for i, w in enumerate(self.words)}
        self.char2id = {c: i for i, c in enumerate(self.chars)}
        self.UNK_word_index = self.word2id[self.UNK]
        self.UNK_char_index = self.char2id[self.UNK]
        self.PAD_word_index = self.word2id[self.PAD]
        self.PAD_char_index = self.char2id[self.PAD]
        
        # update the numbers of words and chars
        self.num_words = len(self.words)
        self.num_chars = len(self.chars)

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

    def process_data(self, sentences, labels, max_len=20):
        x, lens, char_x, char_lens, y = [], [], [], [], []

        for wordseq, tagseq in zip(sentences, labels):
            wiseq = [self.word2id.get(w, self.UNK_word_index) for w in wordseq]

            tiseq = [self.label2id.get(t, 0) for t in tagseq]
            x.append(torch.tensor(wiseq))
            lens.append(torch.tensor(len(tiseq)))
            # 不足最大长度的部分用0填充
            char_x.append(torch.tensor([
                [self.char2id.get(c, self.UNK_char_index)
                 for c in w[:max_len]] + [0] * (max_len - len(w))
                for w in wordseq
            ]))
            char_lens.append(torch.tensor([min(len(w), max_len)
                                           for w in wordseq],
                                          dtype=torch.long))
            y.append(torch.tensor([ti for ti in tiseq], dtype=torch.long))

        # x = pad_sequence(x, True)
        # lens = torch.tensor(lens)
        # char_x = pad_sequence(char_x, True, padding_value=0)
        # char_lens = pad_sequence(char_lens, True, padding_value=0)
        # y = pad_sequence(y, True, padding_value=0)
        # dataset = Data.TensorDataset(x, lens, char_x, char_lens, y)
        return TensorDataSet(x, lens, char_x, char_lens, y)


class Evaluator(object):
    def __init__(self, word2id, char2id, label2id):
        self.pred_num = 0
        self.gold_num = 0
        self.correct_num = 0
        self.word2id = word2id
        self.char2id = char2id
        self.label2id = label2id
        self.labels = list(label2id.keys())

    def clear_num(self):
        self.pred_num = 0
        self.gold_num = 0
        self.correct_num = 0

    def parse(self, wordseq):
        word_lens = list(map(len, wordseq))
        max_word_length = max(word_lens)
        chars = torch.full([len(wordseq), max_word_length], 0, dtype=torch.long)

        for word, char_vector in zip(wordseq, chars):
            for i in range(len(word)):
                char_vector[i] = self.char2id.get(
                    word[i], self.char2id['<UNK>'])

        sentence = torch.tensor(list(
            map(lambda x: self.word2id.get(x, self.word2id['<UNK>']), wordseq)))
        x = sentence.view(1, -1)
        length = torch.tensor([len(wordseq)], dtype=torch.long)
        word_lens = torch.tensor([word_lens], dtype=torch.long)
        return x, length, chars.unsqueeze(0), word_lens

    def eval_tag(self, network, dataset):
        network.eval()
        for wordseq, labelseq in zip(dataset.word_seqs, dataset.label_seqs):
            x, length, chars, word_lens = self.parse(wordseq)
            out = network.forward(x, length, chars, word_lens)
            pred = network.predict(out.view(length, -1))
            pred = list(map(lambda x: self.labels[x], pred))
            correct_num = sum(int(a == b) for a, b in zip(pred, labelseq))
            self.correct_num += correct_num
            self.pred_num += len(pred)
            self.gold_num += len(labelseq)
        precision = self.correct_num/self.pred_num
        return precision
