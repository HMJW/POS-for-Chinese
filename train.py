import argparse
import collections
import datetime
from itertools import chain

import torch.optim as optim
import torch.utils.data as Data
from torch.nn.utils.rnn import *

from config import Config
from model import Char_LSTM_CRF
from utils import *


def process_data(vocab, dataset, max_word_len=20):
    word_idxs, sen_lens, char_idxs, word_lens, label_idxs = [], [], [], [], []

    for wordseq, labelseq in zip(dataset.word_seqs, dataset.label_seqs):
        _word_idxs = vocab.word2id(wordseq)
        _label_idxs = vocab.label2id(labelseq)

        word_idxs.append(torch.tensor(_word_idxs))
        sen_lens.append(torch.tensor(len(_word_idxs)))
        
        _char_idxs = vocab.char2id(wordseq)
        char_idxs.append(torch.tensor(_char_idxs))

        _word_lens = [min(len(w), max_word_len) for w in wordseq]
        word_lens.append(torch.tensor(_word_lens))

        _label_idxs = vocab.label2id(labelseq)
        label_idxs.append(torch.tensor(_label_idxs))

    return TensorDataSet(word_idxs, sen_lens, char_idxs, word_lens, label_idxs)


class Evaluator(object):
    def __init__(self, vocab, task='pos'):
        self.pred_num = 0
        self.gold_num = 0
        self.correct_num = 0
        self.vocab = vocab
        self.task = task
        self.use_cuda = use_cuda
        
    def clear_num(self):
        self.pred_num = 0
        self.gold_num = 0
        self.correct_num = 0

    def eval(self, network, data_loader):
        network.eval()
        total_loss = 0.0

        for word_idxs, sen_lens, char_idxs, word_lens, label_idxs in data_loader:
            batch_size = word_idxs.size(0)
            # mask = torch.arange(y.size(1)) < lens.unsqueeze(-1)
            mask = word_idxs.gt(0)
            
            out = network.forward(word_idxs, sen_lens, char_idxs, word_lens)

            batch_loss = network.get_loss(out.transpose(0, 1), label_idxs.t(), mask.t())
            total_loss += batch_loss * batch_size

            predicts = Decoder.viterbi_batch(network.crf, out.transpose(0, 1), mask.t())
            targets = torch.split(label_idxs[mask], sen_lens.tolist())
            
            if self.task == 'pos':
                for predict, target in zip(predicts, targets):
                    predict = predict.tolist()
                    target = target.tolist()                
                    correct_num = sum(x==y for x,y in zip(predict, target))
                    self.correct_num += correct_num
                    self.pred_num += len(predict)
                    self.gold_num += len(target)
            elif self.task == 'chunking':
                for predict, target in zip(predicts, targets):
                    predict = self.vocab.id2label(predict.tolist())
                    target = self.vocab.id2label(target.tolist())
                    correct_num, pred_num, gold_num = self.cal_num(predict, target)
                    self.correct_num += correct_num
                    self.pred_num += pred_num
                    self.gold_num += gold_num

        if self.task == 'pos':
            precision = self.correct_num/self.pred_num
            self.clear_num()
            return total_loss, precision
        elif self.task == 'chunking':
            precision = self.correct_num/self.pred_num
            recall = self.correct_num/self.gold_num
            Fscore = (2*precision*recall)/(precision+recall)
            self.clear_num()
            return total_loss, precision, recall, Fscore

    def cal_num(self, pred, gold):
        set1 = self.recognize(pred)
        set2 = self.recognize(gold)
        intersction = set1 & set2
        correct_num = len(intersction)
        pred_num = len(set1)
        gold_num = len(set2)
        return correct_num, pred_num, gold_num

    def recognize(self, sequence):
        """
        copy from the paper
        """
        chunks = []
        current = None

        for i, label in enumerate(sequence):
            if label.startswith('B-'):

                if current is not None:
                    chunks.append('@'.join(current))
                current = [label.replace('B-', ''), '%d' % i]

            elif label.startswith('S-'):

                if current is not None:
                    chunks.append('@'.join(current))
                    current = None
                base = label.replace('S-', '')
                chunks.append('@'.join([base, '%d' % i]))

            elif label.startswith('I-'):

                if current is not None:
                    base = label.replace('I-', '')
                    if base == current[0]:
                        current.append('%d' % i)
                    else:
                        chunks.append('@'.join(current))
                        current = [base, '%d' % i]

                else:
                    current = [label.replace('I-', ''), '%d' % i]

            elif label.startswith('E-'):

                if current is not None:
                    base = label.replace('E-', '')
                    if base == current[0]:
                        current.append('%d' % i)
                        chunks.append('@'.join(current))
                        current = None
                    else:
                        chunks.append('@'.join(current))
                        current = [base, '%d' % i]
                        chunks.append('@'.join(current))
                        current = None

                else:
                    current = [label.replace('E-', ''), '%d' % i]
                    chunks.append('@'.join(current))
                    current = None
            else:
                if current is not None:
                    chunks.append('@'.join(current))
                current = None

        if current is not None:
            chunks.append('@'.join(current))

        return set(chunks)


if __name__ == '__main__':
    # init config
    config = Config()

    parser = argparse.ArgumentParser(description='Training with LSTM')

    parser.add_argument('--net_file', default=config.net_file, help='path to save the model')
    parser.add_argument('--vocab_file', default=config.vocab_file, help='path to save the vocab')
    parser.add_argument('--train_file', default=config.train_data_file, help='path to training file')
    parser.add_argument('--dev_file', default=config.dev_data_file, help='path to development file')
    parser.add_argument('--test_file', default=config.test_data_file, help='path to test file')
    parser.add_argument('--emb_file', default=config.embedding_file, help='path to pre embedding file')
    parser.add_argument('--gpu', type=int, default=config.gpu, help='gpu id, set to -1 if use cpu mode')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size (50)')
    parser.add_argument('--epoch', type=int, default=config.epoch, help='maximum epoch number')
    parser.add_argument('--lr', type=float, default=config.learn_rate, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=config.decay, help='decay ratio of learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--patience', type=int, default=config.patience, help='most epoch to exit train if the precision does not become better')
    parser.add_argument('--thread', type=int, default=config.tread_num, help='thread num')
    parser.add_argument('--start_epoch', type=int, default=config.start_epoch, help='start point of epoch')
    parser.add_argument('--pre_emb', action='store_true', help='choose if use pretrain embedding')
    parser.add_argument('--char_lstm', action='store_false', help='choose if use char_lstm')
    parser.add_argument('--char_dim', type=int, default=config.char_dim, help='char embedding dim')
    parser.add_argument('--word_dim', type=int, default=config.word_dim, help='word embeddding dim')
    parser.add_argument('--char_hidden', type=int, default=config.char_hidden, help='char lstm hidden')
    parser.add_argument('--word_hidden', type=int, default=config.word_hidden, help='word lstm hidden')
    parser.add_argument('--dropout', type=float, default=config.dropout, help='dropout ratio')
    parser.add_argument('--layers', type=int, default=config.layers, help='layers num')
    parser.add_argument('--update', choices=['sgd', 'adam'], default='adam', help='optimizer choice')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    args = parser.parse_args()
    print('setting:')
    print(args)
    print()

    # choose GPU and init seed
    if args.gpu >= 0:
        use_cuda = True
        torch.cuda.set_device(args.gpu)
        torch.set_num_threads(args.thread)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        print('using GPU device : %d' % args.gpu)
        print('GPU seed = %d' % torch.cuda.initial_seed())
        print('CPU seed = %d' % torch.initial_seed())
    else:
        use_cuda = False
        torch.set_num_threads(args.thread)
        torch.manual_seed(args.seed)
        print('CPU seed = %d' % torch.initial_seed())

    # read training , dev and test file
    print('loading three datasets...')
    train = Corpus(args.train_file)
    dev = Corpus(args.dev_file)
    test = Corpus(args.test_file)

    # collect all words , characters and labels in trainning data
    labels = sorted(set(chain(*train.label_seqs)))
    words = sorted(set(chain(*train.word_seqs)))
    chars = sorted(set(''.join(words)))
    vocab = Vocab(words, labels, chars)

    # choose if use pretrained word embedding
    if args.pre_emb:
        print('loading pretrained embedding...')
        pre_embedding = vocab.read_embedding(args.emb_file)
    print('Words : %d，Characters : %d，labels : %d' %
          (vocab.num_words, vocab.num_chars, vocab.num_labels))
    save_pkl(vocab, args.vocab_file)

    # init evaluator
    evaluator = Evaluator(vocab, task='pos')

    # process training data , change string to index
    print('processing datasets...')
    train_data = process_data(vocab, train, max_word_len=20)
    dev_data = process_data(vocab, dev, max_word_len=20)
    test_data = process_data(vocab, test, max_word_len=20)

    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn if not use_cuda else collate_fn_cuda
    )
    dev_loader = Data.DataLoader(
        dataset=dev_data,
        batch_size=100,
        shuffle=False,
        collate_fn=collate_fn if not use_cuda else collate_fn_cuda
    )
    test_loader = Data.DataLoader(
        dataset=test_data,
        batch_size=100,
        shuffle=False,
        collate_fn=collate_fn if not use_cuda else collate_fn_cuda
    )

    # create neural network
    net = Char_LSTM_CRF(vocab.num_chars, args.char_dim, args.char_hidden, vocab.num_words,
                   args.word_dim, args.layers, args.word_hidden, vocab.num_labels, args.dropout)
    if args.pre_emb:
        net.load_pretrained_embedding(pre_embedding)
    print(net)

    # if use GPU , move all needed tensors to CUDA
    if use_cuda:
        net.cuda()
        
    # choose optimizer
    if args.update == 'sgd':
        print('Using SGD optimizer...')
        optimizer = optim.SGD(net.parameters(), lr=args.lr)
    elif args.update == 'adam':
        print('Using Adam optimizer...')
        optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # record some parameters
    max_precision = 0
    test_precision = 0
    max_epoch = 0
    patience = 0

    # begin to train
    print('start to train the model ')
    for e in range(args.epoch):
        print('==============================Epoch<%d>==============================' % (e + 1))
        net.train()
        time_start = datetime.datetime.now()
        for word_idxs, sen_lens, char_idxs, word_lens, label_idxs in train_loader:
            optimizer.zero_grad()
            # mask = torch.arange(label_idxs.size(1)) < lens.unsqueeze(-1)
            mask = word_idxs.gt(0)

            out = net.forward(word_idxs, sen_lens, char_idxs, word_lens)
            loss = net.get_loss(out.transpose(0,1), label_idxs.t(), mask.t())
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            train_total_loss, train_p = evaluator.eval(net, train_loader)
            print('train : loss = %.4f  precision = %.4f' % (train_total_loss/train.sentence_num, train_p))

            dev_total_loss, dev_p = evaluator.eval(net, dev_loader)
            print('dev   : loss = %.4f  precision = %.4f' % (dev_total_loss/train.sentence_num, dev_p))

            test_total_loss, test_p = evaluator.eval(net, test_loader)
            print('test  : loss = %.4f  precision = %.4f' % (test_total_loss/train.sentence_num, test_p))

        # save the model when dev precision get better
        if dev_p > max_precision:
            max_precision = dev_p
            test_precision = test_p
            max_epoch = e + 1
            patience = 0
            print('save the model...')
            torch.save(net, args.net_file)
        else:
            patience += 1

        time_end = datetime.datetime.now()
        print('iter executing time is ' + str(time_end - time_start) + '\n')
        if patience > args.patience:
            break

    print('train finished with epoch: %d / %d' % (e + 1, args.epoch))
    print('best epoch is epoch = %d ,the dev precision = %.4f the test precision = %.4f' %
          (max_epoch, max_precision, test_precision))
