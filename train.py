import argparse
import collections
import datetime
from itertools import chain

import torch.optim as optim
import torch.utils.data as Data
from torch.nn.utils.rnn import *

from config import Config
from loader import collate_fn
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
    
    args = parser.parse_args()
    print('setting:')
    print(args)
    print()

    # choose GPU and init seed
    if args.gpu >= 0:
        use_gpu = True
        torch.cuda.set_device(args.gpu)
    else:
        use_gpu = False
    torch.set_num_threads(args.thread)
    torch.manual_seed(1)

    # read training , dev and test file
    print('loading three datasets...')
    train = DataSet(args.train_file)
    dev = DataSet(args.dev_file)
    test = DataSet(args.test_file)

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
    evaluator = Evaluator(vocab)

    # process training data , change string to index
    print('processing datasets...')
    train_data = process_data(vocab, train, max_word_len=20)
    dev_data = process_data(vocab, dev, max_word_len=20)
    test_data = process_data(vocab, test, max_word_len=20)

    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    dev_loader = Data.DataLoader(
        dataset=dev_data,
        batch_size=100,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = Data.DataLoader(
        dataset=test_data,
        batch_size=100,
        shuffle=False,
        collate_fn=collate_fn
    )

    # create neural network
    net = Char_LSTM_CRF(vocab.num_chars, args.char_dim, args.char_hidden, vocab.num_words,
                   args.word_dim, args.layers, args.word_hidden, vocab.num_labels, args.dropout)
    if args.pre_emb:
        net.load_pretrained_embedding(pre_embedding)
    print(net)

    # choose optimizer
    if args.update == 'sgd':
        print('Using SGD optimizer...')
        optimizer = optim.SGD(net.parameters(), lr=args.lr)
    elif args.update == 'adam':
        print('Using Adam optimizer...')
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    
    # TODO: if use GPU , move all needed tensors to CUDA
    if use_gpu:
        net.cuda()

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
        for x, lens, char_x, char_lens, y in train_loader:
            optimizer.zero_grad()
            mask = torch.arange(y.size(1)) < lens.unsqueeze(-1)

            out = net.forward(x, lens, char_x, char_lens)
            loss = net.get_loss(out.transpose(0,1), y.t(), mask.t())
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            train_total_loss, train_p = evaluator.eval_tag(net, train_loader)
            print('train : loss = %.4f  precision = %.4f' % (train_total_loss/train.sentence_num, train_p))

            dev_total_loss, dev_p = evaluator.eval_tag(net, dev_loader)
            print('dev   : loss = %.4f  precision = %.4f' % (dev_total_loss/train.sentence_num, dev_p))

            test_total_loss, test_p = evaluator.eval_tag(net, test_loader)
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
