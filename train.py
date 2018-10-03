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

if __name__ == '__main__':
    # init config
    config = Config()

    parser = argparse.ArgumentParser(description='Training with LSTM')

    parser.add_argument('--net_file', default=config.net_file, help='path to save the model')
    parser.add_argument('--corpus_file', default=config.corpus_file, help='path to save the corpus')
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
    corpus = Corpus(words, labels, chars)

    # choose if use pretrained word embedding
    if args.pre_emb:
        print('loading pretrained embedding...')
        pre_embedding = corpus.read_embedding(args.emb_file)
    print('Words : %d，Characters : %d，labels : %d' %
          (corpus.num_words, corpus.num_chars, corpus.num_labels))
    save_pkl(corpus, args.corpus_file)

    # init evaluator
    evaluator = Evaluator(corpus.word2id, corpus.char2id, corpus.label2id)

    # process training data , change string to index
    print('processing datasets...')
    traindata = corpus.process_data(train.word_seqs, train.label_seqs,
                                    max_len=20)
    loader = Data.DataLoader(
        dataset=traindata,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # create neural network
    net = Char_LSTM_CRF(corpus.num_chars, args.char_dim, args.char_hidden, corpus.num_words,
                   args.word_dim, args.layers, args.word_hidden, corpus.num_labels, args.dropout)
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
        for x, lens, char_x, char_lens, y in loader:
            optimizer.zero_grad()
            net.zero_grad()

            mask = torch.arange(y.size(1)) < lens.unsqueeze(-1)
            out = net.forward(x, lens, char_x, char_lens)
            loss = net.get_loss(out.transpose(0,1), y.t(), mask.t())
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            train_p = evaluator.eval_tag(net, train)
            print('train : precision = %.4f' % (train_p))
            evaluator.clear_num()

            dev_p = evaluator.eval_tag(net, dev)
            print('dev : precision = %.4f' % (dev_p))
            evaluator.clear_num()

            test_p = evaluator.eval_tag(net, test)
            print('test : precision = %.4f' % (test_p))
            evaluator.clear_num()

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
