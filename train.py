import argparse

import torch
import torch.utils.data as Data

from config import config
from model import Char_LSTM_CRF
from utils import *


def process_data(vocab, dataset, max_word_len=30, use_cuda=False):
    word_idxs, char_idxs, label_idxs = [], [], []

    for wordseq, labelseq in zip(dataset.word_seqs, dataset.label_seqs):
        _word_idxs = vocab.word2id(wordseq)
        _label_idxs = vocab.label2id(labelseq)
        _char_idxs = vocab.char2id(wordseq, max_word_len)

        if not use_cuda:
            word_idxs.append(torch.tensor(_word_idxs))
            char_idxs.append(torch.tensor(_char_idxs))
            label_idxs.append(torch.tensor(_label_idxs))
        else:
            word_idxs.append(torch.tensor(_word_idxs).cuda())
            char_idxs.append(torch.tensor(_char_idxs).cuda())
            label_idxs.append(torch.tensor(_label_idxs).cuda())

    return TensorDataSet(word_idxs, char_idxs, label_idxs)


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
    model_name = 'char_lstm_crf'
    config = config[model_name]
    for name, value in vars(config).items():
        print('%s = %s' %(name, str(value)))
        
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu', type=int, default=config.gpu, help='gpu id, set to -1 if use cpu mode')
    parser.add_argument('--pre_emb', action='store_true', help='choose if use pretrain embedding')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--thread', type=int, default=config.tread_num, help='thread num')
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
    train = Corpus(config.train_file, lower=False)
    dev = Corpus(config.dev_file, lower=False)
    test = Corpus(config.test_file, lower=False)

    # collect all words, characters and labels in trainning data
    vocab = collect(train, low_freq=0)

    # choose if use pretrained word embedding
    if args.pre_emb and config.embedding_file !=None:
        print('loading pretrained embedding...')
        pre_embedding = vocab.read_embedding(config.embedding_file)
    print('Words : %d，Characters : %d，labels : %d' %
          (vocab.num_words, vocab.num_chars, vocab.num_labels))
    save_pkl(vocab, config.vocab_file)

    # process training data , change string to index
    print('processing datasets...')
    train_data = process_data(vocab, train, max_word_len=20, use_cuda=False)
    dev_data = process_data(vocab, dev, max_word_len=20, use_cuda=False)
    test_data = process_data(vocab, test, max_word_len=20, use_cuda=False)

    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        collate_fn=collate_fn if not use_cuda else collate_fn_cuda
    )
    dev_loader = Data.DataLoader(
        dataset=dev_data,
        batch_size=config.eval_batch,
        shuffle=False,
        collate_fn=collate_fn if not use_cuda else collate_fn_cuda
    )
    test_loader = Data.DataLoader(
        dataset=test_data,
        batch_size=config.eval_batch,
        shuffle=False,
        collate_fn=collate_fn if not use_cuda else collate_fn_cuda
    )

    # create neural network
    net = Char_LSTM_CRF(vocab.num_chars, 
                        config.char_dim, 
                        config.char_hidden, 
                        vocab.num_words,
                        config.word_dim, 
                        config.layers, 
                        config.word_hidden, 
                        vocab.num_labels, 
                        config.dropout
                        )
    if args.pre_emb:
        net.load_pretrained_embedding(pre_embedding)
    print(net)

    # if use GPU , move all needed tensors to CUDA
    if use_cuda:
        net.cuda()
    
    # init evaluator
    evaluator = Evaluator(vocab, task='pos')
    # init trainer
    trainer = Trainer(net, config)
    # start to train
    trainer.train((train_loader, dev_loader, test_loader), evaluator)
