import argparse

from config import Config
from model import Char_LSTM_CRF
from utils import *

if __name__ == '__main__':
    config = Config()

    parser = argparse.ArgumentParser(description='reload the trained model and evaluate test dataset')

    parser.add_argument('--corpus_file', default=config.corpus_file, help='path to corpus pkl file')
    parser.add_argument('--net_file', default=config.net_file, help='path to save the model')
    parser.add_argument('--test_file', default=config.test_data_file, help='path to test file')
    parser.add_argument('--thread', type=int, default=config.tread_num, help='thread num')

    args = parser.parse_args()
    print('setting:')
    print(args)
    print()

    torch.set_num_threads(args.thread)
    
    # loading corpus
    corpus = load_pkl(args.corpus_file)

    # loading network
    print("loading model...")
    net = torch.load(args.net_file)

    # reading test data file
    print('reading test data file...')
    test = DataSet(args.test_file)
    evaluator = Evaluator(corpus.word2id, corpus.char2id, corpus.label2id)

    print('evaluating test data...')
    test_p = evaluator.eval_tag(net, test)
    print('test : precision = %.4f' % (test_p))
    evaluator.clear_num()
