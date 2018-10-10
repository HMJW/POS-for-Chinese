class Config(object):
    def __init__(self):
        # file config
        self.train_data_file = '../data/ctb51-penn2malt-default/train.conll'
        self.dev_data_file = '../data/ctb51-penn2malt-default/dev.conll'
        self.test_data_file = '../data/ctb51-penn2malt-default/test.conll'
        self.embedding_file = '../data/embedding/giga.100.txt'
        self.net_file = './char_lstm_crf.pt'
        self.vocab_file = './vocab.pkl'
        
        # model config
        self.char_dim = 100
        self.word_dim = 100
        self.word_hidden = 300
        self.char_hidden = 200
        self.layers = 2
        self.dropout = 0.6

        # training config
        self.epoch = 100
        self.gpu = -1
        self.start_epoch = 0
        self.learn_rate = 0.001
        self.batch_size = 50
        self.tread_num = 4
        self.decay = 0.05
        self.momentum = 0.9
        self.patience = 10
