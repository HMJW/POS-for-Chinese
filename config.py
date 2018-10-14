class Config(object):
    train_file = '../data/ctb51-penn2malt-default/train.conll'
    dev_file = '../data/ctb51-penn2malt-default/dev.conll'
    test_file = '../data/ctb51-penn2malt-default/test.conll'
    embedding_file = '../data/embedding/giga.100.txt'


class Char_LSTM_CRF_Config(Config):
    model = 'Char_LSTM_CRF'
    net_file = './save/char_lstm_crf.pt'
    vocab_file = './save/vocab.pkl'

    word_hidden = 300
    char_hidden = 200
    layers = 2
    dropout = 0.55
    char_dim = 100
    word_dim = 100

    optimizer = 'adam'
    epoch = 100
    gpu = -1
    lr = 0.001
    batch_size = 50
    eval_batch = 100
    tread_num = 4
    decay = 0.05
    patience = 10
    shuffle = True


config = {
    'char_lstm_crf' : Char_LSTM_CRF_Config,
}