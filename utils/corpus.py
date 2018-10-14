class Corpus(object):
    def __init__(self, filename=None, lower=False):
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
                    if lower:
                        sentence.append(conll[1].lower())
                    else:
                        sentence.append(conll[1])
                    sequence.append(conll[3])
                    self.word_num += 1
        print('%s : sentences:%d，words:%d' % (filename, self.sentence_num, self.word_num))