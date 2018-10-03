import numpy as np
import torch
import torch.utils.data as Data
from torch.nn.utils.rnn import *


def collate_fn(data):
    batch = zip(*data)
    return tuple([torch.tensor(x) if len(x[0].size()) < 1 else pad_sequence(x, True) for x in batch])


class TensorDataSet(Data.Dataset):
    def __init__(self, *data):
        super(TensorDataSet, self).__init__()
        self.items = list(zip(*data))

    def __getitem__(self, index):
        return self.items[index]
    
    def __len__(self):
        return len(self.items)
    

class TensorDataLoader(object):
    def __init__(self, data, batch_size=1, shuffle=False, padding_value=0):
        self.raw_data = list(zip(*data))
        self.num = len(self.raw_data)
        self.padding_value = padding_value
        self.batch_size = batch_size
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.raw_data)
            self.batches = [self.raw_data[i:i+self.batch_size]
                            for i in range(0, self.num, self.batch_size)].__iter__()
        else:
            self.batches = [self.raw_data[i:i+batch_size]
                            for i in range(0, self.num, batch_size)].__iter__()

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.raw_data)
            self.batches = [self.raw_data[i:i+self.batch_size]
                            for i in range(0, self.num, self.batch_size)].__iter__()
        else:
            self.batches = [self.raw_data[i:i+batch_size]
                            for i in range(0, self.num, batch_size)].__iter__()
        return self

    def __next__(self):
        batch = next(self.batches)
        batch = zip(*batch)
        return tuple([torch.tensor(x) if len(x[0].size()) < 1 else pad_sequence(x, True, padding_value=self.padding_value) for x in batch])
