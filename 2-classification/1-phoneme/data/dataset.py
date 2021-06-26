import numpy as np
import torch
from config import VAL_RATIO, BATCH_SIZE, data_path
from torch.utils.data import DataLoader, Dataset


class TIMITDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(np.int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


def load_train_data():
    # print('Loading data ...')

    train = np.load(data_path + 'train_11.npy')
    train_label = np.load(data_path + 'train_label_11.npy')

    print('Size of training data: {}'.format(train.shape))
    return train, train_label


def load_test_data():
    test = np.load(data_path + 'test_11.npy')
    print('Size of testing data: {}'.format(test.shape))
    return test


def split_val_data(train, train_label):
    percent = int(train.shape[0] * (1 - VAL_RATIO))
    train_x, train_y, val_x, val_y = train[:percent], train_label[:percent],\
                                     train[percent:], train_label[percent:]
    print('Size of training set: {}'.format(train_x.shape))
    print('Size of validation set: {}'.format(val_x.shape))
    return train_x, train_y, val_x, val_y


def get_dataloader(train_x, train_y, val_x, val_y):
    train_set = TIMITDataset(train_x, train_y)
    val_set = TIMITDataset(val_x, val_y)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True)  #only shuffle the training data
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader
