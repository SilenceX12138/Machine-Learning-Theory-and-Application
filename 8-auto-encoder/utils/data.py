import os

import numpy as np
import torch
from config import env_config, test_config, train_config
from torch.utils.data import DataLoader, SequentialSampler

from data.FaceDataset import FaceDataset


def get_loader(mode: str):
    data_file = os.path.join(env_config.data_path, '{}ingset.npy'.format(mode))
    # set allow_pickle False for security purpose and data_file in .npy cannot be loaded
    data = np.load(data_file, allow_pickle=True)
    data = torch.tensor(data, dtype=torch.float32)
    data_set = FaceDataset(data)
    data_loader = DataLoader(data_set,
                             batch_size=test_config.batch_size
                             if mode == 'test' else train_config.batch_size,
                             shuffle=(mode == 'train'),
                             num_workers=0,
                             pin_memory=True)
    return data_loader


def IO_format(data: torch.Tensor, model_type: str):
    if model_type in ['cnn', 'vae', 'resnet']:
        img = data.float()
    elif model_type in ['fcn']:
        img = data.float()
        img = img.view(img.shape[0], -1)
    else:
        img = data[0]

    return img
