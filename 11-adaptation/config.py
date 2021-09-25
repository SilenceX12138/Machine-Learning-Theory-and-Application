from argparse import Namespace

import torch

env_config = Namespace(
    workspace='.',
    data_path='data/dataset',
    ckpt_path='checkpoint',
    device='cuda' if torch.cuda.is_available() else 'cpu',
)

train_config = Namespace(seed=2021, lamb=0.1)

test_config = Namespace(seed=2021, )
