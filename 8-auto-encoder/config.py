from argparse import Namespace
import torch

env_config = Namespace(
    workspace = '.',
    data_path = 'data/dataset',
    ckpt_path = 'checkpoint',
    seed      = 2021,
    device    = 'cuda' if torch.cuda.is_available() else 'cpu',
)

train_config = Namespace(
    num_epochs    = 50,
    model_type    = 'cnn', # 'cnn', 'vae', 'resnet'
    batch_size    = 256,
    learning_rate = 1e-3,
)

test_config = Namespace(
    pred_path  = 'prediction.csv',
    model_type = 'cnn',
    batch_size = 256,
)
