import os
from argparse import Namespace

config = Namespace(
    workspace_dir='.',
    seed=2021,
    batch_size=64,
    z_dim=100,
    lr=1e-4,
    n_epoch=5,  # 50
    n_critic=0,  # 5
    clip_value=0.01,
)

project_path = Namespace(
    data_dir=os.path.join(config.workspace_dir, 'data', 'faces'),
    log_dir=os.path.join(config.workspace_dir, 'logs'),
    ckpt_dir=os.path.join(config.workspace_dir, 'checkpoints'),
)
