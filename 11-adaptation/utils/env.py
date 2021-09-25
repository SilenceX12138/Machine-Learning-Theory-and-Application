import os
import random

import numpy as np
import torch
from config import env_config


def build_dir():
    os.makedirs(os.path.join(env_config.workspace, env_config.ckpt_path),
                exist_ok=True)


def enable_reproduce(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
