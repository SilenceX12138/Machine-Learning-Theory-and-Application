import os
import random

import numpy as np
import torch
from config import env_config


def build_dir():
    os.makedirs(env_config.cpkt_path, exist_ok=True)


def enable_reproduce():
    random.seed(env_config.seed)
    np.random.seed(env_config.seed)
    torch.manual_seed(env_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(env_config.seed)
        torch.cuda.manual_seed_all(env_config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
