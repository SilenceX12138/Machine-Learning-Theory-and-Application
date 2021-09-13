import random

import numpy as np
import torch
from config import env_config, train_config


def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if train_config.fp16_training:
        from accelerate import Accelerator
        accelerator = Accelerator(fp16=True)
        device = accelerator.device
        return device, accelerator
    return device, None


def enable_reproduce():
    torch.manual_seed(env_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(env_config.seed)
        torch.cuda.manual_seed_all(env_config.seed)
    np.random.seed(env_config.seed)
    random.seed(env_config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
