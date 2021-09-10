import os

import numpy as np
import torch
from config import project_path


def enable_reproduce(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_dir():
    os.makedirs(project_path.log_dir, exist_ok=True)
    os.makedirs(project_path.ckpt_dir, exist_ok=True)
