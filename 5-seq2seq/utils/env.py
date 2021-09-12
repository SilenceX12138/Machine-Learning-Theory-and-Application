import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from config import config
import fairseq


def enable_reproduce(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def enable_log():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level="INFO",  # "DEBUG" "WARNING" "ERROR"
        stream=sys.stdout,
    )
    proj = "seq2seq.translation"
    logger = logging.getLogger(proj)
    if config.use_wandb:
        import wandb
        wandb.init(project=proj, name=Path(config.savedir).stem, config=config)
    return logger


def get_device():
    cuda_env = fairseq.utils.CudaEnvironment()
    fairseq.utils.CudaEnvironment.pretty_print_cuda_env_list([cuda_env])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return device