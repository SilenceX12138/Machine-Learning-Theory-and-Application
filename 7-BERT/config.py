import torch
from argparse import Namespace

env_config = Namespace(seed=2021, )

train_config = Namespace(
    # Change "fp16_training" to True to support automatic mixed precision training (fp16)
    fp16_training=False,
    batch_size=16,
    num_epoch=10,
    validation=True,
    logging_step=100,
    learning_rate=1e-4, # lr should be very tiny when fine tuning
)
