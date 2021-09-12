import matplotlib.pyplot as plt
import numpy as np
import torch

from config import arch_args, config, task
from utils.criterion import LabelSmoothedCrossEntropyCriterion
from utils.env import enable_log, enable_reproduce, get_device
from utils.model import (build_model, load_data_iterator, train_one_epoch,
                         try_load_checkpoint, validate_and_save)
from utils.optim import NoamOpt


def main():
    logger = enable_log()
    device = get_device()
    enable_reproduce(config.seed)

    model = build_model(arch_args, task)
    logger.info(model)
    criterion = LabelSmoothedCrossEntropyCriterion(
        smoothing=0.1,
        ignore_index=task.target_dictionary.pad(),
    )
    optimizer = NoamOpt(model_size=arch_args.encoder_embed_dim,
                        factor=config.lr_factor,
                        warmup=config.lr_warmup,
                        optimizer=torch.optim.AdamW(model.parameters(),
                                                    lr=0,
                                                    betas=(0.9, 0.98),
                                                    eps=1e-9,
                                                    weight_decay=0.0001))
    # plt.plot(np.arange(1, 100000),
    #          [optimizer.rate(i) for i in range(1, 100000)])
    # plt.legend([f"{optimizer.model_size}:{optimizer.warmup}"])
    # plt.show()

    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("encoder: {}".format(model.encoder.__class__.__name__))
    logger.info("decoder: {}".format(model.decoder.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info("optimizer: {}".format(optimizer.__class__.__name__))
    logger.info("num. model params: {:,} (num. trained: {:,})".format(
        # numel returns the number count within an array
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))
    logger.info(
        f"max tokens per batch = {config.max_tokens}, accumulate steps = {config.accum_steps}"
    )
    logger.info("loading data for epoch 1")
    task.load_dataset(
        split="train", epoch=1,
        combine=True)  # combine if you have back-translation data.
    task.load_dataset(split="valid", epoch=1)

    train(logger, model, criterion, optimizer, device)


def train(logger, model, criterion, optimizer, device):
    model = model.to(device=device)
    criterion = criterion.to(device=device)

    epoch_itr = load_data_iterator(config.seed, task, "train",
                                   config.start_epoch, config.max_tokens,
                                   config.num_workers)
    try_load_checkpoint(logger, model, optimizer, name=config.resume)
    while epoch_itr.next_epoch_idx <= config.max_epoch:
        # train for one epoch
        train_one_epoch(logger, epoch_itr, model, task, criterion, optimizer,
                        device, config.accum_steps)
        stats = validate_and_save(logger,
                                  model,
                                  task,
                                  criterion,
                                  optimizer,
                                  device,
                                  epoch=epoch_itr.epoch)
        logger.info("end of epoch {}".format(epoch_itr.epoch))
        epoch_itr = load_data_iterator(config.seed, task, "train",
                                       epoch_itr.next_epoch_idx,
                                       config.max_tokens, config.num_workers)


if __name__ == '__main__':
    main()
