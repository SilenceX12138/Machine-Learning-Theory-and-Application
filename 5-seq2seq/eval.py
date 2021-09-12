import torch
import tqdm.auto as tqdm

import fairseq
from config import arch_args, config, task
from utils.env import enable_log, get_device
from utils.model import (build_model, inference_step, load_data_iterator,
                         try_load_checkpoint)


def main():
    logger = enable_log()
    device = get_device()
    model = build_model(arch_args, task).to(device)
    try_load_checkpoint(logger, model)

    idxs, hyps = pred(model, task, device)
    save_pred(idxs, hyps)


def pred(model, task, device, split="test"):
    task.load_dataset(split=split, epoch=1)
    itr = load_data_iterator(config.seed, task, split, 1, config.max_tokens,
                             config.num_workers).next_epoch_itr(shuffle=False)

    idxs = []
    hyps = []
    model.eval()
    progress = tqdm.tqdm(itr, desc=f"prediction")
    with torch.no_grad():
        for i, sample in enumerate(progress):
            # validation loss
            # move dict to a device(only `to()` method is not valid)
            sample = fairseq.utils.move_to_cuda(sample, device=device)

            # do inference
            s, h, r = inference_step(sample, model)

            hyps.extend(h)
            idxs.extend(list(sample['id']))

    return idxs, hyps


def save_pred(idxs, hyps, outfile="./prediction.txt"):
    # sort according to preprocess
    hyps = [x for _, x in sorted(zip(idxs, hyps))]
    with open(outfile, "w", encoding='utf-8') as f:
        for h in hyps:
            f.write(h + "\n")


if __name__ == '__main__':
    main()
