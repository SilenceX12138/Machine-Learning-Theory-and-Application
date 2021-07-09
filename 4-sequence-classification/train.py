from models.Conformer import ConvClassifier
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from config import path_config, train_config
from data.dataset import get_dataloader
from models.Attention import Classifier
from models.AttentionStack import AttentionStack
from utils.env import (build_dir, get_cosine_schedule_with_warmup, get_device,
                       model_fn, valid)


def parse_args():
    """arguments"""
    config = {
        "data_dir": path_config['data_dir'],
        "save_path": path_config['save_path'],
        "batch_size": train_config['batch_size'],
        "n_workers": train_config['n_workers'],
        "valid_steps": train_config['valid_steps'],
        "warmup_steps": train_config['warmup_steps'],
        "save_steps": train_config['save_steps'],
        "total_steps": train_config['total_steps'],
        "model_name": train_config['model_name'],
    }

    return config


def main(data_dir, save_path, batch_size, n_workers, valid_steps, warmup_steps,
         total_steps, save_steps, model_name):
    build_dir(model_name)
    device = get_device()
    train_loader, valid_loader, speaker_num = get_dataloader(
        data_dir, batch_size, n_workers)
    train_iterator = iter(train_loader)
    print(f"[Info]: Finish loading data!", flush=True)

    if model_name == 'attention':
        model = Classifier(n_spks=speaker_num).to(device)
    elif model_name == 'attentionstack':
        model = AttentionStack(n_spks=speaker_num).to(device)
    elif model_name == 'conformer':
        model = ConvClassifier(n_spks=speaker_num).to(device)
    else:
        raise Exception(
            'Undefined network type {} is found.'.format(model_name))

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps,
                                                total_steps)
    print("[Info]: Finish creating model {}!".format(model_name), flush=True)

    train(model, train_loader, valid_loader, train_iterator, criterion,
          optimizer, scheduler, valid_steps, total_steps, device, save_steps,
          save_path, model_name)


def train(model, train_loader, valid_loader, train_iterator, criterion,
          optimizer, scheduler, valid_steps, total_steps, device, save_steps,
          save_path, model_name):

    model.train()
    best_accuracy = -1.0
    best_state_dict = None

    pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    for step in range(total_steps):
        # Get data
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss, accuracy = model_fn(batch, model, criterion, device)
        batch_loss = loss.item()
        batch_accuracy = accuracy.item()

        # Updata model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log
        pbar.update()
        pbar.set_postfix(
            loss=f"{batch_loss:.2f}",
            accuracy=f"{batch_accuracy:.2f}",
            step=step + 1,
        )

        # Do validation
        if (step + 1) % valid_steps == 0:
            pbar.close()

            valid_accuracy = valid(valid_loader, model, criterion, device)

            # keep the best model
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_state_dict = model.state_dict()

            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

        # Save the best model so far.
        if (step + 1) % save_steps == 0 and best_state_dict is not None:
            torch.save(best_state_dict,
                       save_path.format(model_name, best_accuracy))
            pbar.write(
                f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})"
            )

    pbar.close()


if __name__ == "__main__":
    main(**parse_args())
