import torch
import torch.nn as nn
import torchvision
from tqdm.auto import tqdm

from config import (SEED, do_semi, is_transfer, learning_rate, model_path,
                    n_epochs, weight_decay)
from data.dataset import (get_dataloader, get_labeled_set, get_unlabeled_set,
                          get_valid_set)
from models.cnn import Classifier
from utils.env import get_device, same_seeds
from utils.semi import update_train_set


def train(model, labeled_set, unlabeled_set, valid_set):
    model.train()
    best_acc = 0.0
    for epoch in range(n_epochs):
        train_loss = []
        train_accs = []
        if do_semi and best_acc > 0.4:
            train_set = update_train_set(model, labeled_set, unlabeled_set)
        else:
            train_set = labeled_set
        train_loader = get_dataloader(train_set)
        for batch in tqdm(train_loader):
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(),
                                                 max_norm=10)
            optimizer.step()
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        valid_acc, valid_loss = val(valid_set)
        print(
            f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}\n"
            +
            f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}"
        )
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), model_path)
            print('saving model with acc {:.3f}'.format(best_acc))


def val(valid_set):
    model.eval()
    valid_loader = get_dataloader(valid_set, mode='valid')
    valid_loss = []
    valid_accs = []
    for batch in tqdm(valid_loader):
        imgs, labels = batch
        with torch.no_grad():
            logits = model(imgs.to(device))
        loss = criterion(logits, labels.to(device))
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        valid_loss.append(loss.item())
        valid_accs.append(acc)
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    return valid_acc, valid_loss


if __name__ == '__main__':
    same_seeds(SEED)
    device = get_device()
    # prepare data
    labeled_set = get_labeled_set()
    unlabeled_set = get_unlabeled_set()
    valid_set = get_valid_set()
    # Initialize a model, and put it on the device specified.
    model = Classifier().to(device)
    # model = torchvision.models.resnet101().to(device)
    if is_transfer:
        model.load_state_dict(torch.load(model_path))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    train(model, labeled_set, unlabeled_set, valid_set)
