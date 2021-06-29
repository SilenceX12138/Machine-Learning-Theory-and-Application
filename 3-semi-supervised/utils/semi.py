import numpy as np
import torch
import torch.nn as nn
from data.dataset import get_dataloader
from torch.utils.data import ConcatDataset, TensorDataset
from tqdm.auto import tqdm

from utils.env import get_device


def get_pseudo_labels(dataset, model, threshold=0.75):
    device = get_device()
    data_loader = get_dataloader(dataset, mode='test')
    softmax = nn.Softmax(dim=1)
    model.eval()
    pseudo_data_list = []
    pseudo_label_list = []
    for batch in tqdm(data_loader):
        img, _ = batch
        img = img.to(device)
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img)
        probs, pred = torch.max(softmax(logits), dim=1)
        valid_list = torch.nonzero(probs > threshold).squeeze(1)
        pseudo_data_list.extend(
            torch.index_select(img, dim=0, index=valid_list).cpu())
        pseudo_label_list.extend(
            torch.index_select(pred, dim=0, index=valid_list).cpu())

    model.train()
    pseudo_cnt = len(pseudo_data_list)
    if pseudo_cnt > 0:
        print('Add {} new samples into train set.'.format(
            len(pseudo_data_list)))
        return TensorDataset(torch.stack(pseudo_data_list),
                             torch.Tensor(pseudo_label_list).long())
    else:
        return None


def update_train_set(model, labeled_set, unlabeled_set):
    pseudo_set = get_pseudo_labels(unlabeled_set, model)
    if pseudo_set is not None:
        # all sub datasets need to be of same type
        concat_dataset = ConcatDataset([labeled_set, pseudo_set])
    return concat_dataset
