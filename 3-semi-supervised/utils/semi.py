import torch
import torch.nn as nn
from data.dataset import FOOD11DataSet, get_dataloader
from torch.utils.data import ConcatDataset
from tqdm.auto import tqdm

from utils.env import get_device


def get_pseudo_labels(dataset: FOOD11DataSet, model, threshold=0.75):
    device = get_device()
    data_loader = get_dataloader(dataset, mode='train')
    softmax = nn.Softmax(dim=1)
    model.eval()
    pseudo_data_index_list = []
    pseudo_label_list = []
    for batch in tqdm(data_loader):
        img, _, idx = batch
        img, idx = img.to(device), idx.to(device)
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img)
        probs, pred = torch.max(softmax(logits), dim=1)
        valid_list = torch.nonzero(probs > threshold).squeeze(1)
        pseudo_data_index_list.extend(
            torch.index_select(idx, dim=0, index=valid_list).cpu())
        pseudo_label_list.extend(
            torch.index_select(pred, dim=0, index=valid_list).cpu())

    model.train()
    pseudo_cnt = len(pseudo_data_index_list)
    if pseudo_cnt > 0:
        print('Add {} new samples into train set.'.format(pseudo_cnt))
        return dataset.sample_dataset(mode='train',
                                      index_list=pseudo_data_index_list,
                                      label_list=pseudo_label_list)
    else:
        return None


def update_train_set(model, labeled_set, unlabeled_set):
    pseudo_set = get_pseudo_labels(unlabeled_set, model)
    if pseudo_set is not None:
        # all sub datasets need to be of same type
        labeled_set = ConcatDataset([labeled_set, pseudo_set])
    # original labeled_set won't be influenced
    return labeled_set
