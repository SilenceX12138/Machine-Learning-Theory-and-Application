import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import env_config, test_config
from utils.data import IO_format, get_loader
from utils.env import enable_reproduce


def eval(model_name: str):
    enable_reproduce()
    test_loader = get_loader('test')
    model_path = os.path.join(env_config.workspace, env_config.ckpt_path,
                              test_config.model_type, model_name)
    model = torch.load(model_path)

    pred_list = pred(model, test_loader)
    save_pred(pred_list)


def pred(model: nn.Module, test_loader: DataLoader):
    model.eval()

    model = model.to(env_config.device)
    criterion = nn.MSELoss(reduction='none')
    abnormality_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            data = data.to(env_config.device)
            img = IO_format(data, test_config.model_type).to(env_config.device)
            output = model(img)
            output = IO_format(output, test_config.model_type)
            if test_config.model_type in ['fcn']:
                loss = criterion(output, img).sum(-1)
            else:
                # criterion(output, img): 256*3*64*64
                # loss: [256]
                loss = criterion(output, img).sum([1, 2, 3])
            abnormality_list.append(loss)
    return abnormality_list


def save_pred(pred_list: list, test=None):
    pred_list = torch.cat(pred_list,
                          axis=0)  # 27 tensors of [128] to 1 tensor of [19999]
    # Due to MSE loss so we can take sqrt, but it doesn't influence AUC calculation.
    # pred_list = torch.sqrt(pred_list).cpu().numpy()
    pred_list = pred_list.cpu().numpy()
    df = pd.DataFrame(pred_list, columns=['Predicted'])
    df.to_csv(test_config.pred_path, index_label='Id')


if __name__ == '__main__':
    eval('0.0074.pth')
