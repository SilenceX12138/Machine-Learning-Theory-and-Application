import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import env_config, test_config
from model.Extractor import FeatureExtractor
from model.Predictor import LabelPredictor
from utils.data import get_loader
from utils.env import enable_reproduce


def eval(extractor_name: str, predictor_name: str):
    enable_reproduce(test_config.seed)
    predictor_path = os.path.join(env_config.workspace, env_config.ckpt_path,
                                  predictor_name)
    extractor_path = os.path.join(env_config.workspace, env_config.ckpt_path,
                                  extractor_name)

    test_loader, _ = get_loader('test')

    feature_extractor = FeatureExtractor()
    feature_extractor.load_state_dict(torch.load(extractor_path))
    label_predictor = feature_extractor.to(env_config.device)
    label_predictor = LabelPredictor()
    label_predictor.load_state_dict(torch.load(predictor_path))
    label_predictor = label_predictor.to(env_config.device)

    pred_list = pred(feature_extractor, label_predictor, test_loader)
    save_pred(pred_list)


def pred(feature_extractor: nn.Module, label_predictor: nn.Module,
         test_loader: DataLoader):
    label_predictor.eval()
    feature_extractor.eval()

    result = []
    for test_data, _ in tqdm(test_loader):
        test_data = test_data.to(env_config.device)

        class_logits = label_predictor(feature_extractor(test_data))

        x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
        result.append(x)
    return result


def save_pred(pred_list: list):
    result = np.concatenate(pred_list)

    # Generate your submission
    df = pd.DataFrame({'id': np.arange(0, len(result)), 'label': result})
    df.to_csv('prediction.csv', index=False)


if __name__ == '__main__':
    eval('extractor_model.bin', 'predictor_model.bin')
