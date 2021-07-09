import csv
import json
from models.Conformer import ConvClassifier
from models.AttentionStack import AttentionStack
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import path_config, test_config
from data.dataset import InferenceDataset, inference_collate_batch
from models.Attention import Classifier
from utils.env import get_device


def parse_args():
    """arguments"""
    config = {
        "data_dir":
        path_config['data_dir'],
        "model_path":
        path_config['model_path'].format(test_config['model_name'], '0.74035'),
        "output_path":
        path_config['output_path'],
        'model_name':
        test_config['model_name'],
    }

    return config


def main(data_dir, model_path, output_path, model_name):
    """Main function."""
    device = get_device()

    mapping_path = Path(data_dir) / "mapping.json"
    mapping = json.load(mapping_path.open())

    dataset = InferenceDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # can only be set to 1
        shuffle=False,
        drop_last=False,
        # num_workers=8,
        collate_fn=inference_collate_batch,
    )
    print(f"[Info]: Finish loading data!", flush=True)

    speaker_num = len(mapping["id2speaker"])
    if model_name == 'attention':
        model = Classifier(n_spks=speaker_num).to(device)
    elif model_name == 'attentionstack':
        model = AttentionStack(n_spks=speaker_num).to(device)
    elif model_name == 'conformer':
        model = ConvClassifier(n_spks=speaker_num).to(device)
    else:
        raise Exception(
            'Undefined network type {} is found.'.format(model_name))
    model.load_state_dict(torch.load(model_path))
    print("[Info]: Finish creating model {}!".format(model_name), flush=True)

    results = pred(model, dataloader, device, mapping)
    save_pred(output_path, results)


def pred(model, dataloader, device, mapping):
    model.eval()
    results = [["Id", "Category"]]
    for feat_paths, mels in tqdm(dataloader):
        with torch.no_grad():
            mels = mels.to(device)
            outs = model(mels)
            preds = outs.argmax(1).cpu().numpy()
            for feat_path, pred in zip(feat_paths, preds):
                results.append([feat_path, mapping["id2speaker"][str(pred)]])
    return results


def save_pred(output_path, results):
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)


if __name__ == "__main__":
    main(**parse_args())
