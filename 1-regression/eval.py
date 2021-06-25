import csv
from models.dnn import DeepNeuralNet
import torch

from param import *
from data.dataset import prep_dataloader
from models.nn import NeuralNet
from utils.device import get_device


def test(tt_set, model, device):
    model.eval()  # set model to evalutation mode
    preds = []
    for x in tt_set:  # iterate through the dataloader
        x = x.to(device)  # move data to device (cpu/cuda)
        with torch.no_grad():  # disable gradient calculation
            pred = model(x)  # forward pass (compute output)
            preds.append(pred.detach().cpu())  # collect prediction
    # concatenate all predictions and convert to a numpy array
    preds = torch.cat(preds, dim=0).numpy()
    return preds


def save_pred(preds, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    with open(file, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


if __name__ == '__main__':
    tt_set = prep_dataloader(tt_path,
                             'test',
                             config['batch_size'],
                             target_only=target_only)
    device = get_device()
    if model_name == 'nn':
        model = NeuralNet(tt_set.dataset.dim).to(device)
    elif model_name == 'dnn':
        model = DeepNeuralNet(tt_set.dataset.dim).to(device)
    ckpt = torch.load(config['save_path'].format(model_name, 0.74979),
                      map_location='cpu')  # Load your best model
    model.load_state_dict(ckpt)
    # predict COVID-19 cases with your model
    preds = test(tt_set, model, device)
    save_pred(preds, 'pred.csv')  # save prediction file to pred.csv