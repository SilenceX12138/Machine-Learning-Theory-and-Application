import numpy as np
import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, model_path
from data.dataset import TIMITDataset, load_test_data
from models.nn import Classifier
from utils.env import get_device


def test():
    model.eval()  # set the model to evaluation mode
    predict = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # get the index of the class with the highest probability
            _, test_pred = torch.max(outputs, 1)
            for y in test_pred.cpu().numpy():
                predict.append(y)
    save_pred(predict)


def save_pred(predict):
    with open('prediction.csv', 'w') as f:
        f.write('Id,Class\n')
        for i, y in enumerate(predict):
            f.write('{},{}\n'.format(i, y))


if __name__ == '__main__':
    device = get_device()
    # create testing dataset
    test_data = load_test_data()
    test_set = TIMITDataset(test_data, None)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    # create model and load weights from checkpoint
    model = Classifier().to(device)
    model.load_state_dict(torch.load(model_path))
    # start to test
    test()