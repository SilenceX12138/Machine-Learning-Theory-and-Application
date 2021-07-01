import torch
from tqdm.auto import tqdm

from config import model_path, test_path
from data.dataset import FOOD11DataSet, get_dataloader
from models.cnn import Classifier
from utils.env import get_device


def test():
    model.eval()
    predictions = []
    for batch in tqdm(test_loader):
        imgs, labels = batch
        with torch.no_grad():
            logits = model(imgs.to(device))
        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    save_pred(predictions)


def save_pred(predictions):
    with open("predict.csv", "w") as f:
        f.write("Id,Category\n")
        for i, pred in enumerate(predictions):
            f.write(f"{i},{pred}\n")


if __name__ == '__main__':
    device = get_device()
    test_set = FOOD11DataSet(mode='test', path=test_path)
    test_loader = get_dataloader(test_set, mode='test')
    model = Classifier().to(device)
    # model = torchvision.models.resnet101().to(device)
    model.load_state_dict(torch.load(model_path))
    test()
