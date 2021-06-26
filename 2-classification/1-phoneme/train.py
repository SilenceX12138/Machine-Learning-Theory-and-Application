import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from config import SEED, learning_rate, model_path, num_epoch
from data.dataset import get_dataloader, load_train_data, split_val_data
from models.nn import Classifier
from utils.env import get_device, same_seeds


def train():
    model.train()  # set the model to training mode
    best_acc = 0.0  # procedural record variables
    for epoch in range(num_epoch):
        train_acc = 0.0
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            # get the index of the class with the highest probability
            # _.shape: 64 train_pred.shape: 64 outputs.shape: 64*39
            # _ is max value and train_pred is index
            _, train_pred = torch.max(outputs, 1)
            batch_loss.backward()
            optimizer.step()

            train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
            train_loss += batch_loss.item()

        val_acc, val_loss = validate()
        print(
            '[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'
            .format(epoch + 1, num_epoch, train_acc / train_cnt,
                    train_loss / len(train_loader), val_acc / val_cnt,
                    val_loss / len(val_loader)))
        # if the model improves, save a checkpoint at this epoch
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print('saving model with acc {:.3f}'.format(best_acc / val_cnt))


def validate():
    model.eval()  # set the model to evaluation mode
    val_acc = 0.0
    val_loss = 0.0
    # validation
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            # get the index of the class with the highest probability
            _, val_pred = torch.max(outputs, 1)
            val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
            val_loss += batch_loss.item()
    return val_acc, val_loss


if __name__ == '__main__':
    # fix random seed for reproducibility
    same_seeds(SEED)
    # get device
    device = get_device()
    print(f'DEVICE: {device}')
    # create model, define a loss function, and optimizer
    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # prepare data
    x, y = load_train_data()
    train_x, train_y, val_x, val_y = split_val_data(x, y)
    train_loader, val_loader = get_dataloader(train_x, train_y, val_x, val_y)
    train_cnt = len(train_x)  # length of dataloader equals to count of batches
    val_cnt = len(val_x)
    # start to train
    train()
