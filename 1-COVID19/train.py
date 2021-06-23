import os

import torch

from data.dataset import prep_dataloader
from models.nn import NeuralNet
from models.dnn import DeepNeuralNet
from param import *
from utils.device import get_device
from utils.visualize import plot_learning_curve, plot_pred


def train(tr_set, dv_set, model, config, device):
    ''' DNN training '''

    n_epochs = config['n_epochs']  # Maximum number of epochs

    # Setup optimizer
    optimizer = getattr(torch.optim,
                        config['optimizer'])(model.parameters(),
                                             **config['optim_hparas'])

    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}  # for recording training loss
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()  # set model to training mode
        for x, y in tr_set:  # iterate through the dataloader
            optimizer.zero_grad()  # set gradient to zero
            x, y = x.to(device), y.to(device)  # move data to device (cpu/cuda)
            pred = model(x)  # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
            mse_loss.backward()  # compute gradient (backpropagation)
            optimizer.step()  # update model with optimizer
            loss_record['train'].append(mse_loss.detach().cpu().item())

        # After each epoch, test your model on the validation (development) set.
        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            # Save model if your model improved
            min_mse = dev_mse
            print('epoch = {:4d}, loss = {:.4f}'.format(
                epoch + 1, min_mse))
            # torch.save(model.state_dict(), config['save_path'].format(
            #     model_name, min_mse))  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            # Stop training if your model stops improving for "config['early_stop']" epochs.
            break

    torch.save(model.state_dict(), config['save_path'].format(
        model_name, min_mse))  # Save model to specified path
    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record


def dev(dv_set, model, device):
    model.eval()  # set model to evalutation mode
    total_loss = 0
    for x, y in dv_set:  # iterate through the dataloader
        x, y = x.to(device), y.to(device)  # move data to device (cpu/cuda)
        with torch.no_grad():  # disable gradient calculation
            pred = model(x)  # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
        total_loss += mse_loss.detach().cpu().item() * len(
            x)  # accumulate loss
    total_loss = total_loss / len(dv_set.dataset)  # compute averaged loss

    return total_loss


if __name__ == '__main__':
    device = get_device()  # get the current available device ('cpu' or 'cuda')
    os.makedirs(
        'checkpoints/{}'.format(model_name),
        exist_ok=True)  # The trained model will be saved to ./checkpoints/
    tr_set = prep_dataloader(tr_path,
                             'train',
                             config['batch_size'],
                             target_only=target_only)
    dv_set = prep_dataloader(tr_path,
                             'dev',
                             config['batch_size'],
                             target_only=target_only)
    # Construct model and move to device
    if model_name == 'nn':
        model = NeuralNet(tr_set.dataset.dim).to(device)
    elif model_name == 'dnn':
        model = DeepNeuralNet(tr_set.dataset.dim).to(device)
    model_loss, model_loss_record = train(tr_set, dv_set, model, config,
                                          device)
    plot_learning_curve(model_loss_record, title='deep model')

    # model = NeuralNet(tr_set.dataset.dim).to(device)
    # ckpt = torch.load(config['save_path'].format(),
    #                   map_location='cpu')  # Load your best model
    # model.load_state_dict(ckpt)
    # plot_pred(dv_set, model, device)  # Show prediction on the validation set
