import os

import numpy as np
import torch
import torch.nn as nn
from config import env_config, train_config
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model.AutoEncoder import VAE, Resnet, conv_autoencoder, fcn_autoencoder
from utils.data import IO_format


def build_model(model_type: str):
    model_classes = {
        'resnet': Resnet(),
        'fcn': fcn_autoencoder(),
        'cnn': conv_autoencoder(),
        'vae': VAE(),
    }
    model = model_classes[model_type]
    return model


def model_function(model: nn.Module, train_loader: DataLoader):
    print('Start Training Model')
    model.train()

    model.to(env_config.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=train_config.learning_rate)

    best_loss = np.inf
    for epoch in range(train_config.num_epochs):
        tot_loss = []
        for data in tqdm(train_loader):
            img = IO_format(data, train_config.model_type)
            img = img.to(env_config.device)
            # 3*64*64 to 3*64*64
            output = model(img)
            if train_config.model_type in ['vae']:
                loss = loss_vae(output[0], img, output[1], output[2],
                                criterion)
            else:
                loss = criterion(output, img)
            tot_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mean_loss = np.mean(tot_loss)
        print('Epoch {:03} | Loss: {:7.4f}'.format(epoch, mean_loss))
        if mean_loss < best_loss:
            best_loss = mean_loss
            save_model(model, mean_loss)


def save_model(model: nn.Module, mean_loss: float):
    print('Save model with mean loss: {:7.4f}'.format(mean_loss))
    model_path = os.path.join(env_config.workspace, env_config.ckpt_path,
                              train_config.model_type)
    os.makedirs(model_path, exist_ok=True)
    torch.save(model, os.path.join(model_path, '{:.4f}.pth'.format(mean_loss)))


def loss_vae(recon_x, x, mu, logvar, criterion):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    mse = criterion(recon_x, x)  # mse loss
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return mse + KLD
