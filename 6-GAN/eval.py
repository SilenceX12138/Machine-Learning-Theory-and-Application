import os
from utils.env import enable_reproduce

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable

from config import config, project_path
from models.Generator import Generator


def eval(model_name: str = 'G.pth.bak'):
    enable_reproduce(config.seed)
    G = Generator(config.z_dim).cuda()
    G.load_state_dict(
        torch.load(os.path.join(project_path.ckpt_dir, model_name)))

    imgs_sample = gene_img(G, 200)
    show_img(imgs_sample, 20)
    save_img(imgs_sample)


def gene_img(G: nn.Module, n_output: int = 100):
    G.eval()
    # Generate 100 images and make a grid to save them.
    z_sample = Variable(torch.randn(n_output, config.z_dim)).cuda()
    imgs_sample = (G(z_sample).data + 1) / 2.0
    filename = os.path.join(project_path.log_dir, 'result.jpg')
    torchvision.utils.save_image(imgs_sample, filename, nrow=10)
    return imgs_sample


def show_img(imgs_sample, cnt: int):
    # Show cnt of the images.
    grid_img = torchvision.utils.make_grid(imgs_sample[:cnt].cpu(), nrow=10)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()


def save_img(imgs_sample):
    # Save the generated images.
    os.makedirs('output', exist_ok=True)
    for i in range(len(imgs_sample)):
        # linear map rgb values into [0,255]
        torchvision.utils.save_image(imgs_sample[i], f'output/{i+1}.jpg')


if __name__ == '__main__':
    eval('G.pth.bak')
