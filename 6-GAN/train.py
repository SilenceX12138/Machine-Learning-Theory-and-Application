import torch
from torch.autograd import Variable

from config import config, project_path
from data.CrypkoDataset import get_dataloader, get_dataset
from models.Discriminator import Discriminator
from models.Generator import Generator
from utils.env import build_dir, enable_reproduce
from utils.model import model_function


def train():
    enable_reproduce(config.seed)
    build_dir()

    # Training hyperparameters
    z_sample = Variable(torch.randn(100, config.z_dim)).cuda()

    # Model
    G = Generator(in_dim=config.z_dim).cuda()
    D = Discriminator(3).cuda()
    G.train()
    D.train()

    # DataLoader
    dataset = get_dataset(project_path.data_dir)
    dataloader = get_dataloader(dataset)

    model_function(dataloader, G, D, z_sample)


if __name__ == '__main__':
    train()
