from utils.data import get_loader
from utils.env import enable_reproduce
from utils.model import model_function
from model.AutoEncoder import conv_autoencoder


def train():
    enable_reproduce()

    model = conv_autoencoder()
    train_loader = get_loader('train')

    model_function(model, train_loader)


if __name__ == '__main__':
    train()