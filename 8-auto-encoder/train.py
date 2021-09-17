from config import train_config
from utils.data import get_loader
from utils.env import build_dir, enable_reproduce
from utils.model import build_model, model_function


def train():
    enable_reproduce()
    build_dir()

    model = build_model(train_config.model_type)
    train_loader = get_loader('train')

    model_function(model, train_loader)


if __name__ == '__main__':
    train()
