import os

import torch

from config import env_config, train_config
from model.Classifier import DomainClassifier
from model.Extractor import FeatureExtractor
from model.Predictor import LabelPredictor
from utils.data import get_loader
from utils.env import build_dir, enable_reproduce
from utils.model import train_epoch, train_epoch_GRL


def train():
    enable_reproduce(train_config.seed)
    build_dir()

    source_loader, target_loader = get_loader('train')

    extractor = FeatureExtractor().to(env_config.device)
    predictor = LabelPredictor().to(env_config.device)
    classifier = DomainClassifier().to(env_config.device)

    # train 200 epochs
    for epoch in range(200):
        # train_D_loss, train_F_loss, train_acc = train_epoch(
        #     source_loader, target_loader, extractor, predictor, classifier)
        train_D_loss, train_F_loss, train_acc = train_epoch_GRL(
            source_loader, target_loader, extractor, predictor, classifier)

        torch.save(
            extractor.state_dict(),
            os.path.join(env_config.workspace, env_config.ckpt_path,
                         'extractor_model.pth'))
        torch.save(
            predictor.state_dict(),
            os.path.join(env_config.workspace, env_config.ckpt_path,
                         'predictor_model.pth'))

        print(
            'epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'
            .format(epoch, train_D_loss, train_F_loss, train_acc))


if __name__ == '__main__':
    train()
