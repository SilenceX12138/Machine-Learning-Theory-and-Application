import os

import cv2
import numpy as np
from config import env_config
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


def get_dataset():
    source_transform = transforms.Compose([
        # Turn RGB to grayscale. (Bacause Canny do not support RGB images.)
        transforms.Grayscale(),
        # cv2 do not support skimage.Image, so we transform it to np.array,
        # and then adopt cv2.Canny algorithm.
        transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
        # Transform np.array back to the skimage.Image.
        transforms.ToPILImage(),
        # 50% Horizontal Flip. (For Augmentation)
        transforms.RandomHorizontalFlip(),
        # Rotate +- 15 degrees. (For Augmentation), and filled with zero
        # if there's empty pixel after rotation.
        transforms.RandomRotation(15, fill=(0, )),
        # Transform to tensor for model inputs.
        transforms.ToTensor(),
    ])
    target_transform = transforms.Compose([
        # Turn RGB to grayscale.
        transforms.Grayscale(),
        # Resize: size of source data is 32x32, thus we need to
        #  enlarge the size of target data from 28x28 to 32x32。
        transforms.Resize((32, 32)),
        # 50% Horizontal Flip. (For Augmentation)
        transforms.RandomHorizontalFlip(),
        # Rotate +- 15 degrees. (For Augmentation), and filled with zero
        # if there's empty pixel after rotation.
        transforms.RandomRotation(15, fill=(0, )),
        # Transform to tensor for model inputs.
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    source_dataset = ImageFolder(os.path.join(env_config.workspace,
                                              env_config.data_path,
                                              'train_data'),
                                 transform=source_transform)
    target_dataset = ImageFolder(os.path.join(env_config.workspace,
                                              env_config.data_path,
                                              'test_data'),
                                 transform=target_transform)
    test_dataset = ImageFolder(os.path.join(env_config.workspace,
                                            env_config.data_path, 'test_data'),
                               transform=test_transform)

    return source_dataset, target_dataset, test_dataset


def get_loader(mode: str):
    source_dataset, target_dataset, test_dataset = get_dataset()
    if mode == 'train':
        source_loader = DataLoader(source_dataset, batch_size=32, shuffle=True)
        target_loader = DataLoader(target_dataset, batch_size=32, shuffle=True)
        return source_loader, target_loader
    elif mode == 'test':
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        return test_loader, None
