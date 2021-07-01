import os

import torch
from config import batch_size
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms

train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomCrop(128, padding=16),
    transforms.RandomRotation(90),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307, ),
        (0.3081, )),  # normalize should be executed after ToTensor()
])

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307, ),
        (0.3081, )),  # normalize should be executed after ToTensor()
])


class FOOD11DataSet(Dataset):
    def __init__(self, mode, path=None, data=None, label=None) -> None:
        super().__init__()
        self.mode = mode

        # Construct a dataset with given data
        if data != None and label != None:
            self.data = data
            self.label = label
            return

        # Construct a dataset with given path
        if mode in ['train', 'semi']:
            self.transform = train_tfm
        else:
            self.transform = test_tfm

        self.data = []
        self.label = []
        food_list = os.listdir(path)
        for food in food_list:
            img_path = path + food + '/'
            img_list = os.listdir(img_path)
            for img in img_list:
                self.data.append(img_path + img)
                self.label.append(int(food))

        self.label = torch.LongTensor(self.label)

    def __getitem__(self, index):
        # transfrom the image every time when loading
        img = Image.open(self.data[index])
        if self.mode == 'train':
            return train_tfm(img), self.label[index]
        elif self.mode == 'semi':
            return train_tfm(img), self.label[index], index
        else:
            return test_tfm(img), self.label[index]

    def __len__(self):
        return len(self.data)

    def sample_dataset(self, mode, index_list, label_list=None):
        """
        return a new FOOD11Dataset according to index
        """
        sample_data = [self.data[index] for index in index_list]
        if label_list is None:
            sample_label = [self.label[index] for index in index_list]
        else:
            sample_label = label_list
        return FOOD11DataSet(mode=mode, data=sample_data, label=sample_label)

    def drop_dataset(self, mode, index_list):
        """
        return a new dataset that drops data and labels according to index_list
        """
        left_data = [
            self.data[i] for i in range(len(self.data)) if i not in index_list
        ]
        left_label = [
            self.label[i] for i in range(len(self.label))
            if i not in index_list
        ]
        return FOOD11DataSet(mode=mode, data=left_data, label=left_label)


def get_dataloader(dataset, mode='train'):
    """
    set data set and loader apart for flexible transforms
    """
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=(mode == 'train'),
                            pin_memory=True)
    return dataloader
