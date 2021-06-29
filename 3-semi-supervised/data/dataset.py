import torch
from config import batch_size
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import DatasetFolder
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


def load_img(x):
    return Image.open(x)


def get_labeled_set():
    label_set = DatasetFolder("data/training/labeled",
                              loader=load_img,
                              extensions="jpg",
                              transform=train_tfm)

    # change DatasetFolder into TensorDataset for semi-supervised learning when concatenating dataset
    data = [sample[0] for sample in list(label_set)]
    label = label_set.targets

    return TensorDataset(torch.stack(data), torch.Tensor(label).long())


def get_unlabeled_set():
    # The argument "loader" tells how torchvision reads the data.
    unlabeled_set = DatasetFolder("data/training/unlabeled",
                                  loader=load_img,
                                  extensions="jpg",
                                  transform=train_tfm)

    return unlabeled_set


def get_valid_set():
    # Windows doesn't support lambda function loader
    valid_set = DatasetFolder("data/validation",
                              loader=load_img,
                              extensions="jpg",
                              transform=test_tfm)
    return valid_set


def get_test_set():
    test_set = DatasetFolder("data/testing",
                             loader=load_img,
                             extensions="jpg",
                             transform=test_tfm)
    return test_set


def get_dataloader(dataset, mode='train'):
    """
    set data set and loader apart for flexible transforms
    """
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=(mode == 'train'),
                            pin_memory=True)
    return dataloader
