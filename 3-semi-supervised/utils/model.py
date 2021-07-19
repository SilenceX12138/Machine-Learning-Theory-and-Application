import torchvision
from models.cnn import Classifier
from torch import nn


def set_parameter_requires_grad(model, feature_extract: bool):
    for param in model.parameters():
        param.requires_grad = feature_extract


def build_model(name: str, pre_trained: bool = False, num_classes: int = 11):
    if name == 'raw':
        model = Classifier()
    elif name == 'resnet':
        if pre_trained:
            model = torchvision.models.resnet18(pretrained=True)
            set_parameter_requires_grad(model, False)
            model.fc = nn.Linear(512, num_classes)  # resnet18
            # model.fc = nn.Linear(2048, num_classes) # resnet152
        else:
            model = torchvision.models.resnet18(num_classes=num_classes)
    else:
        raise Exception('No model named {} is found.'.format(name))

    return model
