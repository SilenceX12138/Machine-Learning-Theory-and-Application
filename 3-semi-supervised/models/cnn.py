import torch.nn as nn

# The basic model here is simply a stack of convolutional layers followed by some fully-connected layers.
#
# Since there are three channels for a color image (RGB), the input channels of the network must be three.
# In each convolutional layer, typically the channels of inputs grow, while the height and width shrink (or remain unchanged, according to some hyperparameters like stride and padding).
#
# Before fed into fully-connected layers, the feature map must be flattened into a single one-dimensional vector (for each image).
# These features are then transformed by the fully-connected layers, and finally, we obtain the "logits" for each class.
#
# You are free to modify the model architecture here for further improvement.
# However, if you want to use some well-known architectures such as ResNet50, please make sure **NOT** to load the pre-trained weights.
# Using such pre-trained models is considered cheating and therefore you will be punished.
# Similarly, it is your responsibility to make sure no pre-trained weights are used if you use **torch.hub** to load any modules.
#
# For example, if you use ResNet-18 as your model:
#
# model = torchvision.models.resnet18(pretrained=**False**) → This is fine.
#
# model = torchvision.models.resnet18(pretrained=**True**)  → This is **NOT** allowed.


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # input image size: [3, 128, 128]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),
        )
        self.fc_layers = nn.Sequential(nn.Linear(256 * 8 * 8, 256), nn.ReLU(),
                                       nn.Linear(256, 256), nn.ReLU(),
                                       nn.Linear(256, 11))

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]
        # Extract features by convolutional layers.
        x = self.cnn_layers(x)
        # The extracted feature map must be flatten before going to fully-connected layers.
        x = x.flatten(1)
        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x