from torch import nn
from torchvision.models import convnext_small

from models.model_mnist import ModelForMNIST


class GoogLeNetMNIST(ModelForMNIST):
    def __init__(self):
        super().__init__()
        self.model = GoogLeNet(num_classes=10)
        self.model.conv1.conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
