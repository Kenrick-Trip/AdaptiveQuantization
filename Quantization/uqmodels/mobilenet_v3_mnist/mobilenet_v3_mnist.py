from torch import nn
from torchvision.models.mobilenet import mobilenet_v3_small

from uqmodels.mobilenet_v3_mnist.model_mnist import ModelForMNIST


class MobileNetV3MNIST(ModelForMNIST):
    def __init__(self):
        super().__init__()
        self.model = mobilenet_v3_small(num_classes=10)
        self.model.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

