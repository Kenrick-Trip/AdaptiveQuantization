import pytorch_lightning as pl
import torch
from torch import nn
from torchvision.models import resnet18
from torch.quantization import QuantStub, DeQuantStub


class ResNet18MNIST(pl.LightningModule):
    def __init__(self, quantize=False):
        super().__init__()
        self.model = resnet18(num_classes=10)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.loss = nn.CrossEntropyLoss()
        self.quantize = quantize
        if self.quantize:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

    def forward(self, x):
        if self.quantize:
            x = self.quant(x)
        x = self.model(x)
        if self.quatize:
            x = self.dequant(x)
        return x

    def training_step(self, batch, batch_no):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=0.005)
