import pytorch_lightning as pl
import torch
from torch import nn
from torchvision.models import resnet18


class ResNet18MNIST(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = resnet18(num_classes=10)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_no):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=0.005)
