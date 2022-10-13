import pytorch_lightning as pl
import torch
from torch import nn
from torchvision.models.mobilenet import mobilenet_v3_small


class MobileNetV3MNIST(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = mobilenet_v3_small(num_classes=10)
        self.model.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_no):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=0.005)
