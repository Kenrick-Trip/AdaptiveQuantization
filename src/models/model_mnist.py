import pytorch_lightning as pl
import torch
from torch import nn
from torchvision.models import resnet18


class ModelForMNIST(pl.LightningModule):
    def __init__(self):
        super().__init__()

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
