import torch
import wandb
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from tqdm.autonotebook import tqdm
from sklearn.metrics import accuracy_score, classification_report
from pytorch_lightning.loggers import WandbLogger

from models.googlenet_mnist import GoogLeNetMNIST
from models.mobilenet_v3_mnist import MobileNetV3MNIST
from models.resnet18_mnist import ResNet18MNIST


def train(model: pl.LightningModule):
    batch_size = 32
    train_ds = MNIST("mnist", train=True, download=True, transform=ToTensor())
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    wlogger = WandbLogger()

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu",
        devices=[0],
        logger=wlogger
    )
    trainer.fit(model, train_dl)
    trainer.save_checkpoint(model.__class__.__name__+".pt")


def get_prediction(x, model: pl.LightningModule):
    model.freeze()  # prepares model for predicting
    probabilities = torch.softmax(model(x), dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class, probabilities


def inference(inference_model: pl.LightningModule):
    batch_size = 32
    test_ds = MNIST("mnist", train=False, download=True, transform=ToTensor())
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=4)
    true_y, pred_y = [], []
    for batch in tqdm(iter(test_dl), total=len(test_dl)):
        x, y = batch
        true_y.extend(y)
        preds, probs = get_prediction(x, inference_model)
        pred_y.extend(preds.cpu())

    wandb.log({"accuracy": accuracy_score(true_y, pred_y)})
    print(classification_report(true_y, pred_y, digits=3))


if __name__ == "__main__":
    wandb.init(project="AdaptiveQuantization", entity="cbakos")
    model_class = MobileNetV3MNIST
    model = model_class()
    # train(model)
    inf_model = model_class.load_from_checkpoint(model.__class__.__name__ + ".pt", map_location="cpu")
    inference(inf_model)
