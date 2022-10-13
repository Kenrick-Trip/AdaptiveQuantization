import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from tqdm.autonotebook import tqdm
from sklearn.metrics import classification_report

from models.resnet18_mnist import ResNet18MNIST


def main():
    train_ds = MNIST("mnist", train=True, download=True, transform=ToTensor())
    test_ds = MNIST("mnist", train=False, download=True, transform=ToTensor())

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=64)

    model = ResNet18MNIST()

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu",
        devices=[0]
    )

    trainer.fit(model, train_dl)
    trainer.save_checkpoint("resnet18_mnist.pt")

    inference_model = ResNet18MNIST.load_from_checkpoint("resnet18_mnist.pt", map_location="cpu")
    true_y, pred_y = [], []
    for batch in tqdm(iter(test_dl), total=len(test_dl)):
        x, y = batch
        true_y.extend(y)
        preds, probs = get_prediction(x, inference_model)
        pred_y.extend(preds.cpu())
    print(classification_report(true_y, pred_y, digits=3))


def get_prediction(x, model: pl.LightningModule):
    model.freeze()  # prepares model for predicting
    probabilities = torch.softmax(model(x), dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class, probabilities


if __name__ == "__main__":
    main()
