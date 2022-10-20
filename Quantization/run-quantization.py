import os
from uqmodels.resnet18.resnet18_mnist import ResNet18MNIST
from uqmodels.mobilenet_v3_mnist.mobilenet_v3_mnist import MobileNetV3MNIST

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm.autonotebook import tqdm
from sklearn.metrics import accuracy_score
import torch
from torch import nn


class RunInference:
    def __init__(self, batch_size, inference_model):
        self.batch_size = batch_size
        self.inference_model = inference_model

    def get_prediction(self, x):
        self.inference_model.freeze()  # prepares model for predicting
        probabilities = torch.softmax(self.inference_model(x), dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        return predicted_class, probabilities

    def inference(self):
        test_ds = MNIST("mnist", train=False, download=False, transform=ToTensor())
        test_dl = DataLoader(test_ds, batch_size=self.batch_size, num_workers=4)
        true_y, pred_y = [], []
        for batch in tqdm(iter(test_dl), total=len(test_dl)):
            x, y = batch
            true_y.extend(y)
            preds, probs = self.get_prediction(x)
            pred_y.extend(preds.cpu())

        return accuracy_score(true_y, pred_y)

class Quantize:

    def __init__(self, model_name, dtype):
        self.model_name = model_name
        self.dtype = dtype  # qint8 or qfloat16
        self.layers = set()

    def load_model(self):
        # todo: add other models
        if self.model_name == "resnet18":
            self.uqmodel = ResNet18MNIST.load_from_checkpoint(
                "uqmodels/resnet18/resnet18_mnist.pt", map_location="cpu")

            for name, layer in self.uqmodel.named_modules():
                print(name)
                # self.layers.add(layer)

            self.layers = {nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.Linear}

        elif self.model_name == "mobilenetv3":
            self.uqmodel = MobileNetV3MNIST.load_from_checkpoint(
                "uqmodels/mobilenet_v3_mnist/MobileNetV3MNIST.pt", map_location="cpu")
        elif self.model_name == "googlenet":
            self.uqmodel = MobileNetV3MNIST.load_from_checkpoint(
                "uqmodels/googlenet_mnist/googlenet_mnist.pt", map_location="cpu")
        else:
            print("ERROR: model name: {} , not known".format(self.model_name))


    def run_quantization(self):
        self.load_model()
        return torch.quantization.quantize_dynamic(self.uqmodel, {}, dtype=self.dtype)

    def save_model(self, filename):
        model = self.run_quantization()
        dir = "/resultsets/models/"

        if not os.path.exists(dir):
            os.makedirs(dir)

        torch.save(model.state_dict(), os.path.join(dir, filename))


if __name__ == "__main__":
    dir = "/resultsets/models/"
    if not os.path.exists(dir):
        os.makedirs(dir)

    resnet_model = ResNet18MNIST.load_from_checkpoint("uqmodels/resnet18/resnet18_mnist.pt", map_location="cpu")
    torch.save(resnet_model.state_dict(), os.path.join(dir, "resnet18-uq.pt"))


    # possible models for resnet:
    models = [
        # resnet models:
        ["resnet18", torch.qint8],
        ["resnet18", torch.float16],
        ["mobilenetv3", torch.qint8],
        ["mobilenetv3", torch.float16],
        # ["googlenet", torch.qint8],
        # ["googlenet", torch.float16],
    ]

    for m in range(len(models)):
        qmodel = Quantize(models[m][0], models[m][1]).run_quantization()
        print(qmodel)
        acc = RunInference(32, qmodel).inference()
        # acc_uq = RunInference(32, resnet_model).inference()
        print(acc)
