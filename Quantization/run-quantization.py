import os
import sys

import torch
from uqmodels.resnet18.resnet18_mnist import ResNet18MNIST
from uqmodels.observers import MinMaxObserver, PerChannelMinMaxObserver


class Quantize:

    def __init__(self, type, model, dtype, qscheme, qloc):
        self.type = type # dynamic or static
        self.model = model
        self.dtype = dtype  # qint8 or qfloat16
        self.qscheme = qscheme
        self.qloc = qloc

    def load_model(self):
        # todo: add other models
        if self.model == "resnet18":
            self.uqmodel = ResNet18MNIST.load_from_checkpoint("uqmodels/resnet18/resnet18_mnist.pt", map_location="cpu")
            self.qmodel = ResNet18MNIST(quantize=True).load_from_checkpoint("uqmodels/resnet18/resnet18_mnist.pt", map_location="cpu")

    def config_static_quantization(self):
        if self.qscheme == "affine" and self.qloc == "tensor":
            return torch.quantization.QConfig(
                activation=MinMaxObserver.with_args(dtype=self.dtype, qscheme=torch.per_tensor_affine),
                weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine))
        elif self.qscheme == "symmetric" and self.qloc == "tensor":
            return torch.quantization.QConfig(
                activation=MinMaxObserver.with_args(dtype=self.dtype, qscheme=torch.per_tensor_symmetric),
                weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))
        elif self.qscheme == "affine" and self.qloc == "channel":
            return torch.quantization.QConfig(
                activation=PerChannelMinMaxObserver.with_args(dtype=self.dtype, qscheme=torch.per_channel_affine),
                weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_affine))
        elif self.qscheme == "symmetric" and self.qloc == "channel":
            return torch.quantization.QConfig(
                activation=PerChannelMinMaxObserver.with_args(dtype=self.dtype, qscheme=torch.per_channel_symmetric),
                weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
        else:
            error = "wrong inputs for qscheme and qloc"
            assert print(error)

    def run_quantization(self):
        self.load_model()
        if self.type == "dynamic":
            return torch.quantization.quantize_dynamic(self.uqmodel, dtype=self.dtype)
        elif self.type == "static":
            state_dict = self.uqmodel.state_dict()
            self.uqmodel = self.uqmodel.to('cpu')
            self.qmodel.load_state_dict(state_dict)
            self.qmodel.qconfig = self.config_static_quantization()

            torch.quantization.prepare(self.qmodel, inplace=True)
            torch.quantization.convert(self.qmodel, inplace=True)
            return self.qmodel

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
    torch.save(resnet_model.state_dict(), os.path.join(dir, "resnet18-0.pt"))


    # possible models for resnet:
    models = [
        # resnet models:
        ["static", "resnet18", torch.qint8,  "affine", "tensor"],
        ["static", "resnet18", torch.qint8, "affine", "channel"],
        ["static", "resnet18", torch.qint8, "symmetric", "tensor"],
        ["static", "resnet18", torch.qint8, "symmetric", "channel"],
        ["static", "resnet18", torch.quint8, "affine", "tensor"],
        ["static", "resnet18", torch.quint8, "affine", "channel"],
        ["static", "resnet18", torch.quint8, "symmetric", "tensor"],
        ["static", "resnet18", torch.quint8, "symmetric", "channel"],
        ["static", "resnet18", torch.qint8, "affine", "tensor"],
        ["static", "resnet18", torch.qint8, "affine", "channel"],
        ["static", "resnet18", torch.qint8, "symmetric", "tensor"],
        ["static", "resnet18", torch.qint8, "symmetric", "channel"],
        ["static", "resnet18", torch.quint8, "affine", "tensor"],
        ["static", "resnet18", torch.quint8, "affine", "channel"],
        ["static", "resnet18", torch.quint8, "symmetric", "tensor"],
        ["static", "resnet18", torch.quint8, "symmetric", "channel"],
        ["dynamic", "resnet18", torch.qint8, None, None],
        ["dynamic", "resnet18", torch.float16, None, None]
    ]

    for m in range(len(models)):
        Quantize(models[m][0], models[m][1], models[m][2], models[m][3],
                 models[m][4]).save_model("{}-{}.pt".format(models[m][1], m+1))

    # todo: Jasper, what do we do with the lines below?
    # f = open("/resultsets/models/example.csv", "w")
    # f.write("testline")
    # f.close()

    # todo: naming the model and saving