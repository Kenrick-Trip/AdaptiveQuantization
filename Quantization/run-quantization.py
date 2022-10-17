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


if __name__ == "__main__":

    nqmodel = ResNet18MNIST.load_from_checkpoint("uqmodels/resnet18/resnet18_mnist.pt", map_location="cpu")

    # possible models for resnet:
    m1 = ["static", "resnet18", torch.qint8,  "affine", "tensor"]
    m2 = ["static", "resnet18", torch.qint8, "affine", "channel"]
    m3 = ["static", "resnet18", torch.qint8, "symmetric", "tensor"]
    m4 = ["static", "resnet18", torch.qint8, "symmetric", "channel"]
    m5 = ["static", "resnet18", torch.quint8, "affine", "tensor"]
    m6 = ["static", "resnet18", torch.quint8, "affine", "channel"]
    m7 = ["static", "resnet18", torch.quint8, "symmetric", "tensor"]
    m8 = ["static", "resnet18", torch.quint8, "symmetric", "channel"]
    m9 = ["static", "resnet18", torch.qint8, "affine", "tensor"]
    m10 = ["static", "resnet18", torch.qint8, "affine", "channel"]
    m11 = ["static", "resnet18", torch.qint8, "symmetric", "tensor"]
    m12 = ["static", "resnet18", torch.qint8, "symmetric", "channel"]
    m13 = ["static", "resnet18", torch.quint8, "affine", "tensor"]
    m14 = ["static", "resnet18", torch.quint8, "affine", "channel"]
    m15 = ["static", "resnet18", torch.quint8, "symmetric", "tensor"]
    m16 = ["static", "resnet18", torch.quint8, "symmetric", "channel"]
    m17 = ["dynamic", "resnet18", torch.qint8, None, None]
    m18 = ["dynamic", "resnet18", torch.float16, None, None]

    ex = m7

    qmodel = Quantize(ex[0], ex[1], ex[2], ex[3], ex[4]).run_quantization()
    print(qmodel)

    # todo: naming the model and saving