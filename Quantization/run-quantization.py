import os
import torch
import torch.nn as nn
import copy

from Train.utils import save_torchscript_model, calibrate_model, load_model, set_random_seeds, prepare_dataloader, \
    create_model


class QuantizedResNet(nn.Module):
    """
    Helper class for static quantization. This inserts observers in
    the model that will observe activation tensors during calibration.
    """

    def __init__(self, model_fp32):
        super(QuantizedResNet, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x


class Quantize:

    def __init__(self, model, dtype, qscheme):
        self.model = model
        self.dtype = dtype  # qint8 or qfloat16
        self.qscheme = qscheme
        self.dir = "/resultsets/models/"

        # settings:
        self.random_seed = 0
        self.num_classes = 10
        self.cpu_device = torch.device("cpu")

    def create_fused_model(self, original_model):
        fused_model = copy.deepcopy(original_model)

        # The model has to be switched to evaluation mode before any layer fusion.
        # Otherwise the quantization will not work correctly.
        fused_model.eval()

        # Fuse the model in place rather manually.
        fused_model = torch.quantization.fuse_modules(fused_model, [["conv1", "bn1", "relu"]], inplace=True)
        for module_name, module in fused_model.named_children():
            if "layer" in module_name:
                for basic_block_name, basic_block in module.named_children():
                    torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]],
                                                    inplace=True)
                    for sub_block_name, sub_block in basic_block.named_children():
                        if sub_block_name == "downsample":
                            torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)
        return fused_model

    def create_static_quantized_model(self, trained_original_model, train_loader):
        fused_model = self.create_fused_model(original_model=trained_original_model)
        # Prepare the model for static quantization.
        quantized_model = QuantizedResNet(model_fp32=fused_model)
        # Using un-fused model will fail.
        # Because there is no quantized layer implementation for a single batch normalization layer.

        if self.qscheme == "affine":
            quantization_config = torch.quantization.QConfig(
                activation=torch.quantization.MinMaxObserver.with_args(dtype=self.dtype),
                weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine)
            )
        elif self.qscheme == "symmetric":
            quantization_config = torch.quantization.QConfig(
                activation=torch.quantization.MinMaxObserver.with_args(dtype=self.dtype),
                weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
            )
            # torch.quantization.qconfig.float16_static_qconfig
        else:
            error = "wrong inputs for qscheme"
            assert print(error)

        quantized_model.qconfig = quantization_config

        # Print quantization configurations
        print(quantized_model.qconfig)

        torch.quantization.prepare(quantized_model, inplace=True)
        # Use training data for calibration.
        calibrate_model(model=quantized_model, loader=train_loader, device=torch.device("cpu:0"))
        quantized_model = torch.quantization.convert(quantized_model, inplace=True)
        quantized_model.eval()

        return quantized_model

    def load_model(self):
        # todo: add other models
        if self.model == "resnet18":
            # Create an untrained model.
            model = create_model(num_classes=self.num_classes)
            # Load a pretrained model.
            self.uqmodel = load_model(model=model, model_filepath="uqmodels/resnet18/resnet18_mnist.pt",
                                      device=self.cpu_device)
        elif self.model == "resnet34":
            # Create an untrained model.
            model = create_model(num_classes=self.num_classes)
            # Load a pretrained model.
            self.uqmodel = load_model(model=model, model_filepath="uqmodels/resnet34/resnet34_mnist.pt",
                                      device=self.cpu_device)


    def run_quantization(self):
        self.load_model()
        set_random_seeds(random_seed=self.random_seed)
        train_loader, test_loader = prepare_dataloader(num_workers=4, train_batch_size=128, eval_batch_size=32)
        self.qmodel = self.create_static_quantized_model(trained_original_model=self.uqmodel, train_loader=train_loader)
        self.qmodel.eval()

        return self.qmodel

    def save_quatized_model(self, filename):
        model = self.run_quantization()

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        # Save quantized model.
        save_torchscript_model(model=model, model_dir=self.dir, model_filename=filename)

    def save_unquatized_model(self, filename):
        self.load_model()

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        # Save quantized model.
        save_torchscript_model(model=self.uqmodel, model_dir=self.dir, model_filename=filename)


if __name__ == "__main__":
    # save unquantized model as jit model:
    Quantize("resnet18", None, None).save_unquatized_model("resnet18_jit_mnist.pt")
    # todo: also put resnet34 here

    # possible models for resnet:
    models = [
        # resnet18 models:
        ["resnet18", torch.qint8,  "affine"],
        ["resnet18", torch.qint8, "symmetric"],
        ["resnet18", torch.quint8, "affine"],
        ["resnet18", torch.quint8, "symmetric"],
        # resnet34 models:
        # ["resnet34", torch.qint8, "affine"],
        # ["resnet34", torch.qint8, "symmetric"],
        # ["resnet34", torch.quint8, "affine"],
        # ["resnet34", torch.quint8, "symmetric"]
    ]

    for m in range(len(models)):
        Quantize(models[m][0], models[m][1], models[m][2]).save_quatized_model(
            "{}_jit_q_{}_mnist.pt".format(models[m][0], m)
        )
