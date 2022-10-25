import os
import pickle

import torch
import torch.nn as nn
import copy

from Train.utils import save_torchscript_model, calibrate_model, load_model, set_random_seeds, prepare_dataloader, \
    create_model, evaluate_model


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

    def __init__(self, model_name, qscheme, dir):
        self.model_name = model_name
        self.qscheme = qscheme
        self.dir = dir

        # settings:
        self.random_seed = 0
        self.num_classes = 10
        self.cpu_device = torch.device("cpu")
        self.train_loader, self.test_loader = prepare_dataloader(num_workers=4,
                                                                 train_batch_size=128,
                                                                 eval_batch_size=128)

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

        # DO NOT change DTYPE
        if self.qscheme == "affine":
            quantization_config = torch.quantization.QConfig(
                activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8),
                weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine)
            )
        elif self.qscheme == "symmetric":
            quantization_config = torch.quantization.QConfig(
                activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8),
                weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8,
                                                                   qscheme=torch.per_tensor_symmetric)
            )
        elif self.qscheme == "histogram":
            quantization_config = torch.quantization.QConfig(
                activation=torch.quantization.HistogramObserver.with_args(reduce_range=True),
                weight=torch.quantization.HistogramObserver.with_args(reduce_range=True, dtype=torch.qint8)
            )
        else:
            raise "wrong inputs for qscheme"

        quantized_model.qconfig = quantization_config

        # Print quantization configurations
        print(quantized_model.qconfig)

        torch.quantization.prepare(quantized_model, inplace=True)
        # Use training data for calibration.

        calibrate_model(model=quantized_model, loader=train_loader, device=torch.device("cpu:0"))
        quantized_model = torch.quantization.convert(quantized_model, inplace=False)
        quantized_model.eval()

        return quantized_model

    def load_model(self):
        path = "Train/saved_models/{}_mnist.pt".format(self.model_name)
        model = create_model(num_classes=self.num_classes, model_name=self.model_name)
        self.uqmodel = load_model(model=model, model_filepath=path, device=self.cpu_device)
        return self.uqmodel

    def run_quantization(self):
        self.load_model()
        set_random_seeds(random_seed=self.random_seed)
        self.qmodel = self.create_static_quantized_model(trained_original_model=self.uqmodel,
                                                         train_loader=self.train_loader)
        self.qmodel.eval()
        return self.qmodel

    def save_quatized_model(self, filename):
        model = self.run_quantization()

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        # Save quantized model.
        save_torchscript_model(model=model, model_dir=self.dir, model_filename=filename)
        return model

    def save_unquantized_model(self, filename):
        self.load_model()

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        # Save quantized model.
        save_torchscript_model(model=self.uqmodel, model_dir=self.dir, model_filename=filename)

    def get_accuracy(self, model):
        accuracy = evaluate_model(model=model, test_loader=self.test_loader, device=self.cpu_device)
        return accuracy.item()


if __name__ == "__main__":
    # Save accuracies for unquantized and quantized versions
    model_names = ["resnet18", "resnet34"]
    accuracies = {k: {} for k in model_names}
    model_sizes = {k: {} for k in model_names}
    # possible quant schemes
    q_configs = ["histogram", "affine", "symmetric"]
    dir = "/resultsets/models/"

    for m_name in model_names:
        # save unquantized model as jit model:
        uquant = Quantize(m_name, None, dir)
        path = "{}{}_jit_None_mnist.pt".format(dir, m_name)
        uquant.save_unquantized_model(path)
        accuracies[m_name]["None"] = uquant.get_accuracy(uquant.uqmodel)
        model_sizes[m_name]["None"] = os.path.getsize(path) / (1024 ** 2)  # in MB

        for scheme in q_configs:
            path = "{}{}_jit_{}_mnist.pt".format(dir, m_name, scheme)
            quant = Quantize(m_name, scheme, dir)
            model = quant.save_quatized_model(path)
            accuracies[m_name][scheme] = quant.get_accuracy(model)
            model_sizes[m_name]["None"] = os.path.getsize(path) / (1024 ** 2)  # in MB

    with open(dir+'accuracies.pickle', 'wb') as handle:
        pickle.dump(accuracies, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(dir + 'model_sizes.pickle', 'wb') as handle:
        pickle.dump(model_sizes, handle, protocol=pickle.HIGHEST_PROTOCOL)
