# Adapted from: https://leimao.github.io/blog/PyTorch-Static-Quantization/
import os
import torch
import torch.nn as nn
import copy

from Train.train_resnet import train
from Train.utils import measure_inference_latency, load_torchscript_model, evaluate_model, save_torchscript_model, \
    calibrate_model, load_model, set_random_seeds, prepare_dataloader, create_model


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


def create_fused_model(original_model):
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


def create_static_quantized_model(trained_original_model, train_loader, dtype=torch.qint8, qscheme=torch.per_tensor_affine):
    fused_model = create_fused_model(original_model=trained_original_model)
    # Prepare the model for static quantization.
    quantized_model = QuantizedResNet(model_fp32=fused_model)
    # Using un-fused model will fail.
    # Because there is no quantized layer implementation for a single batch normalization layer.

    quantization_config = torch.quantization.QConfig(
        activation=torch.quantization.MinMaxObserver.with_args(dtype=dtype),
        weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=qscheme)
    )

    quantized_model.qconfig = quantization_config

    # Print quantization configurations
    print(quantized_model.qconfig)

    torch.quantization.prepare(quantized_model, inplace=True)

    # Use training data for calibration.
    calibrate_model(model=quantized_model, loader=train_loader, device=torch.device("cpu:0"))

    quantized_model = torch.quantization.convert(quantized_model, inplace=True)

    quantized_model.eval()

    return quantized_model


def inference():
    random_seed = 0
    num_classes = 10
    cpu_device = torch.device("cpu")

    model_dir = "saved_models"
    model_filename = "resnet18_mnist.pt"
    jit_quantized_model_filename = "resnet18_jit_quantized_mnist.pt"
    jit_model_filename = "resnet18_jit_mnist.pt"
    model_filepath = os.path.join(model_dir, model_filename)
    jit_quantized_model_filepath = os.path.join(model_dir, jit_quantized_model_filename)
    jit_model_filepath = os.path.join(model_dir, jit_model_filename)

    set_random_seeds(random_seed=random_seed)
    train_loader, test_loader = prepare_dataloader(num_workers=4, train_batch_size=128, eval_batch_size=32)

    # Create an untrained model.
    model = create_model(num_classes=num_classes)

    # Load a pretrained model.
    model = load_model(model=model, model_filepath=model_filepath, device=cpu_device)

    # model.to(cpu_device)
    quantized_model = create_static_quantized_model(trained_original_model=model, train_loader=train_loader)

    model.eval()

    # Use TorchScript for faster inference both for original and quantized models
    save_torchscript_model(model=model, model_dir=model_dir, model_filename=jit_model_filename)

    # Save quantized model.
    save_torchscript_model(model=quantized_model, model_dir=model_dir, model_filename=jit_quantized_model_filename)

    # Load quantized model.
    quantized_jit_model = load_torchscript_model(model_filepath=jit_quantized_model_filepath,
                                                 device=torch.device("cpu:0"))
    jit_model = load_torchscript_model(model_filepath=jit_model_filepath, device=torch.device("cpu:0"))

    input_size = (1, 1, 32, 32)

    _, fp32_eval_accuracy = evaluate_model(model=jit_model, test_loader=test_loader, device=cpu_device, criterion=None)
    _, int8_eval_accuracy = evaluate_model(model=quantized_jit_model, test_loader=test_loader, device=cpu_device,
                                           criterion=None)

    print("FP32 evaluation accuracy: {:.3f}".format(fp32_eval_accuracy))
    print("INT8 evaluation accuracy: {:.3f}".format(int8_eval_accuracy))

    fp32_jit_cpu_inference_latency = measure_inference_latency(model=quantized_jit_model,
                                                               device=cpu_device,
                                                               input_size=input_size,
                                                               num_samples=100)
    fp32_cpu_inference_latency = measure_inference_latency(model=model,
                                                           device=cpu_device,
                                                           input_size=input_size,
                                                           num_samples=100)
    int8_cpu_inference_latency = measure_inference_latency(model=quantized_model,
                                                           device=cpu_device,
                                                           input_size=input_size,
                                                           num_samples=100)
    int8_jit_cpu_inference_latency = measure_inference_latency(model=quantized_jit_model,
                                                               device=cpu_device,
                                                               input_size=input_size,
                                                               num_samples=100)

    print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(fp32_cpu_inference_latency * 1000))
    print("INT8 CPU Inference Latency: {:.2f} ms / sample".format(int8_cpu_inference_latency * 1000))
    print("FP32 JIT CPU Inference Latency: {:.2f} ms / sample".format(fp32_jit_cpu_inference_latency * 1000))
    print("INT8 JIT CPU Inference Latency: {:.2f} ms / sample".format(int8_jit_cpu_inference_latency * 1000))


if __name__ == "__main__":
    # train(learning_rate=1e-2, num_epochs=10)
    inference()
