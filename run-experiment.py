import torch
import torchvision
from torch.ao.quantization import MinMaxObserver, default_observer
from torch.ao.quantization.qconfig import QConfig

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.quantization.QuantStub()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.relu = torch.nn.ReLU()
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.conv(x)
        x = self.relu(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x


class Quantize:

    def __init__(self, type, model, dtype):
        self.type = type # dynamic or static
        self.model = model
        self.dtype = dtype  # qint8 or qfloat16
        self.datasize = (3, 1, 32, 32)

    def run_quatiation(self):
        if (self.type == "dynamic"):
            qmodel = torch.quantization.quantize_dynamic(self.model, dtype=self.dtype)
            return qmodel
        elif (self.type == "static"):
            model_fp32 = M()
            model_fp32.eval()
            my_qconfig = QConfig(
                activation=MinMaxObserver.with_args(dtype=torch.qint8),
                weight=default_observer.with_args(dtype=torch.qint8))
            model_fp32.qconfig = my_qconfig
            model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['conv', 'relu']])
            model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)
            input_fp32 = torch.randn(self.datasize)
            model_fp32_prepared(input_fp32)
            return torch.quantization.convert(model_fp32_prepared)

def write_results(filename):
    f = open("/resultsets/" + filename + ".csv", "x")
    f.write('testline!')
    f.close()

if __name__ == "__main__":
    no_q_model = torchvision.models.resnet18(pretrained=True, progress=True)

    q_type = ["static", "dynamic"]
    data_types = [torch.qint8, torch.quint8, torch.qint32, torch.float16]
    # Quantize:
    model = Quantize(q_type[0], no_q_model, data_types[0]).run_quatiation()

    print(model)