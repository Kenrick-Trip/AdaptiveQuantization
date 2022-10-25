import sys
import torch
import csv
import os

from Train.utils import measure_inference_latency, load_torchscript_model, evaluate_model, prepare_dataloader

class Experiment:
    def __init__(self, model_name, model_number, batch_size):
        self.model = model_name
        self.model_number = model_number
        self.batch_size = batch_size

        self.dir = "/resultsets/models"
        _, self.test_loader = prepare_dataloader(num_workers=4, train_batch_size=128, eval_batch_size=self.batch_size)
        self.input_size = (1, 1, 32, 32)
        self.cpu_device = torch.device("cpu")

    def find_model_size(self, model):
        # https://discuss.pytorch.org/t/finding-model-size/130275
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        return (param_size + buffer_size) / 1024 ** 2   # in MB

    def inference(self):
        model = self.load_model()
        _, accuracy = evaluate_model(model=model, test_loader=self.test_loader, device=self.cpu_device,
                                     criterion=None)
        service_time = measure_inference_latency(model=model, device=self.cpu_device,
                                                 input_size=self.input_size, num_samples=10)
        model_size = self.find_model_size(model)

        return accuracy, service_time, model_size

    def load_model(self):
        if self.model_number == 4:
            return load_torchscript_model(
                model_filepath="{}/{}_jit_mnist.pt".format(self.dir, self.model),
                device=torch.device("cpu:0")
            )
        else:
            return load_torchscript_model(
                model_filepath="{}/{}_jit_q_{}_mnist.pt".format(self.dir, self.model, self.model_number),
                device=torch.device("cpu:0")
            )


class Operators:
    def __init__(self):
        dir = "/resultsets/experiments/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        # argument parsing:
        self.args = sys.argv
        self.required_args = 6  # Should be the number of required parameters + 1
        self.arg_modelname = 1 # resnet18 or resnet34
        self.arg_cpu = 2
        self.arg_mem = 3
        self.arg_batch = 4 # positive integer
        self.arg_quant_setting = 5 # positive integer between 0 and 4 (0 is unquantized model)

        # log file settings:
        self.log_file = open("/resultsets/experiments/{}.csv".format(self.args[self.arg_modelname]), "a")

        self.writer = csv.writer(self.log_file)
        header = ["CPU", "Memory", "Batch size", "Model name", "Data type",
                  "Quantization scheme", "Accuracy (%)", "Service time (ms)", "Size (bytes)"]
        # self.writer.writerow(header)

        self.quantization_settings = [
            ["int8", "affine"],
            ["int8", "symmetric"],
            ["uint8", "affine"],
            ["uint8", "symmetric"],
            ["None", "None"]
        ]

    def print_experiment_params(self):
        print("-------------")
        print("Running experiment with:")
        print("CPU num  : ", self.args[self.arg_cpu])
        print("Mem size : ", self.args[self.arg_mem])
        print("Batch size : ", int(self.args[self.arg_batch]))
        print("Model name : ", str(self.args[self.arg_modelname]))
        print("Quantization data type : ", str(self.quantization_settings[int(self.args[self.arg_quant_setting])][0]))
        print("Quantization scheme : ", str(self.quantization_settings[int(self.args[self.arg_quant_setting])][1]))
        print("-------------")

    def write_results(self, accuracy, service_time, model_size):
        data = [
            str(self.args[self.arg_cpu]),
            str(self.args[self.arg_mem]),
            str(self.args[self.arg_batch]),
            str(self.args[self.arg_modelname]),
            str(self.quantization_settings[int(self.args[self.arg_quant_setting])][0]),
            str(self.quantization_settings[int(self.args[self.arg_quant_setting])][1]),
            str(accuracy.item()),
            str(service_time),
            str(model_size)
        ]
        print(*data, sep = ", ")
        self.writer.writerow(data)

    def quatization_setting_number(self):
        return int(self.args[self.arg_quant_setting])

    def exit(self):
        self.log_file.close()


if __name__ == "__main__":
    dir = "/resultsets/experiments/"
    if not os.path.exists(dir):
        os.makedirs(dir)

    Operators = Operators()
    Operators.print_experiment_params()

    exp = Experiment(
        str(Operators.args[Operators.arg_modelname]),
        int(Operators.args[Operators.arg_quant_setting]),
        int(Operators.args[Operators.arg_batch])
    )

    accuracy, service_time, model_size = exp.inference()
    Operators.write_results(accuracy, service_time, model_size)

    Operators.exit()
