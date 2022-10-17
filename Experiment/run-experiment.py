import sys
import torch
import csv
import os
import time
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm.autonotebook import tqdm
from sklearn.metrics import accuracy_score

# import all model classes
from model_classes.resnet18_mnist import ResNet18MNIST
from model_classes.googlenet_mnist import GoogLeNetMNIST
from model_classes.mobilenet_v3_mnist import MobileNetV3MNIST


class Experiment:
    def __init__(self, model, model_number, batch_size):
        self.model = model
        self.model_number = model_number
        self.batch_size = batch_size

    def get_prediction(self, x):
        self.inference_model.freeze()  # prepares model for predicting
        probabilities = torch.softmax(self.inference_model(x), dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        return predicted_class, probabilities

    def inference(self):
        test_ds = MNIST("mnist", train=False, download=True, transform=ToTensor())
        test_dl = DataLoader(test_ds, batch_size=self.batch_size, num_workers=4)
        true_y, pred_y = [], []
        for batch in tqdm(iter(test_dl), total=len(test_dl)):
            x, y = batch
            true_y.extend(y)
            preds, probs = self.get_prediction(x)
            pred_y.extend(preds.cpu())

        return accuracy_score(true_y, pred_y)

    def load_model(self):
        if self.model == "resnet18":
            self.model_size = os.path.getsize("/resultsets/models/resnet18-{}.pt".format(self.model_number))
            if self.model_number == 0:
                return ResNet18MNIST.load_from_checkpoint(
                    "/resultsets/models/resnet18-{}.pt".format(self.model_number), map_location="cpu")
            else:
                return ResNet18MNIST(quantize=True).load_from_checkpoint(
                    "/resultsets/models/resnet18-{}.pt".format(self.model_number), map_location="cpu")

        # todo: add other models here ...

    def run(self):
        start_time = time.time()
        self.inference_model = self.load_model()
        accuracy = self.inference()
        end_time = time.time()
        service_time = (end_time - start_time)/self.batch_size

        return accuracy, service_time, self.model_size


class Operators:
    def __init__(self):
        # argument parsing:
        self.args = sys.argv
        self.required_args = 6  # Should be the number of required parameters + 1
        self.arg_cpu = 1
        self.arg_mem = 2
        self.arg_batch = 3
        self.model_type = 4
        self.exp_name = 5

        self.experiment_name = str(self.args[self.exp_name])

        # log file settings:
        self.log_file = open("/resultsets/experiments/{}.csv".format(self.experiment_name), "w")
        self.writer = csv.writer(self.log_file)
        header = ["CPU", "Memory", "Batch size", "Model type", "Quantization method", "Data type",
                  "Quantization scheme", "Quantization location", "Accuracy (%)", "Service time (ms)", "Size (bytes)"]
        self.writer.writerow(header)

        self.quantization_settings = [
            ["None", "None", "None", "None"],
            ["static", "int8", "affine", "tensor"],
            ["static", "int8", "affine", "channel"],
            ["static", "int8", "symmetric", "tensor"],
            ["static",  "int8", "symmetric", "channel"],
            ["static", "uint8", "affine", "tensor"],
            ["static", "uint8", "affine", "channel"],
            ["static", "uint8", "symmetric", "tensor"],
            ["static", "uint8", "symmetric", "channel"],
            ["static", "int8", "affine", "tensor"],
            ["static", "int8", "affine", "channel"],
            ["static", "int8", "symmetric", "tensor"],
            ["static", "int8", "symmetric", "channel"],
            ["static", "uint8", "affine", "tensor"],
            ["static", "uint8", "affine", "channel"],
            ["static", "uint8", "symmetric", "tensor"],
            ["static", "uint8", "symmetric", "channel"],
            ["dynamic", "int8", "None", "None"],
            ["dynamic", "float16", "None", "None"]]

    def print_experiment_params(self, model_number):
        print("-------------")
        print("Running experiment with:")
        print("CPU num  : ", self.args[self.arg_cpu])
        print("Mem size : ", self.args[self.arg_mem])
        print("Batch size : ", self.args[self.arg_batch])
        print("Model type : ", str(self.args[self.model_type]))
        print("Model number : ", model_number)
        print("Quatization method : ", self.quantization_settings[model_number][0])
        print("Data type : ", self.quantization_settings[model_number][1])
        print("Quatization scheme : ", self.quantization_settings[model_number][2])
        print("Quatization location : ", self.quantization_settings[model_number][3])
        print("-------------")

    def write_results(self, accuracy, service_time, model_size):
        data = [str(self.args[self.arg_cpu]), str(self.args[self.arg_mem]), str(self.args[self.arg_batch]),
                str(self.args[self.model_type]), str(self.quantization_settings[model_number][0]),
                str(self.quantization_settings[model_number][1]), str(self.quantization_settings[model_number][2]),
                str(self.quantization_settings[model_number][3]), str(accuracy), str(service_time), str(model_size)]

        self.writer.writerow(data)

    def read_settings(self):
        arg_num = len(self.args)
        if arg_num != self.required_args:
            print("Incorrect number of arguments, expected", self.required_args, "but got", arg_num)
            exit()
        param_cpu = self.args[self.arg_cpu]
        param_mem = self.args[self.arg_mem]
        param_batch = self.args[self.arg_batch]
        model_type = str(self.args[self.model_type])
        experiment_name = str(self.args[self.exp_name])

        return param_cpu, param_mem, param_batch, model_type, experiment_name

    def exit(self):
        self.log_file.close()


if __name__ == "__main__":
    # argument format:
    # example: python3 run-experiment.py 2 4 32 resnet18 test-resnet
    # settings:
    num_models_per_type = 19  # example: 19 resnet18 models

    Operators = Operators()
    cpu, memory, batch_size, model_type, exp_name = Operators.read_settings()

    for e in range(num_models_per_type):
        model_number = e
        Operators.print_experiment_params(model_number)

        exp = Experiment(model_type, model_number, batch_size)
        accuracy, service_time, model_size = exp.run()
        Operators.write_results(accuracy, service_time, model_size)

    Operators.exit()
