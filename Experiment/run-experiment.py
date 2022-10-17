import os
import sys

required_args = 6 # Should be the number of required parameters + 1
args = []
arg_modelname = 1
arg_cpu = 2
arg_mem = 3
arg_batch = 4
arg_quant = 5

def print_experiment_params():
    print("-------------")
    print("Running experiment with:")
    print("Model name  :", args[arg_modelname])
    print("Cpu num     :", args[arg_cpu])
    print("Mem size    :", args[arg_mem])
    print("Batch size  :", args[arg_batch])
    print("Quant lv    :", args[arg_quant])
    print("Path        :", get_model_path())
    print("-------------")

def write_results(result):
    f = open("/resultsets/experiments/result-" + str(args[arg_modelname]) + "-" + str(args[arg_cpu]) + "-" + str(args[arg_mem]) + "-" + str(args[arg_batch]) + "-" + str(args[arg_quant]) + ".csv", "w")
    f.write("testline")
    f.close()

def get_model_path():
    if int(args[arg_quant]) == 18:
        return "/resultsets/models/" + str(args[arg_modelname]) + "-uq.pt"
    else:
        return "/resultsets/models/" + str(args[arg_modelname]) + "-q-" + str(args[arg_quant]) + ".pt"

if __name__ == "__main__":
    dir = "/resultsets/experiments/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    arg_num = len(sys.argv)
    if arg_num != required_args:
        print("Incorrect number of arguments, expected", required_args, "but got", arg_num)
        exit()
    args = sys.argv
    print_experiment_params()
    write_results("None yet")

    # Currently, the unquantified models should be called when the quantization level is 18. (0-17 respond to numbered models)
