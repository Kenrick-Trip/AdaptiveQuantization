import sys

required_args = 3 # Should be the number of required parameters + 1
args = []
arg_cpu = 1
arg_mem = 2

def print_experiment_params():
    print("-------------")
    print("Running experiment with:")
    print("Cpu num  :", args[arg_cpu])
    print("Mem size :", args[arg_mem])
    print("-------------")

def write_results(result):
    f = open("/resultsets/experiments/result-" + str(args[arg_cpu]) + "-" + str(args[arg_mem]) + ".csv", "w")
    f.write("testline")
    f.close()

if __name__ == "__main__":
    arg_num = len(sys.argv)
    if arg_num != required_args:
        print("Incorrect number of arguments, expected", required_args, "but got", arg_num)
        exit()
    args = sys.argv
    param_cpu = sys.argv[arg_cpu]
    param_mem = sys.argv[arg_mem]
    print_experiment_params()
    write_results("None yet")