# AdaptiveQuantization
Repo for group 10 of the course CS4215 Quantitative Performance Evaluation for Computing Systems

## Requirements
- Docker

## Repository overview
The repository consists of three main code directories, as well as some additional data directories.

### Quantization
This directory contains the container used to perform the quantization on the trained models and stores them on a volume

### Experiment
This directory contains the container used to perform the inference tasks on models stored on the volume. The results are also stored on a volume from this container.

### Train
This directory contains the utilities used by the Quantization and Experiment containers.

## Executing the experiments
The experiments can be performed as follows:
1. Run the `init.sh` file. This will build the Quantization and Experiment container, it also creates a volume on docker to store the models and results. Finally, it runs the Quantization container once and stores the results on the volume.
2. Run the `run-batch.sh` file. This will start the Experiment container with different parameters iteratively. This way the results are executed in isolation under as consistent situations and environments as possible.
3. Once the `run-batch.sh` has finished, the results can be extracted from the docker volume.
4. (Optional) If you want to perform a repetition, be sure to remove the previous results from the volume. Otherwise, all new results will be appended to the previous results.