FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /app
COPY /Experiment Experiment
COPY /Train Experiment/Train
WORKDIR Experiment
RUN pip3 install -r requirements.txt
#COPY . .

ENTRYPOINT [ "python3", "run-experiment.py"]