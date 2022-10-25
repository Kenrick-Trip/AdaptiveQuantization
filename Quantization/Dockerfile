FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /app
#COPY /Quantization/requirements.txt requirements.txt
COPY /Quantization Quantization
COPY /Train Quantization/Train
WORKDIR Quantization
RUN pip3 install -r requirements.txt
#CMD ["ls", "-a"]
ENTRYPOINT [ "python3", "run-quantization.py"]