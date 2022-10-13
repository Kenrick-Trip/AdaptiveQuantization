# instructions for Quantization:

In order to create quantized models, we need run run-quantization.py file. This file includes the Quantize class

The **Quantize** class takes 5 input arguments:

1. **Quantization type:** either "static" or "dynamic"
2. **Model name:** either "resnet18", "mobilenet" and "googlenet"

    The following options depend on the Quantization type.

3. **Data type:** 
    - if [Quantization type = "static"] - either "torch.qint8" or "torch.quint8"
    - if [Quantization type = "dynamic"] - either "torch.qint8" or "torch.float16"
4. **Quatization scheme:**
    - if [Quantization type = "static"] - either "affine" or "symmetric"
    - if [Quantization type = "dynamic"] - None, has no effect
5. **Locaction of quatization**
    - if [Quantization type = "static"] - either "tensor" or "channel"
    - if [Quantization type = "dynamic"] - None, has no effect

# todo: naming the model and saving