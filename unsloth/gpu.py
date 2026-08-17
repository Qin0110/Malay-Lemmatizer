import torch

# 1. Verify PyTorch sees your GPU
print("CUDA Available:", torch.cuda.is_available())

# 2. Get the name of your active GPU
if torch.cuda.is_available():
    print("GPU Device Name:", torch.cuda.get_device_name(0))
    print("CUDA Device Capability:", torch.cuda.get_device_capability(0))
    
# 3. Check what CUDA version PyTorch was compiled with
print("PyTorch CUDA Version:", torch.version.cuda)