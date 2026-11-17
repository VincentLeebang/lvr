
import sys
import platform
import subprocess
import torch
import transformers

print("=" * 45)
print("LATENT-SIGHT ENVIRONMENT CHECK")
print("=" * 45)

print("Python:", sys.version)
print("Processor:", platform.machine())
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)

# Check Mac unified memory
memory = subprocess.check_output(
    ["sysctl", "-n", "hw.memsize"]
).decode().strip()

memory_gb = int(memory) / (1024 ** 3)

print("Unified Memory:", round(memory_gb, 2), "GB")

# Check Apple GPU
print("CUDA Available:", torch.cuda.is_available())
print("MPS Available:", torch.backends.mps.is_available())

if torch.backends.mps.is_available():

    x = torch.randn(500, 500, device="mps")
    y = x @ x

    torch.mps.synchronize()

    print("Apple GPU Test: SUCCESS")
    print("Device:", y.device)

else:
    print("Apple GPU is not available.")

print("=" * 45)
print("CHECK COMPLETED")
