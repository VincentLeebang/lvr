
import torch
import transformers

print("Checking model support...")

try:
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        AutoProcessor
    )

    print("Qwen2.5-VL architecture: Available")

except Exception as e:
    print("Architecture import failed:", e)

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("MPS:", torch.backends.mps.is_available())
