
from PIL import Image
from transformers import AutoProcessor
from src.dataset.lvr_sft_dataset import SupervisedDatasetLVR

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
IMAGE_PATH = "mac_demo/images/test.jpg"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Create a lightweight dataset object for testing its original method
dataset = SupervisedDatasetLVR.__new__(SupervisedDatasetLVR)
dataset.processor = processor

image = Image.open(IMAGE_PATH).convert("RGB")

bbox = [
    0.3383705650459921,
    0.11100196463654224,
    0.6787122207621551,
    0.9744597249508841
]

# Call the actual repository method
result = dataset.bbox_to_token_idxs_manual(
    images=[image],
    bboxes=[bbox]
)

indices = result[0]

print("\nOriginal dataset method results")
print("Image dimensions:", image.size)
print("Selected token count:", len(indices))
print("Minimum index:", indices.min())
print("Maximum index:", indices.max())
print("First 10 indices:", indices[:10])

assert len(indices) == 627
assert indices.min() >= 0
assert indices.max() < 1944

print("\nOriginal method validation passed!")
