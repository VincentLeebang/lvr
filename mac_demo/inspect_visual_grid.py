
from transformers import AutoProcessor
from PIL import Image

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
IMAGE_PATH = "mac_demo/images/test.jpg"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

image = Image.open(IMAGE_PATH).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What is in this image?"}
        ]
    }
]

from qwen_vl_utils import process_vision_info

image_inputs, _ = process_vision_info(messages)

inputs = processor(
    text=["<|vision_start|><|image_pad|><|vision_end|>"],
    images=image_inputs,
    return_tensors="pt",
    do_resize=False
)

print("\nOriginal image size:", image.size)
print("Image grid THW:", inputs["image_grid_thw"])
print("Pixel values shape:", inputs["pixel_values"].shape)

grid = inputs["image_grid_thw"][0]

merge_size = processor.image_processor.merge_size

print("\nProcessed visual grid:")
print("Height before merge:", grid[1].item())
print("Width before merge:", grid[2].item())
print("Height after merge:", grid[1].item() // merge_size)
print("Width after merge:", grid[2].item() // merge_size)
print(
    "Expected visual tokens:",
    (grid[1].item() // merge_size)
    * (grid[2].item() // merge_size)
)
