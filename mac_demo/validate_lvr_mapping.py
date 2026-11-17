
import json
import numpy as np
from PIL import Image
from transformers import AutoProcessor

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
JSON_PATH = "mac_demo/data/lvr_train.json"

processor = AutoProcessor.from_pretrained(MODEL_ID)

with open(JSON_PATH, "r") as f:
    dataset = json.load(f)

for example in dataset:
    image_path = example["image"][0]
    bbox = example["bboxes"][0]

    image = Image.open(image_path).convert("RGB")

    # Get the actual Qwen visual grid
    from qwen_vl_utils import process_vision_info

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Inspect the image"}
        ]
    }]

    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=["<|vision_start|><|image_pad|><|vision_end|>"],
        images=image_inputs,
        return_tensors="pt",
        do_resize=False
    )

    grid = inputs["image_grid_thw"][0]
    merge = processor.image_processor.merge_size

    grid_h = grid[1].item() // merge
    grid_w = grid[2].item() // merge

    x1, y1, x2, y2 = bbox

    tx1 = int(x1 * grid_w)
    ty1 = int(y1 * grid_h)
    tx2 = min(int(np.ceil(x2 * grid_w)), grid_w)
    ty2 = min(int(np.ceil(y2 * grid_h)), grid_h)

    indices = [
        y * grid_w + x
        for y in range(ty1, ty2)
        for x in range(tx1, tx2)
    ]

    print("\nImage:", image_path)
    print("Image dimensions:", image.size)
    print("Processed grid:", grid_h, "x", grid_w)
    print("Bounding box:", bbox)
    print("Token coordinates:", tx1, ty1, tx2, ty2)
    print("Selected token count:", len(indices))
    print("Minimum index:", min(indices))
    print("Maximum index:", max(indices))
    print("Total visual tokens:", grid_h * grid_w)

    assert len(indices) == 627, "Unexpected token count"
    assert max(indices) < grid_h * grid_w

    print("Validation passed!")
