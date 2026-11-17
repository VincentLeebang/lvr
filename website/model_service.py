import math
from transformers import AutoProcessor
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# ==========================================
# LATENT-SIGHT MODEL SERVICE
# ==========================================

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

processor = None
model = None
device = None


# ==========================================
# LOAD MODEL ONCE
# ==========================================

def load_model():

    global processor, model, device

    if model is not None:
        return

    print("Loading LATENT-SIGHT vision model...")

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    dtype = torch.float16 if device == "mps" else torch.float32

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype
    )

    model = model.to(device)
    model.eval()

    print("Model loaded successfully!")
    print("Device:", device)


# ==========================================
# IMAGE QUESTION ANSWERING
# ==========================================

def analyze_image(image: Image.Image, question: str):

    load_model()

    image = image.convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": question
                }
            ]
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in inputs.items()
    }

    print("Generating answer...")

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False
        )

    # Decode only newly generated tokens
    input_length = inputs["input_ids"].shape[1]

    generated_tokens = output[0][input_length:]

    answer = processor.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    if not answer:
        answer = "The model did not generate a response."

    print("Analysis completed.")

    return answer

# ==========================================
# QWEN LVR ROI-TO-TOKEN MAPPING
# ==========================================

QWEN_PROCESSOR_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

qwen_processor = None


def load_qwen_processor():
    """Load only the Qwen processor, not the 7B model."""

    global qwen_processor

    if qwen_processor is None:
        print("Loading Qwen2.5-VL image processor...")

        qwen_processor = AutoProcessor.from_pretrained(
            QWEN_PROCESSOR_ID
        )

        print("Qwen processor loaded successfully.")

    return qwen_processor


def map_roi_to_qwen_tokens(image, x_min, y_min, x_max, y_max):
    """
    Map an original-image bounding box to Qwen merged visual-token indices.

    Coordinates are measured in original image pixels.
    """

    image = image.convert("RGB")

    image_width, image_height = image.size

    # Validate bounding box.
    if not (
        0 <= x_min < x_max <= image_width
        and 0 <= y_min < y_max <= image_height
    ):
        raise ValueError(
            "Invalid ROI coordinates for the uploaded image."
        )

    processor = load_qwen_processor()

    # Run the actual Qwen image processor.
    processed = processor.image_processor(
        images=image,
        return_tensors="pt"
    )

    image_grid = processed["image_grid_thw"]

    # Qwen grid dimensions before spatial merging.
    temporal, grid_height, grid_width = (
        int(value) for value in image_grid[0].tolist()
    )

    merge_size = int(
        getattr(processor.image_processor, "merge_size", 2)
    )

    # Normalize ROI coordinates.
    nx0 = x_min / image_width
    ny0 = y_min / image_height
    nx1 = x_max / image_width
    ny1 = y_max / image_height

    # Map normalized coordinates to Qwen patch grid.
    gx0 = math.floor(nx0 * grid_width)
    gy0 = math.floor(ny0 * grid_height)

    gx1 = math.ceil(nx1 * grid_width)
    gy1 = math.ceil(ny1 * grid_height)

    # Convert patch-grid coordinates to merged-token coordinates.
    tx0 = gx0 // merge_size
    ty0 = gy0 // merge_size

    tx1 = (gx1 + merge_size - 1) // merge_size
    ty1 = (gy1 + merge_size - 1) // merge_size

    merged_height = grid_height // merge_size
    merged_width = grid_width // merge_size

    # Keep indices inside the merged visual grid.
    tx0 = max(0, min(tx0, merged_width))
    tx1 = max(0, min(tx1, merged_width))
    ty0 = max(0, min(ty0, merged_height))
    ty1 = max(0, min(ty1, merged_height))

    # Generate row-major visual-token indices.
    token_indices = []

    for y in range(ty0, ty1):
        for x in range(tx0, tx1):
            token_indices.append(
                y * merged_width + x
            )

    return {
        "image_width": image_width,
        "image_height": image_height,
        "image_grid_thw": [
            temporal,
            grid_height,
            grid_width
        ],
        "merge_size": merge_size,
        "merged_grid_height": merged_height,
        "merged_grid_width": merged_width,
        "roi_token_grid": {
            "x_start": tx0,
            "y_start": ty0,
            "x_end": tx1,
            "y_end": ty1
        },
        "token_indices": token_indices,
        "selected_token_count": len(token_indices)
    }
