
import torch
from transformers import AutoProcessor

from src.params import DataArguments
from src.dataset.lvr_sft_dataset import (
    SupervisedDatasetLVR,
    DataCollatorForSupervisedDatasetLVR,
)

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DATA_PATH = "mac_demo/data/lvr_train.json"

print("1. Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Register the LVR tokens
special_tokens = [
    "<|lvr_start|>",
    "<|lvr|>",
    "<|lvr_latent_end|>",
    "<|lvr_end|>",
]
processor.tokenizer.add_tokens(special_tokens, special_tokens=True)

lvr_id = processor.tokenizer.convert_tokens_to_ids("<|lvr|>")

print("2. Creating dataset...")
data_args = DataArguments(
    data_path=DATA_PATH,
    image_folder=None,
)

dataset = SupervisedDatasetLVR(
    data_path=DATA_PATH,
    processor=processor,
    data_args=data_args,
    model_id=MODEL_ID,
)

print("Dataset length:", len(dataset))

print("\n3. Loading first dataset example...")
item = dataset[0]

print("Dataset item keys:", list(item.keys()))
print("Input IDs shape:", item["input_ids"].shape)
print("Labels shape:", item["labels"].shape)
print("Pixel values shape:", item["pixel_values"].shape)
print("Image grid THW:", item["image_grid_thw"])

print("\n4. Checking LVR indices...")
print("Number of LVR groups:", len(item["lvr_tokens"]))

for i, indices in enumerate(item["lvr_tokens"]):
    print(f"Group {i}: {len(indices)} indices")
    print("Minimum index:", indices.min().item())
    print("Maximum index:", indices.max().item())

lvr_positions = (item["input_ids"] == lvr_id).sum().item()

print("LVR positions in input IDs:", lvr_positions)

assert len(item["lvr_tokens"]) == 1
assert len(item["lvr_tokens"][0]) == 627
assert lvr_positions == 627

print("\n5. Testing batch collator...")

collator = DataCollatorForSupervisedDatasetLVR(
    pad_token_id=processor.tokenizer.pad_token_id
)

batch = collator([item])

print("Batch keys:", list(batch.keys()))
print("Batch input IDs shape:", batch["input_ids"].shape)
print("Batch pixel values shape:", batch["pixel_values"].shape)
print("Batch image grid:", batch["image_grid_thw"])
print("Batch LVR groups:", len(batch["lvr_tokens"]))
print("Batch LVR indices:", len(batch["lvr_tokens"][0]))

batch_lvr_positions = (
    batch["input_ids"] == lvr_id
).sum().item()

print("Batch LVR placeholder count:", batch_lvr_positions)

assert batch_lvr_positions == 627
assert len(batch["lvr_tokens"]) == 1
assert len(batch["lvr_tokens"][0]) == 627

print("\nALL DATASET AND COLLATOR TESTS PASSED!")
