
import torch

from transformers import AutoProcessor
from src.params import DataArguments
from src.dataset.lvr_sft_dataset import (
    SupervisedDatasetLVR,
    DataCollatorForSupervisedDatasetLVR,
)

print("\nLATENT-SIGHT - ACTUAL BATCH ALIGNMENT TEST")

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DATA_PATH = "mac_demo/data/lvr_train.json"

# Load processor only; no 7B model weights are loaded
processor = AutoProcessor.from_pretrained(MODEL_ID)

processor.tokenizer.add_tokens(
    [
        "<|lvr_start|>",
        "<|lvr|>",
        "<|lvr_latent_end|>",
        "<|lvr_end|>",
    ],
    special_tokens=True,
)

lvr_id = processor.tokenizer.convert_tokens_to_ids("<|lvr|>")

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

# Create a batch containing two examples
batch = DataCollatorForSupervisedDatasetLVR(
    pad_token_id=processor.tokenizer.pad_token_id
)([dataset[0], dataset[0]])

# Get visual grid information
grid = batch["image_grid_thw"].long()
merge_size = processor.image_processor.merge_size

# Calculate visual tokens contributed by each image
tokens_per_image = (
    grid[:, 0] * grid[:, 1] * grid[:, 2]
    // (merge_size ** 2)
)

# Calculate cumulative image offsets
offsets = torch.cat([
    torch.zeros(1, dtype=torch.long),
    torch.cumsum(tokens_per_image, dim=0)[:-1]
])

print("Visual tokens per image:", tokens_per_image.tolist())
print("Image offsets:", offsets.tolist())

assert len(batch["lvr_tokens"]) == 2
assert batch["input_ids"].shape[0] == 2

all_global_indices = []

for i, local_indices in enumerate(batch["lvr_tokens"]):

    local_indices = local_indices.long()
    offset = offsets[i]

    # Convert local indices to global image-feature positions
    global_indices = local_indices + offset

    image_token_count = tokens_per_image[i]

    # Validate the actual selected indices
    assert len(local_indices) == 627
    assert local_indices.min().item() >= 0
    assert local_indices.max().item() < image_token_count
    assert len(torch.unique(local_indices)) == len(local_indices)

    # Validate global positions
    assert global_indices.min().item() >= offset.item()
    assert global_indices.max().item() < (
        offset + image_token_count
    ).item()

    # Check LVR placeholder count
    placeholder_count = (
        batch["input_ids"][i] == lvr_id
    ).sum().item()

    assert placeholder_count == len(local_indices)

    all_global_indices.append(global_indices)

    print(f"\nExample {i}:")
    print("Local index count:", len(local_indices))
    print("Local minimum:", local_indices.min().item())
    print("Local maximum:", local_indices.max().item())
    print("Global minimum:", global_indices.min().item())
    print("Global maximum:", global_indices.max().item())
    print("LVR placeholder count:", placeholder_count)

# Ensure the two examples refer to separate image regions
assert all_global_indices[0].max() < all_global_indices[1].min()

print("\nALL ACTUAL BATCH ALIGNMENT CHECKS PASSED!")
