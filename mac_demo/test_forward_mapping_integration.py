
import torch
import torch.nn.functional as F
from transformers import AutoProcessor

from src.params import DataArguments
from src.dataset.lvr_sft_dataset import (
    SupervisedDatasetLVR,
    DataCollatorForSupervisedDatasetLVR,
)

print("\nLATENT-SIGHT - FORWARD MAPPING INTEGRATION TEST")

# 1. Load processor
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

data_args = DataArguments()
data_args.data_path = "mac_demo/data/lvr_train.json"

processor = AutoProcessor.from_pretrained(
    model_id,
    min_pixels=data_args.image_min_pixels,
    max_pixels=data_args.image_max_pixels,
)

# Register LVR tokens, as done during training initialization.
processor.tokenizer.add_tokens(
    [
        "<|lvr_start|>",
        "<|lvr|>",
        "<|lvr_latent_end|>",
        "<|lvr_end|>",
    ],
    special_tokens=True,
)

# 2. Load real dataset and collator
dataset = SupervisedDatasetLVR(
    data_path=data_args.data_path,
    processor=processor,
    data_args=data_args,
    model_id=model_id,
)

collator = DataCollatorForSupervisedDatasetLVR(
    pad_token_id=processor.tokenizer.pad_token_id
)

# Create batch of two real examples
batch = collator([dataset[0], dataset[0]])

input_ids = batch["input_ids"]
lvr_tokens = batch["lvr_tokens"]

image_token_id = processor.tokenizer.convert_tokens_to_ids(
    "<|image_pad|>"
)
lvr_token_id = processor.tokenizer.convert_tokens_to_ids(
    "<|lvr|>"
)

# 3. Create image-token mask
image_mask = input_ids == image_token_id
lvr_mask = input_ids == lvr_token_id

tokens_per_example = image_mask.sum(dim=1)

print("\nIMAGE TOKEN COUNTS:", tokens_per_example.tolist())
print("LVR PLACEHOLDER COUNTS:", lvr_mask.sum(dim=1).tolist())

# 4. Generate image feature offsets
image_token_offsets = torch.cumsum(
    F.pad(tokens_per_example, (1, 0)), dim=0
)[:-1]

print("IMAGE OFFSETS:", image_token_offsets.tolist())

# 5. Generate global ROI feature indices
global_indices = []

for i, roi_indices in enumerate(lvr_tokens):
    global_indices.append(
        roi_indices + image_token_offsets[i]
    )

global_indices = torch.cat(global_indices)

# 6. Create synthetic visual embeddings
# Hidden dimension 32 is used only to test mapping mechanics.
hidden_size = 32
total_image_tokens = int(tokens_per_example.sum().item())

image_embeds = torch.randn(
    total_image_tokens,
    hidden_size
)

inputs_embeds = torch.randn(
    input_ids.shape[0],
    input_ids.shape[1],
    hidden_size
)

# 7. Insert image embeddings into image placeholders
image_mask_expanded = (
    image_mask.unsqueeze(-1)
    .expand_as(inputs_embeds)
)

inputs_embeds = inputs_embeds.masked_scatter(
    image_mask_expanded,
    image_embeds
)

# 8. Select ROI embeddings
selected_lvr_embeds = image_embeds[global_indices]

# 9. Find LVR placeholder positions
batch_indices, seq_positions = torch.nonzero(
    lvr_mask,
    as_tuple=True
)

# 10. Insert ROI embeddings into LVR placeholders
assert len(batch_indices) == len(selected_lvr_embeds), (
    "Mismatch between LVR placeholders and selected features"
)

inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds

# 11. Verify the results
inserted_embeds = inputs_embeds[
    batch_indices,
    seq_positions
]

print("\nVALIDATION RESULTS")
print("Total image features:", total_image_tokens)
print("Total ROI indices:", len(global_indices))
print("Total LVR placeholders:", len(batch_indices))
print("Selected ROI shape:", selected_lvr_embeds.shape)
print("Inserted ROI shape:", inserted_embeds.shape)
print("All indices valid:", bool(
    global_indices.min() >= 0
    and global_indices.max() < total_image_tokens
))
print("All inserted values finite:", bool(
    torch.isfinite(inserted_embeds).all()
))
print("ROI insertion matches selected features:", bool(
    torch.allclose(inserted_embeds, selected_lvr_embeds)
))

# Final assertions
assert len(global_indices) == 1254
assert len(batch_indices) == 1254
assert inserted_embeds.shape == (1254, hidden_size)
assert torch.isfinite(inserted_embeds).all()
assert torch.allclose(inserted_embeds, selected_lvr_embeds)

print("\nFORWARD MAPPING INTEGRATION TEST PASSED!")
