
import torch
import torch.nn.functional as F

from mac_demo.test_real_batch import dataset, collator

print("\nREAL BATCH ALIGNMENT TEST")

example = dataset[0]
batch = collator([example, example])

input_ids = batch["input_ids"]
image_grid_thw = batch["image_grid_thw"]
lvr_tokens = batch["lvr_tokens"]

image_token_id = dataset.processor.tokenizer.convert_tokens_to_ids(
    "<|image_pad|>"
)

image_mask = input_ids == image_token_id

tokens_per_example = image_mask.sum(dim=1)

offsets = torch.cumsum(
    F.pad(tokens_per_example, (1, 0)), dim=0
)[:-1]

global_indices = []

for i, roi_indices in enumerate(lvr_tokens):
    global_indices.append(roi_indices + offsets[i])

global_indices = torch.cat(global_indices)

print("Image tokens per example:", tokens_per_example.tolist())
print("Image offsets:", offsets.tolist())
print("ROI groups:", len(lvr_tokens))
print("Total selected ROI indices:", len(global_indices))
print("Minimum global index:", global_indices.min().item())
print("Maximum global index:", global_indices.max().item())
print("Total image features:", image_mask.sum().item())

assert len(lvr_tokens) == 2
assert len(global_indices) == 1254
assert global_indices.min() >= 0
assert global_indices.max() < image_mask.sum()

print("\nREAL BATCH ALIGNMENT TEST PASSED")
