
import torch
import torch.nn.functional as F

from transformers import AutoProcessor
from src.params import DataArguments
from src.dataset.lvr_sft_dataset import (
    SupervisedDatasetLVR,
    DataCollatorForSupervisedDatasetLVR,
)

print("\nLATENT-SIGHT - FORWARD MAPPING SIMULATION")

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DATA_PATH = "mac_demo/data/lvr_train.json"

# Load processor only, not the 7B model
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

collator = DataCollatorForSupervisedDatasetLVR(
    pad_token_id=processor.tokenizer.pad_token_id
)

batch = collator([dataset[0], dataset[0]])

input_ids = batch["input_ids"]
lvr_tokens = batch["lvr_tokens"]

# Reproduce image mask and image feature offsets
image_mask = input_ids == processor.tokenizer.convert_tokens_to_ids(
    "<|image_pad|>"
)

total_tokens = image_mask.sum(dim=1)

merge_size = processor.image_processor.merge_size
grid = batch["image_grid_thw"].long()

expected_tokens = (
    grid[:, 0] * grid[:, 1] * grid[:, 2]
    // (merge_size ** 2)
)

assert torch.equal(total_tokens, expected_tokens), (
    f"Image token counts do not match: "
    f"{total_tokens.tolist()} vs {expected_tokens.tolist()}"
)

image_token_offsets = torch.cumsum(
    F.pad(total_tokens, (1, 0)), dim=0
)[:-1]

print("Image tokens per example:", total_tokens.tolist())
print("Image offsets:", image_token_offsets.tolist())

# Create synthetic image features.
# Each feature row contains its own global index.
num_features = int(total_tokens.sum().item())
hidden_size = 4

image_embeds = (
    torch.arange(num_features, dtype=torch.float32)
    .unsqueeze(1)
    .repeat(1, hidden_size)
)

# Simulate the input embedding tensor
inputs_embeds = torch.zeros(
    input_ids.shape[0],
    input_ids.shape[1],
    hidden_size,
)

# Reproduce image feature insertion
image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(
    inputs_embeds
)

inputs_embeds = inputs_embeds.masked_scatter(
    image_mask_unsqueeze,
    image_embeds,
)

# Reproduce the LVR insertion logic
lvr_mask = input_ids == lvr_id
batch_indices, seq_positions = torch.nonzero(
    lvr_mask, as_tuple=True
)

global_lvr_token_indices = []

for b, lvr_ids in enumerate(lvr_tokens):
    offset = image_token_offsets[b].item()
    global_lvr_token_indices.append(lvr_ids + offset)

global_lvr_token_indices = torch.cat(
    global_lvr_token_indices, dim=0
)

selected_lvr_embeds = image_embeds[
    global_lvr_token_indices
]

assert len(batch_indices) == len(global_lvr_token_indices)

inputs_embeds[batch_indices, seq_positions] = (
    selected_lvr_embeds
)

# Verify each example independently
for b in range(input_ids.shape[0]):
    positions = torch.nonzero(
        lvr_mask[b], as_tuple=True
    )[0]

    offset = image_token_offsets[b].item()
    expected = lvr_tokens[b] + offset

    actual = inputs_embeds[b, positions, 0]

    assert torch.equal(
        actual,
        expected.float()
    ), f"Feature mapping failed for example {b}"

    print(
        f"Example {b}: {len(positions)} features inserted correctly"
    )

print("\nFORWARD MAPPING SIMULATION PASSED!")
