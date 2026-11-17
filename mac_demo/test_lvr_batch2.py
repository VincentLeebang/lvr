
from transformers import AutoProcessor
from src.params import DataArguments
from src.dataset.lvr_sft_dataset import (
    SupervisedDatasetLVR,
    DataCollatorForSupervisedDatasetLVR,
)

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DATA_PATH = "mac_demo/data/lvr_train.json"

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

# Load one example twice to simulate a batch of two examples
item1 = dataset[0]
item2 = dataset[0]

collator = DataCollatorForSupervisedDatasetLVR(
    pad_token_id=processor.tokenizer.pad_token_id
)

batch = collator([item1, item2])

print("Batch input shape:", batch["input_ids"].shape)
print("Batch pixel shape:", batch["pixel_values"].shape)
print("Batch image grid:", batch["image_grid_thw"])
print("Number of LVR groups:", len(batch["lvr_tokens"]))

for i in range(batch["input_ids"].shape[0]):
    count = (batch["input_ids"][i] == lvr_id).sum().item()
    print(f"Example {i}: {count} LVR positions")

for i, indices in enumerate(batch["lvr_tokens"]):
    print(
        f"Group {i}: {len(indices)} indices, "
        f"min={indices.min().item()}, "
        f"max={indices.max().item()}"
    )

assert batch["input_ids"].shape[0] == 2
assert len(batch["lvr_tokens"]) == 2
assert all(len(indices) == 627 for indices in batch["lvr_tokens"])

print("\nBATCH SIZE 2 VALIDATION PASSED!")
