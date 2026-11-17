
import torch

from src.dataset.lvr_sft_dataset import (
    SupervisedDatasetLVR,
    DataCollatorForSupervisedDatasetLVR,
)

print("\nLATENT-SIGHT - REAL BATCH TEST")

# This test uses the existing dataset and processor.
# We will reuse the processor-loading approach from
# your existing dataset pipeline test.

from transformers import AutoProcessor
from src.params import DataArguments

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

data_args = DataArguments()
data_args.data_path = "mac_demo/data/lvr_train.json"

processor = AutoProcessor.from_pretrained(
    model_id,
    min_pixels=data_args.image_min_pixels,
    max_pixels=data_args.image_max_pixels,
)

dataset = SupervisedDatasetLVR(
    data_path=data_args.data_path,
    processor=processor,
    data_args=data_args,
    model_id=model_id,
)

collator = DataCollatorForSupervisedDatasetLVR(
    pad_token_id=processor.tokenizer.pad_token_id
)

# Duplicate the single example to create a batch of two.
example = dataset[0]
batch = collator([example, example])

print("\nBATCH RESULTS")
for key, value in batch.items():
    if isinstance(value, torch.Tensor):
        print(key, value.shape)
    elif isinstance(value, list):
        print(key, "number of groups:", len(value))
        for i, group in enumerate(value):
            print(f"  Group {i}:", group.shape)

print("\nREAL BATCH TEST COMPLETED")
