
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.lvr_sft_dataset import (
    SupervisedDatasetLVR,
    DataCollatorForSupervisedDatasetLVR
)


MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
DATA_PATH = PROJECT_ROOT / "mac_demo/data/lvr_train.json"


def main():

    print("\n" + "=" * 55)
    print("LATENT-SIGHT - DATA COLLATOR TEST")
    print("=" * 55)

    # Step 1: Load processor
    print("\nStep 1: Loading processor...")

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    special_tokens = [
        "<|lvr_start|>",
        "<|lvr|>",
        "<|lvr_latent_end|>",
        "<|lvr_end|>"
    ]

    processor.tokenizer.add_tokens(
        special_tokens,
        special_tokens=True
    )

    print("Processor loaded.")

    # Step 2: Configure dataset
    print("\nStep 2: Configuring dataset...")

    data_args = SimpleNamespace(
        data_path=str(DATA_PATH),
        image_folder="",
        image_min_pixels=3136,
        image_max_pixels=12845056,
        video_min_pixels=100352,
        video_max_pixels=602112,
        image_resized_width=None,
        image_resized_height=None,
        video_resized_width=None,
        video_resized_height=None,
        fps=1.0
    )

    dataset = SupervisedDatasetLVR(
        data_path=str(DATA_PATH),
        processor=processor,
        data_args=data_args,
        model_id=MODEL_ID
    )

    print("Dataset samples:", len(dataset))

    # Step 3: Load one sample
    print("\nStep 3: Loading sample...")

    sample = dataset[0]

    print("Sample loaded.")

    # Step 4: Initialize collator
    print("\nStep 4: Initializing data collator...")

    collator = DataCollatorForSupervisedDatasetLVR(
        pad_token_id=processor.tokenizer.pad_token_id
    )

    # Step 5: Create batch
    print("\nStep 5: Creating batch...")

    batch = collator([sample])

    print("Batch created successfully.")

    # Step 6: Inspect batch
    print("\n" + "=" * 55)
    print("BATCH DETAILS")
    print("=" * 55)

    for key, value in batch.items():

        print(f"\n{key}:")

        if isinstance(value, torch.Tensor):

            print("Shape:", tuple(value.shape))
            print("Data type:", value.dtype)

        elif isinstance(value, list):

            print("Number of groups:", len(value))

            for index, item in enumerate(value):

                if isinstance(item, torch.Tensor):
                    print(
                        f"Group {index}:",
                        tuple(item.shape),
                        item.dtype
                    )
                else:
                    print(f"Group {index}: {type(item)}")

        else:
            print("Type:", type(value))

    # Step 7: Validate batch
    print("\n" + "=" * 55)
    print("COLLATOR VALIDATION")
    print("=" * 55)

    assert batch["input_ids"].shape[0] == 1
    print("Batch dimension: PASS")

    assert batch["labels"].shape == batch["input_ids"].shape
    print("Labels shape: PASS")

    assert batch["attention_mask"].shape == batch["input_ids"].shape
    print("Attention mask: PASS")

    assert "pixel_values" in batch
    print("Image pixel values: PASS")

    assert "image_grid_thw" in batch
    print("Image grid: PASS")

    assert len(batch["lvr_tokens"]) == 1
    print("LVR token groups: PASS")

    assert batch["lvr_tokens"][0].ndim == 1
    print("LVR index dimensions: PASS")

    print("\nALL COLLATOR TESTS PASSED")


if __name__ == "__main__":

    try:
        main()

    except Exception as error:
        print("\nCOLLATOR TEST FAILED")
        print("Error:", error)
        raise
