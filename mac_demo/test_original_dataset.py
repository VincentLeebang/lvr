
import sys
from pathlib import Path
from types import SimpleNamespace
import traceback

import torch
from transformers import AutoProcessor

# ==========================================================
# PROJECT CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.lvr_sft_dataset import SupervisedDatasetLVR


MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

DATA_PATH = PROJECT_ROOT / "mac_demo/data/lvr_train.json"


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def print_section(title):
    print("\n" + "=" * 55)
    print(title)
    print("=" * 55)


# ==========================================================
# MAIN FUNCTION
# ==========================================================

def main():

    print_section("LATENT-SIGHT - ORIGINAL LVR DATASET TEST")

    # ------------------------------------------------------
    # Step 1: Check dataset file
    # ------------------------------------------------------

    print("\nStep 1: Checking dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    print("Dataset found:", DATA_PATH)

    # ------------------------------------------------------
    # Step 2: Load Qwen processor
    # ------------------------------------------------------

    print("\nStep 2: Loading Qwen processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )

    print("Processor loaded successfully.")

    # ------------------------------------------------------
    # Step 3: Register LVR special tokens
    # ------------------------------------------------------

    print("\nStep 3: Registering LVR tokens...")

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

    for token in special_tokens:
        token_id = processor.tokenizer.convert_tokens_to_ids(token)
        print(f"{token}: {token_id}")

    # ------------------------------------------------------
    # Step 4: Configure dataset arguments
    # ------------------------------------------------------

    print("\nStep 4: Configuring dataset arguments...")

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

    # ------------------------------------------------------
    # Step 5: Load original LVR dataset
    # ------------------------------------------------------

    print("\nStep 5: Loading original LVR dataset...")

    dataset = SupervisedDatasetLVR(
        data_path=str(DATA_PATH),
        processor=processor,
        data_args=data_args,
        model_id=MODEL_ID
    )

    print("Dataset loaded successfully.")
    print("Number of samples:", len(dataset))

    if len(dataset) == 0:
        raise ValueError("Dataset contains no samples.")

    # ------------------------------------------------------
    # Step 6: Process first sample
    # ------------------------------------------------------

    print("\nStep 6: Processing first sample...")

    sample = dataset[0]

    print("Sample processed successfully.")

    # ------------------------------------------------------
    # Step 7: Inspect output tensors
    # ------------------------------------------------------

    print_section("OUTPUT TENSOR DETAILS")

    for key, value in sample.items():

        print(f"\n{key}:")

        if isinstance(value, torch.Tensor):

            print("Shape:", tuple(value.shape))
            print("Data type:", value.dtype)

        elif key == "lvr_tokens":

            print("Number of groups:", len(value))

            for index, group in enumerate(value):

                if isinstance(group, torch.Tensor):
                    print(
                        f"Group {index}:",
                        "Shape =", tuple(group.shape),
                        "Data type =", group.dtype
                    )
                else:
                    print(
                        f"Group {index}:",
                        "Type =", type(group)
                    )

        else:
            print("Type:", type(value))

    # ------------------------------------------------------
    # Step 8: Verify LVR token groups
    # ------------------------------------------------------

    print_section("LVR TOKEN STRUCTURE CHECK")

    lvr_groups = sample["lvr_tokens"]

    if not lvr_groups:
        raise ValueError("No LVR token groups found.")

    for index, group in enumerate(lvr_groups):

        if not isinstance(group, torch.Tensor):
            raise TypeError(
                f"LVR group {index} is not a tensor."
            )

        if group.ndim != 1:
            raise ValueError(
                f"LVR group {index} must be one-dimensional."
            )

        if group.numel() == 0:
            raise ValueError(
                f"LVR group {index} is empty."
            )

        print(
            f"Group {index}: PASS | "
            f"Selected indices: {group.numel()}"
        )

    selected_indices = torch.cat(lvr_groups)

    print(
        "\nTotal selected visual-token indices:",
        selected_indices.numel()
    )

    # ------------------------------------------------------
    # Step 9: Verify image grid and visual-token count
    # ------------------------------------------------------

    print_section("VISUAL TOKEN ALIGNMENT CHECK")

    if "image_grid_thw" not in sample:
        raise KeyError("image_grid_thw is missing.")

    grid = sample["image_grid_thw"]

    if grid.ndim != 2 or grid.shape[1] != 3:
        raise ValueError(
            f"Unexpected image grid shape: {tuple(grid.shape)}"
        )

    image_processor = processor.image_processor

    merge_size = getattr(
        image_processor,
        "merge_size",
        None
    )

    if merge_size is None:
        raise AttributeError(
            "Could not find merge_size in the Qwen image processor."
        )

    print("Image grid:", grid.tolist())
    print("Merge size:", merge_size)

    total_visual_tokens = 0

    for row in grid:

        temporal, height, width = row.tolist()

        visual_tokens = (
            temporal * height * width
        ) // (merge_size ** 2)

        total_visual_tokens += visual_tokens

    print("Total visual tokens:", total_visual_tokens)

    print(
        "Selected LVR tokens:",
        selected_indices.numel()
    )

    print(
        "Minimum selected index:",
        selected_indices.min().item()
    )

    print(
        "Maximum selected index:",
        selected_indices.max().item()
    )

    # ------------------------------------------------------
    # Step 10: Check index range
    # ------------------------------------------------------

    print_section("INDEX VALIDATION")

    valid_indices = (
        (selected_indices >= 0)
        & (selected_indices < total_visual_tokens)
    )

    if valid_indices.all().item():

        print("All selected indices are within the grid range.")
        print("Visual-token index range check: PASSED")

    else:

        invalid_count = (
            (~valid_indices).sum().item()
        )

        print("Invalid indices:", invalid_count)
        print("Visual-token index range check: FAILED")

    # ------------------------------------------------------
    # Step 11: Final summary
    # ------------------------------------------------------

    print_section("FINAL DATASET TEST SUMMARY")

    print("Dataset loading: PASSED")
    print("Image processing: PASSED")
    print("LVR tensor structure: PASSED")
    print("Image grid inspection: PASSED")

    if valid_indices.all().item():
        print("Visual-token index range: PASSED")
    else:
        print("Visual-token index range: FAILED")

    print("\nDataset diagnostic completed.")


# ==========================================================
# EXECUTION
# ==========================================================

if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print_section("DATASET TEST FAILED")

        print("Error:", error)

        traceback.print_exc()
