
import json
from pathlib import Path

JSON_PATH = Path("mac_demo/data/lvr_train.json")

# Load the JSON dataset
with open(JSON_PATH, "r", encoding="utf-8") as file:
    dataset = json.load(file)

# Check dataset structure
assert isinstance(dataset, list), "Dataset must be a list"
assert len(dataset) > 0, "Dataset is empty"

for i, sample in enumerate(dataset):

    assert "image" in sample, f"Sample {i}: image field missing"
    assert "bboxes" in sample, f"Sample {i}: bboxes field missing"
    assert "conversations" in sample, f"Sample {i}: conversations field missing"

    # Check image
    assert len(sample["image"]) > 0, f"Sample {i}: no image path"

    image_path = Path(sample["image"][0])
    assert image_path.exists(), f"Missing image: {image_path}"

    # Check bounding boxes
    assert len(sample["bboxes"]) > 0, f"Sample {i}: no bounding box"

    for bbox in sample["bboxes"]:
        assert len(bbox) == 4, f"Sample {i}: bbox must contain 4 values"
        assert all(0 <= value <= 1 for value in bbox), (
            f"Sample {i}: bbox values must be between 0 and 1"
        )
        assert bbox[0] < bbox[2], f"Sample {i}: invalid horizontal coordinates"
        assert bbox[1] < bbox[3], f"Sample {i}: invalid vertical coordinates"

    # Combine conversation text
    conversation_text = " ".join(
        message["value"] for message in sample["conversations"]
    )

    # Check the dataset placeholder
    assert "<lvr>" in conversation_text, (
        f"Sample {i}: <lvr> placeholder missing"
    )

    # Check answer tag
    assert "<answer>" in conversation_text, (
        f"Sample {i}: <answer> tag missing"
    )

print("=" * 45)
print("LVR JSON VALIDATION")
print("=" * 45)
print("Number of samples:", len(dataset))
print("Image exists: YES")
print("Bounding boxes: VALID")
print("LVR placeholder: FOUND")
print("Answer tag: FOUND")
print("\nJSON validation completed successfully!")
