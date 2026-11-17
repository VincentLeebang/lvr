
import pandas as pd
import json
from pathlib import Path

CSV_PATH = Path("mac_demo/data/annotations.csv")
OUTPUT_PATH = Path("mac_demo/data/lvr_train.json")

df = pd.read_csv(CSV_PATH)

lvr_dataset = []

for index, row in df.iterrows():

    image_path = Path(row["image_path"])

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image missing: {image_path}"
        )

    width = float(row["image_width"])
    height = float(row["image_height"])

    # Convert pixel coordinates to normalized coordinates
    bbox = [
        float(row["x_min"]) / width,
        float(row["y_min"]) / height,
        float(row["x_max"]) / width,
        float(row["y_max"]) / height
    ]

    # Official LVR Stage-1 data structure
    sample = {
        "dataset": "custom_mac_demo",
        "split": "train",
        "question_id": int(index + 1),

        "image": [
            image_path.as_posix()
        ],

        "conversations": [
            {
                "from": "human",
                "value": str(row["question"])
            },
            {
                "from": "gpt",
                "value": (
                    "<lvr>\n"
                    "<answer> "
                    + str(row["answer"])
                    + " </answer>"
                )
            }
        ],

        "bboxes": [bbox]
    }

    lvr_dataset.append(sample)


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:

    json.dump(
        lvr_dataset,
        file,
        indent=4,
        ensure_ascii=False
    )

print("=" * 45)
print("LVR DATASET CONVERSION")
print("=" * 45)

print("Total samples:", len(lvr_dataset))
print("Output:", OUTPUT_PATH)
print("First sample:")
print(json.dumps(lvr_dataset[0], indent=4))

print("\nConversion completed successfully!")
