
import pandas as pd
from pathlib import Path
from PIL import Image


class LVRDataset:

    def __init__(self, csv_path):

        self.csv_path = Path(csv_path).resolve()
        self.project_dir = self.csv_path.parents[2]

        self.data = pd.read_csv(self.csv_path)

        required_columns = [
            "image_path",
            "question",
            "answer",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
            "image_width",
            "image_height"
        ]

        missing = [
            col for col in required_columns
            if col not in self.data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

        print("Dataset loaded successfully!")
        print("Total records:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = (
            self.project_dir / row["image_path"]
        )

        image = Image.open(image_path).convert("RGB")

        width, height = image.size

        bbox = [
            int(row["x_min"]),
            int(row["y_min"]),
            int(row["x_max"]),
            int(row["y_max"])
        ]

        # Normalize bounding box coordinates
        normalized_bbox = [
            bbox[0] / width,
            bbox[1] / height,
            bbox[2] / width,
            bbox[3] / height
        ]

        return {
            "image": image,
            "image_path": str(image_path),
            "question": str(row["question"]),
            "answer": str(row["answer"]),
            "bbox": bbox,
            "normalized_bbox": normalized_bbox,
            "image_width": width,
            "image_height": height
        }


if __name__ == "__main__":

    dataset = LVRDataset(
        "mac_demo/data/annotations.csv"
    )

    sample = dataset[0]

    print("\nFirst dataset sample")
    print("Question:", sample["question"])
    print("Answer:", sample["answer"])
    print("Bounding Box:", sample["bbox"])
    print("Normalized ROI:", sample["normalized_bbox"])
    print("Image Size:",
          sample["image_width"],
          "x",
          sample["image_height"])
