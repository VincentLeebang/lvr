
import pandas as pd
from pathlib import Path
from PIL import Image

CSV_PATH = Path("mac_demo/data/annotations.csv")

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

print("=" * 50)
print("LATENT-SIGHT DATASET VALIDATION")
print("=" * 50)

if not CSV_PATH.exists():
    raise FileNotFoundError(CSV_PATH)

df = pd.read_csv(CSV_PATH)

errors = []

# Check columns
for column in required_columns:
    if column not in df.columns:
        errors.append(f"Missing column: {column}")

if errors:
    print("\n".join(errors))
    raise SystemExit("Dataset validation failed.")

print("Total records:", len(df))

for index, row in df.iterrows():

    print(f"\nChecking record {index + 1}")

    image_path = Path(row["image_path"])

    if not image_path.is_absolute():
        image_path = Path.cwd() / image_path

    # Check image existence
    if not image_path.exists():
        errors.append(
            f"Image missing: {image_path}"
        )
        continue

    # Load image
    image = Image.open(image_path)
    width, height = image.size

    # Check text fields
    if pd.isna(row["question"]) or not str(row["question"]).strip():
        errors.append(f"Empty question in record {index + 1}")

    if pd.isna(row["answer"]) or not str(row["answer"]).strip():
        errors.append(f"Empty answer in record {index + 1}")

    # Read bounding box
    x_min = int(row["x_min"])
    y_min = int(row["y_min"])
    x_max = int(row["x_max"])
    y_max = int(row["y_max"])

    # Validate coordinates
    if not (
        0 <= x_min < x_max <= width
        and 0 <= y_min < y_max <= height
    ):
        errors.append(
            f"Invalid bounding box in record {index + 1}"
        )

    # Check image dimensions
    if int(row["image_width"]) != width:
        errors.append(
            f"Width mismatch in record {index + 1}"
        )

    if int(row["image_height"]) != height:
        errors.append(
            f"Height mismatch in record {index + 1}"
        )

    print("Image:", image_path.name)
    print("Dimensions:", width, "x", height)
    print("Bounding box:", x_min, y_min, x_max, y_max)

print("\n" + "=" * 50)

if errors:
    print("VALIDATION FAILED")
    for error in errors:
        print("-", error)
else:
    print("VALIDATION SUCCESSFUL!")
    print("All dataset records passed the checks.")

print("=" * 50)
