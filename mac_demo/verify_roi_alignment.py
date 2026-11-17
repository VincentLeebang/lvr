
import csv
import numpy as np
from PIL import Image
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "annotations.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

ROI_PATH = OUTPUT_DIR / "roi_crop.jpg"

print("=" * 55)
print("LATENT-SIGHT - ROI ALIGNMENT VERIFICATION")
print("=" * 55)

# Load annotation
with open(CSV_PATH, "r", encoding="utf-8") as file:
    annotations = list(csv.DictReader(file))

if not annotations:
    raise ValueError("No annotations found.")

# Use first annotation
sample = annotations[0]

IMAGE_PATH = Path(sample["image_path"])

if not IMAGE_PATH.is_absolute():
    IMAGE_PATH = BASE_DIR.parent / IMAGE_PATH

# Read image
image = Image.open(IMAGE_PATH).convert("RGB")
image_width, image_height = image.size

# Read bounding-box coordinates
x_min = int(sample["x_min"])
y_min = int(sample["y_min"])
x_max = int(sample["x_max"])
y_max = int(sample["y_max"])

print("\nImage:", IMAGE_PATH)
print("Image dimensions:", image.size)

print("\nBounding box:")
print("X minimum:", x_min)
print("Y minimum:", y_min)
print("X maximum:", x_max)
print("Y maximum:", y_max)

# Validate coordinates
if not (
    0 <= x_min < x_max <= image_width
    and 0 <= y_min < y_max <= image_height
):
    raise ValueError("Bounding box coordinates are invalid.")

print("\nBounding box coordinates are valid.")

# Calculate ROI dimensions
roi_width = x_max - x_min
roi_height = y_max - y_min

print("ROI width:", roi_width)
print("ROI height:", roi_height)

# Generate crop again from original image
expected_roi = image.crop(
    (x_min, y_min, x_max, y_max)
)

# Check existing ROI crop
if not ROI_PATH.exists():
    raise FileNotFoundError(
        f"Existing ROI crop not found: {ROI_PATH}"
    )

saved_roi = Image.open(ROI_PATH).convert("RGB")

print("\nExpected ROI size:", expected_roi.size)
print("Saved ROI size:", saved_roi.size)

# Compare crop dimensions and pixel values
if expected_roi.size != saved_roi.size:
    print("\nRESULT: ROI dimensions do not match.")
else:
    expected_array = np.asarray(expected_roi)
    saved_array = np.asarray(saved_roi)

    if np.array_equal(expected_array, saved_array):
        print("\nRESULT: ROI ALIGNMENT VERIFIED.")
        print("Saved ROI exactly matches the annotated crop.")
    else:
        difference = np.abs(
            expected_array.astype(np.int16)
            - saved_array.astype(np.int16)
        )

        print("\nRESULT: ROI pixel values differ.")
        print(
            "Mean absolute pixel difference:",
            round(float(difference.mean()), 4)
        )

# Save a lossless PNG reference
PNG_PATH = OUTPUT_DIR / "roi_crop_verified.png"
expected_roi.save(PNG_PATH)

# Reload PNG and compare exact pixels
png_roi = Image.open(PNG_PATH).convert("RGB")

if np.array_equal(
    np.asarray(expected_roi),
    np.asarray(png_roi)
):
    print("\nPNG VERIFICATION: PASSED")
    print("Lossless ROI pixels match exactly.")
else:
    print("\nPNG VERIFICATION: FAILED")


print("\nROI alignment verification completed.")
print("=" * 55)
