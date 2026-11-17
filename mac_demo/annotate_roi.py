
import cv2
import csv
from pathlib import Path

# Project paths
IMAGE_PATH = Path("mac_demo/images/test.jpg")
OUTPUT_DIR = Path("mac_demo/data")
CSV_PATH = OUTPUT_DIR / "annotations.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load image
image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

height, width = image.shape[:2]

print("Image loaded successfully!")
print("Image width:", width)
print("Image height:", height)

# Select the region of interest
print("\nDraw a rectangle around the important objects.")
print("Press ENTER or SPACE to confirm.")
print("Press ESC to cancel.")

roi = cv2.selectROI(
    "Select Region of Interest",
    image,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyAllWindows()

x, y, w, h = map(int, roi)

if w == 0 or h == 0:
    raise ValueError("No ROI selected. Run again.")

x_min = x
y_min = y
x_max = x + w
y_max = y + h

# Enter question and verified answer
question = input("\nEnter your image question: ").strip()
answer = input("Enter the verified correct answer: ").strip()

if not question or not answer:
    raise ValueError("Question and answer cannot be empty.")

# Save annotation
columns = [
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

record = {
    "image_path": str(IMAGE_PATH),
    "question": question,
    "answer": answer,
    "x_min": x_min,
    "y_min": y_min,
    "x_max": x_max,
    "y_max": y_max,
    "image_width": width,
    "image_height": height
}

file_exists = CSV_PATH.exists()

with open(CSV_PATH, "a", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(file, fieldnames=columns)

    if not file_exists:
        writer.writeheader()

    writer.writerow(record)

print("\nROI annotation saved successfully!")
print("Saved at:", CSV_PATH)
print("Bounding box:", x_min, y_min, x_max, y_max)
