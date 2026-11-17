
import torch
import torchvision.models as models
import torchvision.transforms as transforms

import pandas as pd
from PIL import Image
from pathlib import Path


print("=" * 55)
print("LATENT-SIGHT - ROI FEATURE EXTRACTION")
print("=" * 55)

# Select device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("Device:", DEVICE)

# Load annotation
csv_path = Path("mac_demo/data/annotations.csv")
annotations = pd.read_csv(csv_path)

sample = annotations.iloc[0]

# Load image
image_path = Path(sample["image_path"])
image = Image.open(image_path).convert("RGB")

# Read ROI coordinates
x_min = int(sample["x_min"])
y_min = int(sample["y_min"])
x_max = int(sample["x_max"])
y_max = int(sample["y_max"])

print("Image:", image_path)
print("Original image size:", image.size)

# Crop ROI
roi_image = image.crop(
    (x_min, y_min, x_max, y_max)
)

print("ROI size:", roi_image.size)

# Save ROI image
output_dir = Path("mac_demo/outputs")
output_dir.mkdir(parents=True, exist_ok=True)

roi_image.save(output_dir / "roi_crop.jpg")

# Load pretrained ResNet18
print("\nLoading pretrained ResNet18...")

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Remove final classification layer
model.fc = torch.nn.Identity()

model = model.to(DEVICE)
model.eval()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

input_tensor = preprocess(
    roi_image
).unsqueeze(0).to(DEVICE)

# Extract features
with torch.inference_mode():
    features = model(input_tensor)

# Save features
features = features.cpu()

torch.save(
    features,
    output_dir / "roi_features.pt"
)

print("\nFEATURE EXTRACTION SUCCESSFUL!")
print("Feature shape:", features.shape)
print("Saved ROI image:", output_dir / "roi_crop.jpg")
print("Saved features:", output_dir / "roi_features.pt")

print("\nEXPERIMENT COMPLETED!")
