

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "images" / "test.jpg"
ROI_PATH = BASE_DIR / "outputs" / "roi_crop_verified.png"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# Select device
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)

# Load pretrained ResNet18
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.fc = torch.nn.Identity()
model = model.to(device)
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

def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(image_tensor)

    return features

# Extract full image features
print("\nExtracting full image features...")
full_features = extract_features(IMAGE_PATH)

# Extract ROI features
print("Extracting ROI features...")
roi_features = extract_features(ROI_PATH)

# Cosine similarity
similarity = F.cosine_similarity(
    full_features,
    roi_features,
    dim=1
).item()

# Save features
torch.save(
    full_features.cpu(),
    OUTPUT_DIR / "full_image_features.pt"
)

torch.save(
    roi_features.cpu(),
    OUTPUT_DIR / "roi_comparison_features.pt"
)

# Save comparison results
results = pd.DataFrame({
    "Feature Type": ["Full Image", "ROI"],
    "Feature Dimensions": [
        str(tuple(full_features.shape)),
        str(tuple(roi_features.shape))
    ]
})

results.to_csv(
    OUTPUT_DIR / "feature_comparison.csv",
    index=False
)

print("\n========== COMPARISON RESULTS ==========")
print("Full image feature shape:", full_features.shape)
print("ROI feature shape:", roi_features.shape)
print(f"Cosine similarity: {similarity:.4f}")

print("\nSaved comparison files successfully!")
print("========================================")
