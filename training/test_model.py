from pathlib import Path
import json

from models.mobilenetv3 import create_model

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LABELS_PATH = PROJECT_ROOT / "documentation" / "labels.json"

# ==========================================================
# Load Labels
# ==========================================================

with open(LABELS_PATH, "r", encoding="utf-8") as file:
    labels = json.load(file)

NUM_CLASSES = len(labels)

# ==========================================================
# Create Model
# ==========================================================

model = create_model(NUM_CLASSES)

# ==========================================================
# Display Information
# ==========================================================

print("=" * 60)
print("MobileNetV3 Test")
print("=" * 60)

print(f"Number of Classes : {NUM_CLASSES}")

print("\nClassifier Output Features:")

print(model.classifier[-1].out_features)

print("\nModel Architecture:\n")

print(model)
