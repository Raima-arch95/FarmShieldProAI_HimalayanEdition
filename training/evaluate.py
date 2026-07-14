from pathlib import Path
import json
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader, random_split

from datasets.farmshield_dataset import FarmShieldDataset
from utils.transforms import get_training_transforms
from models.mobilenetv3 import create_model
from checkpoint import load_checkpoint
from engine import evaluate_model

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "datasets" / "FarmShieldDataset"

LABELS_PATH = PROJECT_ROOT / "documentation" / "labels.json"

# ==========================================================
# Export Paths
# ==========================================================

EXPORT_DIR = PROJECT_ROOT / "exports"

EXPORT_DIR.mkdir(exist_ok=True)

CONFUSION_MATRIX_PATH = (
    EXPORT_DIR /
    "confusion_matrix.png"
)

# ==========================================================
# Load Labels
# ==========================================================

with open(LABELS_PATH, "r", encoding="utf-8") as file:
    label_map = json.load(file)

NUM_CLASSES = len(label_map)

# ==========================================================
# Class Names
# ==========================================================

class_names = [
    f'{item["crop"]} - {item["disease"]}'
    for item in label_map
]

print(f"Loaded {NUM_CLASSES} classes.")

# ==========================================================
# Dataset
# ==========================================================

dataset = FarmShieldDataset(
    dataset_path=DATASET_PATH,
    labels_path=LABELS_PATH,
    transform=get_training_transforms()
)

print(f"Dataset contains {len(dataset)} images.")

# ==========================================================
# Train / Validation Split
# ==========================================================

train_size = int(0.8 * len(dataset))

validation_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, validation_dataset = random_split(
    dataset,
    [train_size, validation_size],
    generator=generator
)

print(f"Training Images   : {len(train_dataset)}")
print(f"Validation Images : {len(validation_dataset)}")

# ==========================================================
# DataLoaders
# ==========================================================

validation_loader = DataLoader(
    validation_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print(f"Validation Batches : {len(validation_loader)}")

# ==========================================================
# Model
# ==========================================================

model = create_model(NUM_CLASSES)

model.to(DEVICE)

checkpoint = load_checkpoint(model)

print(
    f"Loaded checkpoint from Epoch "
    f"{checkpoint['epoch']} "
    f"(Validation Accuracy: "
    f"{checkpoint['validation_accuracy']:.2f}%)"
)

# ==========================================================
# Evaluation
# ==========================================================

predictions, labels = evaluate_model(
    model=model,
    dataloader=validation_loader,
    device=DEVICE,
)

correct = sum(
    prediction == label
    for prediction, label in zip(predictions, labels)
)

accuracy = 100 * correct / len(labels)

print("\n====================================")
print("Evaluation Results")
print("====================================")
print(f"Overall Accuracy : {accuracy:.2f}%")

print("\n====================================")
print("Classification Report")
print("====================================")

print(
    classification_report(
        labels,
        predictions,
        target_names=class_names,
        digits=4,
    )
)

cm = confusion_matrix(
    labels,
    predictions
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(20, 20))

display.plot(
    ax=ax,
    xticks_rotation=90,
    colorbar=False
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300,
    bbox_inches="tight"
)

print(
    f"\nConfusion matrix saved to:\n"
    f"{CONFUSION_MATRIX_PATH}"
)

plt.close()