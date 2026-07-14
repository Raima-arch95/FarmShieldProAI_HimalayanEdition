from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import json
from engine import train_one_epoch, validate
from torch.utils.data import DataLoader, random_split
from checkpoint import save_checkpoint

from datasets.farmshield_dataset import FarmShieldDataset
from utils.transforms import get_training_transforms
from models.mobilenetv3 import create_model

# ==========================================================
# Training Configuration
# ==========================================================

EPOCHS = 10

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "datasets" / "FarmShieldDataset"

LABELS_PATH = PROJECT_ROOT / "documentation" / "labels.json"

# ==========================================================
# Load Labels
# ==========================================================

with open(LABELS_PATH, "r", encoding="utf-8") as file:
    labels = json.load(file)

NUM_CLASSES = len(labels)

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

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print(f"Training Batches   : {len(train_loader)}")
print(f"Validation Batches : {len(validation_loader)}")

# ==========================================================
# Model
# ==========================================================

model = create_model(NUM_CLASSES)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)


print("\nModel created successfully.")

print(model.__class__.__name__)


# ==========================================================
# Training Loop
# ==========================================================

best_accuracy = 0.0

for epoch in range(EPOCHS):

    training_loss = train_one_epoch(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
    )

    validation_loss, validation_accuracy = validate(
        model=model,
        dataloader=validation_loader,
        criterion=criterion,
        device=DEVICE,
    )


    if validation_accuracy > best_accuracy:

        best_accuracy = validation_accuracy

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            validation_accuracy=validation_accuracy,
        )

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    print(f"Training Loss      : {training_loss:.4f}")

    print(f"Validation Loss    : {validation_loss:.4f}")

    print(f"Validation Accuracy: {validation_accuracy:.2f}%")

    print(f"Best Accuracy      : {best_accuracy:.2f}%")

