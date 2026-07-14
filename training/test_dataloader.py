from pathlib import Path

from torch.utils.data import DataLoader

from datasets.farmshield_dataset import FarmShieldDataset
from utils.transforms import get_training_transforms

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==========================================================
# Dataset
# ==========================================================

dataset = FarmShieldDataset(
    dataset_path=PROJECT_ROOT / "datasets" / "FarmShieldDataset",
    labels_path=PROJECT_ROOT / "documentation" / "labels.json",
    transform=get_training_transforms()
)

# ==========================================================
# DataLoader
# ==========================================================

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

# ==========================================================
# Test One Batch
# ==========================================================

batch = next(iter(dataloader))

images = batch["image"]
labels = batch["label"]

print("=" * 60)
print("DataLoader Test")
print("=" * 60)

print("Images Shape :", images.shape)
print("Labels Shape :", labels.shape)

print("\nFirst Batch Labels:")

print(labels)
