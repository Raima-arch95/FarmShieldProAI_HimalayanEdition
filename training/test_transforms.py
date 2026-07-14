from pathlib import Path

from datasets.farmshield_dataset import FarmShieldDataset
from utils.transforms import get_training_transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent

dataset = FarmShieldDataset(
    dataset_path=PROJECT_ROOT / "datasets" / "FarmShieldDataset",
    labels_path=PROJECT_ROOT / "documentation" / "labels.json",
    transform=get_training_transforms()
)

sample = dataset[0]

print(type(sample["image"]))
print(sample["image"].shape)
print(sample["label"])
