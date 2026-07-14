from pathlib import Path

from datasets.farmshield_dataset import FarmShieldDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent

dataset = FarmShieldDataset(
    dataset_path=PROJECT_ROOT / "datasets" / "FarmShieldDataset",
    labels_path=PROJECT_ROOT / "documentation" / "labels.json"
)

print("=" * 60)

print(f"Images : {len(dataset)}")

sample = dataset[0]

print(sample["crop"])

print(sample["disease"])

print(sample["label"])

print(sample["path"])
