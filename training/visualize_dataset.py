from pathlib import Path
import random

import matplotlib.pyplot as plt

from datasets.farmshield_dataset import FarmShieldDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent

dataset = FarmShieldDataset(
    dataset_path=PROJECT_ROOT / "datasets" / "FarmShieldDataset",
    labels_path=PROJECT_ROOT / "documentation" / "labels.json"
)

sample = dataset[random.randint(0, len(dataset)-1)]

image = sample["image"]

plt.figure(figsize=(6,6))
plt.imshow(image)
plt.axis("off")

plt.title(
    f"{sample['crop']} | {sample['disease']}\n"
    f"Label: {sample['label']}"
)

plt.show()
