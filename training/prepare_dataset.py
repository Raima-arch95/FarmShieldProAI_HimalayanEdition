from pathlib import Path
import json
import shutil

# -----------------------------
# Project Paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_PLANTVILLAGE = (
    PROJECT_ROOT
    / "datasets"
    / "plantvillage dataset"
    / "color"
)

RAW_WHEAT = (
    PROJECT_ROOT
    / "datasets"
    / "Wheat_Dataset"
    / "train"
)

OUTPUT_DATASET = (
    PROJECT_ROOT
    / "datasets"
    / "FarmShieldDataset"
)

MAPPING_FILE = (
    PROJECT_ROOT
    / "documentation"
    / "dataset_mapping.json"
)

# -----------------------------
# Validation
# -----------------------------

if not RAW_PLANTVILLAGE.exists():
    raise FileNotFoundError(
        f"PlantVillage dataset not found:\n{RAW_PLANTVILLAGE}"
    )

if not RAW_WHEAT.exists():
    raise FileNotFoundError(
        f"Wheat dataset not found:\n{RAW_WHEAT}"
    )

if not OUTPUT_DATASET.exists():
    raise FileNotFoundError(
        f"Output dataset not found:\n{OUTPUT_DATASET}"
    )

if not MAPPING_FILE.exists():
    raise FileNotFoundError(
        f"Mapping file not found:\n{MAPPING_FILE}"
    )

# -----------------------------
# Load Mapping
# -----------------------------

with open(MAPPING_FILE, "r", encoding="utf-8") as file:
    mapping = json.load(file)

print(f"Loaded {len(mapping)} mapping entries.\n")

# -----------------------------
# Prepare Dataset
# -----------------------------

copied_images = 0

PLANTVILLAGE_PREFIXES = (
    "Apple___",
    "Potato___",
    "Tomato___",
    "Corn_(maize)",
    "Pepper,_bell"
)

for source_folder_name, target in mapping.items():

    # Decide which raw dataset this class belongs to

    if source_folder_name.startswith(PLANTVILLAGE_PREFIXES):
        source_folder = RAW_PLANTVILLAGE / source_folder_name
    else:
        source_folder = RAW_WHEAT / source_folder_name

    if not source_folder.exists():
        print(f"Missing source folder: {source_folder_name}")
        continue

    crop = target["crop"]
    disease = target["disease"]

    destination = OUTPUT_DATASET / crop / disease

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    print(source_folder_name)
    print(f"  -> {crop}/{disease}")

    for image in source_folder.iterdir():

        if image.is_file():

            shutil.copy2(
                image,
                destination / image.name
            )

            copied_images += 1

print("\n----------------------------------")
print("Dataset preparation complete.")
print(f"Images copied: {copied_images}")
print("----------------------------------")