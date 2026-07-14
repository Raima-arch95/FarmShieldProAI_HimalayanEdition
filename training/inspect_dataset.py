from pathlib import Path

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "datasets" / "FarmShieldDataset"


# ==========================================================
# Validation
# ==========================================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )


# ==========================================================
# Count Images
# ==========================================================

def count_images(folder: Path):
    """
    Count all image files inside a disease folder.
    """

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    }

    return sum(
        1
        for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in image_extensions
    )


# ==========================================================
# Inspect Dataset
# ==========================================================

def inspect_dataset():

    total_crops = 0
    total_classes = 0
    total_images = 0

    print("=" * 60)
    print("Farm Shield Dataset Inspection")
    print("=" * 60)

    for crop in sorted(DATASET_PATH.iterdir()):

        if not crop.is_dir():
            continue

        total_crops += 1

        print(f"\n{crop.name}")

        for disease in sorted(crop.iterdir()):

            if not disease.is_dir():
                continue

            total_classes += 1

            image_count = count_images(disease)

            total_images += image_count

            print(
                f"   {disease.name:<30} {image_count:>5} images"
            )

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print(f"Crops          : {total_crops}")
    print(f"Disease Classes: {total_classes}")
    print(f"Images         : {total_images}")


# ==========================================================
# Main
# ==========================================================

def main():
    inspect_dataset()


if __name__ == "__main__":
    main()