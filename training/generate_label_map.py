from pathlib import Path
import csv
import json

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "datasets" / "FarmShieldDataset"

OUTPUT_JSON = PROJECT_ROOT / "documentation" / "labels.json"

OUTPUT_CSV = PROJECT_ROOT / "documentation" / "labels.csv"

# ==========================================================
# Validation
# ==========================================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

# ==========================================================
# Generate Label Map
# ==========================================================

def generate_label_map():
    """
    Generate label mappings from the standardized dataset.
    """

    label = 0
    rows = []

    print("=" * 60)
    print("Generating Label Map")
    print("=" * 60)

    for crop in sorted(DATASET_PATH.iterdir()):

        if not crop.is_dir():
            continue

        for disease in sorted(crop.iterdir()):

            if not disease.is_dir():
                continue

            rows.append(
                [label, crop.name, disease.name]
            )

            print(
                f"{label:>2}  "
                f"{crop.name:<12} "
                f"{disease.name}"
            )

            label += 1

    return rows


# ==========================================================
# Save JSON (Primary Output)
# ==========================================================

def save_json(rows):
    """
    Save the label map as JSON.
    This is the primary file that Flutter will use.
    """

    labels = []

    for label, crop, disease in rows:

        labels.append(
            {
                "label": label,
                "crop": crop,
                "disease": disease
            }
        )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            labels,
            file,
            indent=4
        )


# ==========================================================
# Save CSV (Documentation)
# ==========================================================

def save_csv(rows):
    """
    Save the label map as CSV for documentation.
    """

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "Label",
                "Crop",
                "Disease"
            ]
        )

        writer.writerows(rows)


# ==========================================================
# Summary
# ==========================================================

def print_summary(rows):

    print("\n" + "=" * 60)
    print("Label Map Created")
    print("=" * 60)

    print(f"Total Labels : {len(rows)}")

    print("\nGenerated Files:")

    print(f"JSON : {OUTPUT_JSON}")
    print(f"CSV  : {OUTPUT_CSV}")

    print("=" * 60)


# ==========================================================
# Main
# ==========================================================

def main():

    rows = generate_label_map()

    save_json(rows)

    save_csv(rows)

    print_summary(rows)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()