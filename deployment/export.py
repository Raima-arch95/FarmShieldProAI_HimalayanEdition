import litert_torch

from deployment.config import (
    EXPORT_PATH,
    SAMPLE_INPUT,
)

from deployment.model_loader import load_model


def main():

    print("Loading model...")

    model, labels = load_model()

    print("Converting model to LiteRT...")

    edge_model = litert_torch.convert(
        model,
        SAMPLE_INPUT,
    )

    print("Exporting model...")

    edge_model.export(
        str(EXPORT_PATH)
    )

    print("\nConversion successful!")

    print(f"\nSaved to:\n{EXPORT_PATH}")


if __name__ == "__main__":
    main()
