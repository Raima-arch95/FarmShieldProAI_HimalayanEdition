from pathlib import Path
import torch

IMAGE_SIZE = 224

SAMPLE_INPUT = (
    torch.randn(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ),
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "exports"
    / "farmshield_v1.pth"
)

LABELS_PATH = (
    PROJECT_ROOT
    / "documentation"
    / "labels.json"
)

EXPORT_PATH = (
    PROJECT_ROOT
    / "exports"
    / "farmshield_v1.tflite"
)

IMAGE_SIZE = 224
