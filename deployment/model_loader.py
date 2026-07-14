import json

import torch

from training.models.mobilenetv3 import create_model
from deployment.config import (
    CHECKPOINT_PATH,
    LABELS_PATH,
)


def load_model():
    """
    Load the trained MobileNetV3 model.
    Returns the model in evaluation mode.
    """

    # -------------------------------
    # Load labels
    # -------------------------------

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        labels = json.load(file)

    num_classes = len(labels)

    # -------------------------------
    # Create model
    # -------------------------------

    model = create_model(num_classes)

    # -------------------------------
    # Load checkpoint
    # -------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # -------------------------------
    # Evaluation mode
    # -------------------------------

    model.eval()

    return model, labels
