import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)


def create_model(num_classes):
    """
    Create a MobileNetV3-Small model for transfer learning.

    Args:
        num_classes (int):
            Number of output disease classes.

    Returns:
        nn.Module
    """

    # ---------------------------------
    # Load pretrained model
    # ---------------------------------

    model = mobilenet_v3_small(
        weights=MobileNet_V3_Small_Weights.DEFAULT
    )

    # ---------------------------------
    # Freeze feature extractor
    # ---------------------------------

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # ---------------------------------
    # Replace classifier
    # ---------------------------------

    input_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        input_features,
        num_classes
    )

    return model
