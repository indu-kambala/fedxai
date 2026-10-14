import torch.nn as nn
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
)

NUM_CLASSES = 3


def create_efficientnet():
    weights = EfficientNet_B0_Weights.DEFAULT

    model = efficientnet_b0(
        weights=weights
    )

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    input_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        input_features,
        NUM_CLASSES
    )

    return model