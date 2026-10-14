import torch.nn as nn
from torchvision.models import (
    swin_t,
    Swin_T_Weights,
)

NUM_CLASSES = 3


def create_swin():
    weights = Swin_T_Weights.DEFAULT

    model = swin_t(
        weights=weights
    )

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    input_features = model.head.in_features

    model.head = nn.Linear(
        input_features,
        NUM_CLASSES
    )

    return model