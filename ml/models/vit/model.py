import torch.nn as nn
from torchvision.models import (
    vit_b_16,
    ViT_B_16_Weights,
)

NUM_CLASSES = 3


def create_vit():
    weights = ViT_B_16_Weights.DEFAULT

    model = vit_b_16(
        weights=weights
    )

    for parameter in model.parameters():
        parameter.requires_grad = False

    input_features = model.heads.head.in_features

    model.heads.head = nn.Linear(
        input_features,
        NUM_CLASSES
    )

    return model