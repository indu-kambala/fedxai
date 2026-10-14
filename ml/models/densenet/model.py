import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


NUM_CLASSES = 3


def create_densenet(frozen_backbone: bool = True):
    weights = DenseNet121_Weights.DEFAULT

    model = densenet121(
        weights=weights
    )

    if frozen_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    input_features = model.classifier.in_features

    model.classifier = nn.Linear(
        input_features,
        NUM_CLASSES
    )

    return model