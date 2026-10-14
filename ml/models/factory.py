"""Model factory. DenseNet121 is the practical baseline until others are measured."""

from __future__ import annotations

from typing import Callable

import torch.nn as nn

from ml.models.densenet.model import create_densenet
from ml.models.efficientnet.model import create_efficientnet
from ml.models.swin.model import create_swin
from ml.models.vit.model import create_vit


FACTORIES: dict[str, Callable[..., nn.Module]] = {
    "densenet121": create_densenet,
    "efficientnet_b3": create_efficientnet,
    "vit_b16": create_vit,
    "swin_tiny": create_swin,
}

FAMILIES = {
    "densenet121": "cnn",
    "efficientnet_b3": "cnn",
    "vit_b16": "transformer",
    "swin_tiny": "transformer",
}


def create_model(name: str = "densenet121", frozen_backbone: bool = True) -> nn.Module:
    key = name.lower().replace("-", "_")
    aliases = {
        "densenet": "densenet121",
        "efficientnet": "efficientnet_b3",
        "vit": "vit_b16",
        "swin": "swin_tiny",
    }
    key = aliases.get(key, key)
    if key not in FACTORIES:
        raise ValueError(f"Unknown model '{name}'. Known: {sorted(FACTORIES)}")
    factory = FACTORIES[key]
    try:
        return factory(frozen_backbone=frozen_backbone)
    except TypeError:
        return factory()


def model_family(name: str) -> str:
    key = name.lower().replace("-", "_")
    aliases = {
        "densenet": "densenet121",
        "efficientnet": "efficientnet_b3",
        "vit": "vit_b16",
        "swin": "swin_tiny",
    }
    key = aliases.get(key, key)
    return FAMILIES.get(key, "unknown")
