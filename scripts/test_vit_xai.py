import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

import torch

from ml.models.vit.model import create_vit
from ml.xai.vit_explain import generate_vit_attention_map


model = create_vit()

image = torch.randn(
    1,
    3,
    224,
    224
)

attention, target_class, probability, probabilities = (
    generate_vit_attention_map(
        model,
        image
    )
)

print("XAI test successful")
print("Attention shape:", attention.shape)
print("Predicted class:", target_class)
print("Confidence:", probability)
print("Class probabilities:", probabilities)