from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from backend.app.services.vit_inference import (
    load_model,
    preprocess_image,
    CLASS_NAMES,
    DEVICE,
)

from ml.xai.vit_explain import generate_vit_attention_map


PROJECT_ROOT = Path(__file__).resolve().parents[3]

XAI_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "xai"
)

XAI_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def generate_explanation(image: Image.Image):
    model = load_model()

    image_tensor = preprocess_image(image)

    attention, target_class, confidence, probabilities = (
        generate_vit_attention_map(
            model,
            image_tensor,
        )
    )

    predicted_class = CLASS_NAMES[target_class]

    output_path = (
        XAI_DIR
        / "api_explanation.png"
    )

    display_image = image.convert("L")

    plt.figure(figsize=(8, 8))

    plt.imshow(
        display_image,
        cmap="gray",
    )

    plt.imshow(
        attention.numpy(),
        cmap="jet",
        alpha=0.45,
    )

    plt.title(
        f"FedXAI-Med ViT XAI\n"
        f"{predicted_class} "
        f"({confidence:.2%})"
    )

    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    probability_dict = {
        class_name: float(probability)
        for class_name, probability in zip(
            CLASS_NAMES,
            probabilities.tolist(),
        )
    }

    return {
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "probabilities": probability_dict,
        "heatmap_path": str(output_path),
        "disclaimer": (
            "The attention visualization is a model "
            "attribution and must not be treated as "
            "proof of disease."
        ),
    }