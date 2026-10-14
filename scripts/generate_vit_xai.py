import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from ml.models.vit.model import create_vit
from ml.xai.vit_explain import generate_vit_attention_map


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "xai"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


IMAGE_SIZE = 224

CLASS_NAMES = [
    "COVID-19",
    "NORMAL",
    "PNEUMONIA"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    original = image.copy()

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    image_array = (
        np.array(image)
        .astype(np.float32)
        / 255.0
    )

    mean = np.array(
        [0.485, 0.456, 0.406]
    )

    std = np.array(
        [0.229, 0.224, 0.225]
    )

    image_array = (
        image_array - mean
    ) / std

    tensor = torch.tensor(
        image_array,
        dtype=torch.float32
    )

    tensor = tensor.permute(
        2,
        0,
        1
    )

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)

    return original, tensor


def main():

    print(
        f"Device: {DEVICE}"
    )

    model = create_vit()

    checkpoint_path = (
        PROJECT_ROOT
        / "artifacts"
        / "federated"
        / "fedavg_vit"
        / "vit_fedavg_global_model.pt"
    )

    print(
        f"Loading model: {checkpoint_path}"
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)

    model.eval()

    print(
        "Federated ViT model loaded successfully."
    )

    image_paths = []

    for class_name in CLASS_NAMES:

        class_dir = (
            DATA_ROOT
            / "global_test"
            / class_name
        )

        files = list(
            class_dir.glob("*.png")
        )

        if files:

            image_paths.append(
                (
                    class_name,
                    files[0]
                )
            )

    if not image_paths:

        raise RuntimeError(
            "No test images found."
        )

    for true_class, image_path in image_paths:

        print()

        print(
            f"Processing: {image_path.name}"
        )

        original, image_tensor = (
            load_image(image_path)
        )

        (
            attention,
            predicted_class,
            confidence,
            probabilities
        ) = generate_vit_attention_map(
            model,
            image_tensor
        )

        predicted_name = CLASS_NAMES[
            predicted_class
        ]

        print(
            f"True class: {true_class}"
        )

        print(
            f"Predicted class: {predicted_name}"
        )

        print(
            f"Confidence: "
            f"{confidence:.4f}"
        )

        print(
            "Class probabilities:"
        )

        for index, name in enumerate(
            CLASS_NAMES
        ):

            print(
                f"  {name}: "
                f"{probabilities[index]:.4f}"
            )

        attention_array = (
            attention.numpy()
        )

        figure, axes = plt.subplots(
            1,
            3,
            figsize=(15, 5)
        )

        axes[0].imshow(
            original,
            cmap="gray"
        )

        axes[0].set_title(
            f"Original\nTrue: {true_class}"
        )

        axes[0].axis("off")

        axes[1].imshow(
            original,
            cmap="gray"
        )

        axes[1].imshow(
            attention_array,
            alpha=0.5,
            cmap="jet"
        )

        axes[1].set_title(
            f"ViT Attention\n"
            f"Predicted: {predicted_name}"
        )

        axes[1].axis("off")

        axes[2].imshow(
            attention_array,
            cmap="jet"
        )

        axes[2].set_title(
            "Attention Map"
        )

        axes[2].axis("off")

        output_name = (
            f"{true_class.lower()}"
            "_vit_xai.png"
        )

        output_path = (
            OUTPUT_DIR
            / output_name
        )

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved: {output_path}"
        )


if __name__ == "__main__":
    main()