from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "federated"
    / "fedavg_vit"
    / "vit_fedavg_global_model.pt"
)

CLASS_NAMES = [
    "COVID-19",
    "NORMAL",
    "PNEUMONIA",
]

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


_model = None


def load_model():
    global _model

    if _model is not None:
        return _model

    model = vit_b_16(weights=None)

    model.heads.head = torch.nn.Linear(
        model.heads.head.in_features,
        len(CLASS_NAMES),
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    _model = model

    return _model


def preprocess_image(image: Image.Image):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return transform(image).unsqueeze(0).to(DEVICE)


def predict_image(image: Image.Image):
    model = load_model()

    tensor = preprocess_image(image)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1)

    confidence, predicted_index = torch.max(
        probabilities,
        dim=1,
    )

    predicted_class = CLASS_NAMES[
        predicted_index.item()
    ]

    return {
        "predicted_class": predicted_class,
        "model_probability": float(
            confidence.item()
        ),
        "probabilities": {
            class_name: float(probability)
            for class_name, probability in zip(
                CLASS_NAMES,
                probabilities[0].cpu().tolist(),
            )
        },
        "model_name": "ViT-B/16-FedAvg",
        "device": str(DEVICE),
    }