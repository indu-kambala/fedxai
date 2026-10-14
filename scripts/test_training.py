from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

from ml.datasets import create_hospital_dataloaders
from ml.models.densenet.model import create_densenet


DATA_ROOT = Path("data")


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print()
    print("FedXAI-Med Training Smoke Test")
    print("=" * 50)
    print(f"Device: {device}")

    train_loader, validation_loader, classes = (
        create_hospital_dataloaders(
            DATA_ROOT / "hospital_1"
        )
    )

    model = create_densenet()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=1e-4
    )

    images, labels = next(iter(train_loader))

    print(f"Input batch: {images.shape}")
    print(f"Labels: {labels.shape}")

    images = images.to(device)
    labels = labels.to(device)

    model.train()

    optimizer.zero_grad()

    outputs = model(images)

    loss = criterion(
        outputs,
        labels
    )

    loss.backward()

    optimizer.step()

    predictions = outputs.argmax(
        dim=1
    )

    accuracy = (
        predictions == labels
    ).float().mean().item()

    print(f"Output: {outputs.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Batch accuracy: {accuracy:.4f}")

    print()
    print("Training smoke test completed successfully.")
    print("=" * 50)


if __name__ == "__main__":
    main()