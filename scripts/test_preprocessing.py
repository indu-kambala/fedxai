from pathlib import Path

import torch

from ml.datasets import (
    create_hospital_dataloaders,
    create_test_dataloader
)


DATA_ROOT = Path("data")

HOSPITALS = [
    "hospital_1",
    "hospital_2",
    "hospital_3",
    "hospital_4",
    "hospital_5"
]


def main():
    print()
    print("Testing FedXAI-Med preprocessing pipeline")
    print("=" * 50)

    for hospital in HOSPITALS:
        hospital_path = DATA_ROOT / hospital

        train_loader, validation_loader, classes = (
            create_hospital_dataloaders(
                hospital_path
            )
        )

        images, labels = next(iter(train_loader))

        print()
        print(hospital)
        print(f"Classes: {classes}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(validation_loader)}")
        print(f"Image tensor shape: {images.shape}")
        print(f"Label tensor shape: {labels.shape}")

    test_loader, test_classes = create_test_dataloader(
        DATA_ROOT / "global_test"
    )

    test_images, test_labels = next(iter(test_loader))

    print()
    print("GLOBAL TEST")
    print(f"Classes: {test_classes}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Image tensor shape: {test_images.shape}")
    print(f"Label tensor shape: {test_labels.shape}")

    print()
    print("Preprocessing pipeline test completed.")
    print("=" * 50)


if __name__ == "__main__":
    main()