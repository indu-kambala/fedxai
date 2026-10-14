from pathlib import Path
from collections import Counter

from ml.datasets import create_hospital_dataloaders


DATA_ROOT = Path("data")

HOSPITALS = [
    "hospital_1",
    "hospital_2",
    "hospital_3",
    "hospital_4",
    "hospital_5"
]


def get_labels(dataset):
    return [
        dataset.dataset.targets[index]
        for index in dataset.indices
    ]


def main():
    print()
    print("FedXAI-Med Stratified Split Verification")
    print("=" * 55)

    for hospital in HOSPITALS:
        train_loader, validation_loader, classes = (
            create_hospital_dataloaders(
                DATA_ROOT / hospital
            )
        )

        train_labels = get_labels(
            train_loader.dataset
        )

        validation_labels = get_labels(
            validation_loader.dataset
        )

        train_counts = Counter(train_labels)
        validation_counts = Counter(validation_labels)

        print()
        print(hospital)
        print("-" * 30)

        print("Training:")
        for index, class_name in enumerate(classes):
            print(
                f"  {class_name}: "
                f"{train_counts[index]}"
            )

        print("Validation:")
        for index, class_name in enumerate(classes):
            print(
                f"  {class_name}: "
                f"{validation_counts[index]}"
            )

    print()
    print("Split verification completed.")
    print("=" * 55)


if __name__ == "__main__":
    main()