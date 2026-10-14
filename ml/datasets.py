from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


IMAGE_SIZE = 224
BATCH_SIZE = 8
VALIDATION_RATIO = 0.10
RANDOM_SEED = 42

CLASS_NAMES = [
    "COVID-19",
    "NORMAL",
    "PNEUMONIA"
]

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


def get_train_transform():
    return transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.Grayscale(
            num_output_channels=3
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


def get_test_transform():
    return transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.Grayscale(
            num_output_channels=3
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


def stratified_split(dataset):
    targets = torch.tensor(dataset.targets)

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    train_indices = []
    validation_indices = []

    for class_index in range(len(dataset.classes)):
        class_indices = torch.where(
            targets == class_index
        )[0]

        permutation = torch.randperm(
            len(class_indices),
            generator=generator
        )

        class_indices = class_indices[permutation]

        validation_size = max(
            1,
            int(
                len(class_indices)
                * VALIDATION_RATIO
            )
        )

        validation_class_indices = class_indices[
            :validation_size
        ]

        train_class_indices = class_indices[
            validation_size:
        ]

        train_indices.extend(
            train_class_indices.tolist()
        )

        validation_indices.extend(
            validation_class_indices.tolist()
        )

    train_indices = torch.tensor(
        train_indices,
        dtype=torch.long
    )

    validation_indices = torch.tensor(
        validation_indices,
        dtype=torch.long
    )

    train_indices = train_indices[
        torch.randperm(
            len(train_indices),
            generator=generator
        )
    ]

    validation_indices = validation_indices[
        torch.randperm(
            len(validation_indices),
            generator=generator
        )
    ]

    return train_indices, validation_indices


def create_hospital_dataloaders(
    hospital_path,
    batch_size=BATCH_SIZE
):
    hospital_path = Path(hospital_path)

    dataset = datasets.ImageFolder(
        root=hospital_path,
        transform=get_train_transform()
    )

    train_indices, validation_indices = (
        stratified_split(dataset)
    )

    train_dataset = Subset(
        dataset,
        train_indices.tolist()
    )

    validation_dataset = Subset(
        dataset,
        validation_indices.tolist()
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=loader_generator
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return (
        train_loader,
        validation_loader,
        dataset.classes
    )


def create_test_dataloader(
    test_path,
    batch_size=BATCH_SIZE
):
    test_path = Path(test_path)

    dataset = datasets.ImageFolder(
        root=test_path,
        transform=get_test_transform()
    )

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return test_loader, dataset.classes