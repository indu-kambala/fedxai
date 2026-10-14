from pathlib import Path
from collections import defaultdict
import hashlib
import random
import shutil

SOURCE_ROOT = Path("dataset_source")
DATA_ROOT = Path("data")

HOSPITALS = [
    "hospital_1",
    "hospital_2",
    "hospital_3",
    "hospital_4",
    "hospital_5"
]

CLASS_MAPPING = {
    "COVID": "COVID-19",
    "NORMAL": "NORMAL",
    "PNEUMONIA": "PNEUMONIA"
}

TEST_RATIO = 0.20
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg"
}


def calculate_hash(path):
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


def get_images(folder):
    return [
        file
        for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]


def remove_generated_dataset():
    for hospital in HOSPITALS:
        hospital_path = DATA_ROOT / hospital

        for class_name in CLASS_MAPPING.values():
            class_path = hospital_path / class_name

            if class_path.exists():
                shutil.rmtree(class_path)

    global_test = DATA_ROOT / "global_test"

    if global_test.exists():
        shutil.rmtree(global_test)


def create_folders():
    for hospital in HOSPITALS:
        for class_name in CLASS_MAPPING.values():
            (
                DATA_ROOT
                / hospital
                / class_name
            ).mkdir(
                parents=True,
                exist_ok=True
            )

    for class_name in CLASS_MAPPING.values():
        (
            DATA_ROOT
            / "global_test"
            / class_name
        ).mkdir(
            parents=True,
            exist_ok=True
        )


def remove_duplicates(images):
    hash_groups = defaultdict(list)

    for image in images:
        image_hash = calculate_hash(image)
        hash_groups[image_hash].append(image)

    unique_images = []

    for files in hash_groups.values():
        unique_images.append(files[0])

    duplicate_files = len(images) - len(unique_images)

    return unique_images, duplicate_files


def split_images(images):
    random.shuffle(images)

    test_count = int(len(images) * TEST_RATIO)

    test_images = images[:test_count]
    train_images = images[test_count:]

    return train_images, test_images


def copy_test_images(images, class_name):
    for image in images:
        destination = (
            DATA_ROOT
            / "global_test"
            / class_name
            / image.name
        )

        shutil.copy2(image, destination)


def distribute_to_hospitals(images, class_name):
    random.shuffle(images)

    for index, image in enumerate(images):
        hospital = HOSPITALS[
            index % len(HOSPITALS)
        ]

        destination = (
            DATA_ROOT
            / hospital
            / class_name
            / image.name
        )

        shutil.copy2(image, destination)


def main():
    random.seed(RANDOM_SEED)

    if not SOURCE_ROOT.exists():
        print("ERROR: dataset_source directory does not exist.")
        return

    print()
    print("Rebuilding clean FedXAI-Med dataset...")
    print()

    remove_generated_dataset()
    create_folders()

    total_original = 0
    total_unique = 0
    total_duplicates = 0
    total_train = 0
    total_test = 0

    for source_class, target_class in CLASS_MAPPING.items():

        source_folder = SOURCE_ROOT / source_class

        if not source_folder.exists():
            print(
                f"ERROR: Missing source folder: "
                f"{source_folder}"
            )
            return

        images = get_images(source_folder)

        original_count = len(images)

        unique_images, duplicate_count = remove_duplicates(
            images
        )

        train_images, test_images = split_images(
            unique_images
        )

        total_original += original_count
        total_unique += len(unique_images)
        total_duplicates += duplicate_count
        total_train += len(train_images)
        total_test += len(test_images)

        print(f"{source_class}:")
        print(f"  Original:   {original_count}")
        print(f"  Unique:     {len(unique_images)}")
        print(f"  Duplicates: {duplicate_count}")
        print(f"  Training:   {len(train_images)}")
        print(f"  Testing:    {len(test_images)}")
        print()

        copy_test_images(
            test_images,
            target_class
        )

        distribute_to_hospitals(
            train_images,
            target_class
        )

    print("=" * 45)
    print("Dataset preparation completed.")
    print("=" * 45)

    print()
    print(f"Original images:    {total_original}")
    print(f"Unique images:      {total_unique}")
    print(f"Duplicates removed: {total_duplicates}")
    print(f"Training images:    {total_train}")
    print(f"Test images:        {total_test}")
    print()

    print("Hospitals:")
    for hospital in HOSPITALS:
        print(f"  {hospital}")

    print()
    print("Global test set created.")
    print()


if __name__ == "__main__":
    main()