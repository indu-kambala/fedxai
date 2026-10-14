from pathlib import Path
import hashlib
from collections import defaultdict

DATASET_ROOT = Path("dataset_source")

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


def main():
    hashes = defaultdict(list)

    print()
    print("Scanning original dataset for duplicates...")
    print()

    for file in DATASET_ROOT.rglob("*"):
        if not file.is_file():
            continue

        if file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        file_hash = calculate_hash(file)
        hashes[file_hash].append(file)

    duplicate_groups = []

    for file_hash, files in hashes.items():
        if len(files) > 1:
            duplicate_groups.append(files)

    total_duplicate_files = sum(
        len(files) - 1
        for files in duplicate_groups
    )

    print(f"Unique duplicate groups: {len(duplicate_groups)}")
    print(f"Duplicate files: {total_duplicate_files}")
    print()

    for index, files in enumerate(duplicate_groups, start=1):
        print(f"Duplicate group {index}:")

        for file in files:
            print(f"  {file}")

        print()

    print("Duplicate scan completed.")


if __name__ == "__main__":
    main()