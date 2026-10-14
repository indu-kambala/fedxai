from pathlib import Path
from collections import Counter
import hashlib

from PIL import Image

DATASET_ROOT = Path("data")

EXPECTED_HOSPITALS = [f"hospital_{i}" for i in range(1, 6)]

EXPECTED_CLASSES = [
    "NORMAL",
    "PNEUMONIA",
    "COVID-19"
]

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg"
}

DICOM_EXTENSIONS = {
    ".dcm",
    ".dicom"
}

ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | DICOM_EXTENSIONS


def file_hash(path):
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


def validate_image(path):
    try:
        with Image.open(path) as image:
            image.verify()

        with Image.open(path) as image:
            width, height = image.size

        if width <= 0 or height <= 0:
            return {
                "valid": False,
                "width": width,
                "height": height,
                "error": "Invalid image dimensions."
            }

        return {
            "valid": True,
            "width": width,
            "height": height,
            "error": None
        }

    except Exception as exc:
        return {
            "valid": False,
            "width": None,
            "height": None,
            "error": str(exc)
        }


def validate_dataset():
    hospitals = {}
    errors = []
    warnings = []
    hashes = {}

    total_images = 0
    total_corrupt = 0
    total_duplicates = 0
    total_dicom = 0

    width_values = []
    height_values = []

    if not DATASET_ROOT.exists():
        return {
            "status": "FAILED",
            "errors": ["Dataset directory does not exist."],
            "warnings": [],
            "hospitals": {},
            "total_images": 0,
            "total_corrupt": 0,
            "total_duplicates": 0,
            "total_dicom": 0
        }

    for hospital_name in EXPECTED_HOSPITALS:
        hospital_path = DATASET_ROOT / hospital_name

        if not hospital_path.exists():
            errors.append(
                f"Missing hospital folder: {hospital_name}"
            )
            continue

        class_counts = Counter()
        hospital_images = 0
        hospital_corrupt = 0
        hospital_dicom = 0

        for class_name in EXPECTED_CLASSES:
            class_path = hospital_path / class_name

            if not class_path.exists():
                errors.append(
                    f"Missing class folder: "
                    f"{hospital_name}/{class_name}"
                )
                continue

            files = [
                file
                for file in class_path.rglob("*")
                if file.is_file()
            ]

            if not files:
                warnings.append(
                    f"No files found in "
                    f"{hospital_name}/{class_name}"
                )

            for file in files:
                extension = file.suffix.lower()

                if extension not in ALLOWED_EXTENSIONS:
                    warnings.append(
                        f"Unsupported file: {file}"
                    )
                    continue

                if file.stat().st_size == 0:
                    total_corrupt += 1
                    hospital_corrupt += 1

                    errors.append(
                        f"Empty file: {file}"
                    )
                    continue

                hospital_images += 1
                total_images += 1
                class_counts[class_name] += 1

                try:
                    current_hash = file_hash(file)

                    if current_hash in hashes:
                        total_duplicates += 1

                        warnings.append(
                            f"Duplicate file: {file} "
                            f"matches {hashes[current_hash]}"
                        )
                    else:
                        hashes[current_hash] = str(file)

                except Exception as exc:
                    total_corrupt += 1
                    hospital_corrupt += 1

                    errors.append(
                        f"Could not read {file}: {exc}"
                    )

                    continue

                if extension in IMAGE_EXTENSIONS:
                    result = validate_image(file)

                    if not result["valid"]:
                        total_corrupt += 1
                        hospital_corrupt += 1

                        errors.append(
                            f"Corrupt image: {file} - "
                            f"{result['error']}"
                        )

                    else:
                        width_values.append(result["width"])
                        height_values.append(result["height"])

                elif extension in DICOM_EXTENSIONS:
                    total_dicom += 1
                    hospital_dicom += 1

        hospitals[hospital_name] = {
            "total_files": hospital_images,
            "corrupt_files": hospital_corrupt,
            "dicom_files": hospital_dicom,
            "classes": dict(class_counts)
        }

    if total_corrupt > 0:
        status = "FAILED"

    elif errors:
        status = "FAILED"

    elif total_images == 0:
        status = "FAILED"

        errors.append(
            "No valid dataset files found."
        )

    elif total_duplicates > 0:
        status = "WARNING"

    else:
        status = "READY"

    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "hospitals": hospitals,
        "total_images": total_images,
        "total_corrupt": total_corrupt,
        "total_duplicates": total_duplicates,
        "total_dicom": total_dicom,
        "image_dimensions": {
            "min_width": min(width_values)
            if width_values else None,
            "max_width": max(width_values)
            if width_values else None,
            "min_height": min(height_values)
            if height_values else None,
            "max_height": max(height_values)
            if height_values else None
        }
    }