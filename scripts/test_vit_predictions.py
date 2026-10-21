
from collections import Counter
from pathlib import Path
import random

from PIL import Image
from torchvision import datasets

from backend.app.services.vit_inference import predict_image

DATASET_PATH = Path("data/global_test")
SAMPLES_PER_CLASS = 10
random.seed(42)

dataset = datasets.ImageFolder(DATASET_PATH)

print("Dataset class order:", dataset.classes)
print()

overall = Counter()

for class_name in dataset.classes:
    samples = [
        path
        for path, label in dataset.samples
        if dataset.classes[label] == class_name
    ]

    selected = random.sample(
        samples,
        min(SAMPLES_PER_CLASS, len(samples)),
    )

    predictions = Counter()

    print(f"ACTUAL CLASS: {class_name}")

    for path in selected:
        with Image.open(path) as image:
            result = predict_image(image.convert("RGB"))

        predicted = result["predicted_class"]
        probability = result["model_probability"]

        predictions[predicted] += 1
        overall[predicted] += 1

        print(
            f"  Predicted: {predicted:12} "
            f"Probability: {probability:.3f} "
            f"File: {Path(path).name}"
        )

    print("Prediction counts:", dict(predictions))
    print()

print("OVERALL PREDICTION COUNTS:", dict(overall))
