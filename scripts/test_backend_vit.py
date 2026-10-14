from pathlib import Path

from PIL import Image

from backend.app.services.vit_inference import predict_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "global_test"
    / "COVID-19"
    / "COVID_1005.png"
)


def main():
    image = Image.open(IMAGE_PATH)

    result = predict_image(image)

    print("=== BACKEND ViT TEST ===")
    print(f"Model: {result['model_name']}")
    print(f"Device: {result['device']}")
    print(f"Prediction: {result['predicted_class']}")
    print(
        f"Confidence: "
        f"{result['model_probability']:.4f}"
    )

    print()
    print("Probabilities:")

    for class_name, probability in result[
        "probabilities"
    ].items():
        print(
            f"  {class_name}: "
            f"{probability:.4f}"
        )


if __name__ == "__main__":
    main()