import torch

from ml.models.densenet.model import create_densenet


def main():
    print()
    print("Testing DenseNet121")
    print("=" * 40)

    model = create_densenet()

    print(model.classifier)

    test_input = torch.randn(
        2,
        3,
        224,
        224
    )

    output = model(test_input)

    print()
    print(f"Input shape:  {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print()
    print("Model test completed.")
    print("=" * 40)


if __name__ == "__main__":
    main()