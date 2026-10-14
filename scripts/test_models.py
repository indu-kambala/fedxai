import torch

from ml.models.efficientnet.model import create_efficientnet
from ml.models.vit.model import create_vit
from ml.models.swin.model import create_swin


def test_model(name, create_model):

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    model = create_model()
    model.eval()

    images = torch.randn(
        2,
        3,
        224,
        224
    )

    with torch.no_grad():
        output = model(images)

    print(
        f"Output shape: {output.shape}"
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen = sum(
        parameter.numel()
        for parameter in model.parameters()
        if not parameter.requires_grad
    )

    print(
        f"Trainable parameters: {trainable}"
    )

    print(
        f"Frozen parameters: {frozen}"
    )


def main():

    test_model(
        "EfficientNet-B0",
        create_efficientnet
    )

    test_model(
        "ViT-B/16",
        create_vit
    )

    test_model(
        "Swin-T",
        create_swin
    )


if __name__ == "__main__":
    main()