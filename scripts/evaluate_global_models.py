from pathlib import Path
import json

import torch

from ml.datasets import create_test_dataloader
from ml.models.densenet.model import create_densenet
from ml.training.trainer import evaluate


REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = REPO_ROOT / "data"

EXPERIMENTS_ROOT = (
    REPO_ROOT / "artifacts" / "experiments"
)

HOSPITAL_IDS = [
    "hospital_1",
    "hospital_2",
    "hospital_3",
    "hospital_4",
    "hospital_5",
]


def load_model(checkpoint_path, device):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = create_densenet()

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model = model.to(device)

    model.eval()

    return model, checkpoint


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "FedXAI-Med — Global Test Evaluation"
    )
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    global_test_path = (
        DATA_ROOT / "global_test"
    )

    if not global_test_path.exists():
        raise FileNotFoundError(
            f"Global test directory not found: "
            f"{global_test_path}"
        )

    global_test_loader, classes = (
    create_test_dataloader(
        global_test_path
        )
    )

    print(
        f"Global test batches: "
        f"{len(global_test_loader)}"
    )

    print(
        f"Classes: {list(classes)}"
    )

    results = {}

    for hospital_id in HOSPITAL_IDS:

        print()
        print("-" * 70)
        print(
            f"Evaluating {hospital_id}"
        )
        print("-" * 70)

        checkpoint_path = (
            EXPERIMENTS_ROOT
            / f"densenet121_{hospital_id}_frozen"
            / "checkpoint.pt"
        )

        if not checkpoint_path.exists():
            print(
                f"Checkpoint not found: "
                f"{checkpoint_path}"
            )
            continue

        model, checkpoint = load_model(
            checkpoint_path,
            device,
        )

        metrics = evaluate(
            model,
            global_test_loader,
            device,
            class_names=list(classes),
        )

        results[hospital_id] = metrics

        print(
            f"Accuracy: "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{metrics['precision_weighted']:.4f}"
        )

        print(
            f"Recall: "
            f"{metrics['recall_weighted']:.4f}"
        )

        print(
            f"F1-score: "
            f"{metrics['f1_weighted']:.4f}"
        )

        print()
        print("Confusion Matrix:")

        for row in metrics[
            "confusion_matrix"
        ]:
            print(row)

        print()
        print("Per-class metrics:")

        for class_name, class_metrics in (
            metrics["per_class"].items()
        ):
            print(
                f"{class_name}: "
                f"Precision="
                f"{class_metrics['precision']:.4f}, "
                f"Recall="
                f"{class_metrics['recall']:.4f}, "
                f"F1="
                f"{class_metrics['f1_score']:.4f}, "
                f"Support="
                f"{class_metrics['support']}"
            )

    output_dir = (
        REPO_ROOT
        / "artifacts"
        / "global_evaluation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "densenet121_hospital_models.json"
    )

    output_path.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("GLOBAL EVALUATION COMPLETED")
    print("=" * 70)

    print(
        f"Results saved to:"
    )

    print(output_path)


if __name__ == "__main__":
    main()