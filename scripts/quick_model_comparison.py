from pathlib import Path
import copy
import json
import time

import torch
from torch.optim import Adam

from ml.datasets import (
    create_hospital_dataloaders,
    create_test_dataloader,
)

from ml.models.efficientnet.model import create_efficientnet
from ml.models.vit.model import create_vit
from ml.models.swin.model import create_swin

from ml.training.trainer import (
    evaluate,
    train_one_epoch,
    set_random_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"

HOSPITAL_IDS = [
    "hospital_1",
    "hospital_2",
    "hospital_3",
    "hospital_4",
    "hospital_5",
]

ROUNDS = 1
LOCAL_EPOCHS = 1
LEARNING_RATE = 1e-4

MODELS = {
    "efficientnet_b0": create_efficientnet,
    "vit_b16": create_vit,
    "swin_t": create_swin,
}


def average_weights(
    states,
    sample_counts,
):
    total = sum(sample_counts)

    averaged = copy.deepcopy(states[0])

    for key in averaged:

        averaged[key] = (
            states[0][key]
            * sample_counts[0]
            / total
        )

        for i in range(1, len(states)):
            averaged[key] += (
                states[i][key]
                * sample_counts[i]
                / total
            )

    return averaged


def run_model(
    model_name,
    create_model,
    device,
    test_loader,
    classes,
):

    print()
    print("=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    global_model = create_model()
    global_model = global_model.to(device)

    local_states = []
    sample_counts = []

    for hospital_id in HOSPITAL_IDS:

        print()
        print(
            f"Local training: {hospital_id}"
        )

        train_loader, _, _ = (
            create_hospital_dataloaders(
                DATA_ROOT / hospital_id
            )
        )

        local_model = create_model()

        local_model.load_state_dict(
            global_model.state_dict()
        )

        local_model = local_model.to(device)

        optimizer = Adam(
            filter(
                lambda parameter:
                parameter.requires_grad,
                local_model.parameters(),
            ),
            lr=LEARNING_RATE,
        )

        for epoch in range(LOCAL_EPOCHS):

            (
                loss,
                accuracy,
                batches,
                samples,
            ) = train_one_epoch(
                local_model,
                train_loader,
                optimizer,
                device,
            )

            print(
                f"Epoch {epoch + 1}/{LOCAL_EPOCHS} | "
                f"Loss={loss:.4f} | "
                f"Accuracy={accuracy:.4f}"
            )

        local_states.append(
            copy.deepcopy(
                local_model.state_dict()
            )
        )

        sample_counts.append(
            len(train_loader.dataset)
        )

    global_state = average_weights(
        local_states,
        sample_counts,
    )

    global_model.load_state_dict(
        global_state
    )

    metrics = evaluate(
        global_model,
        test_loader,
        device,
        class_names=list(classes),
    )

    print()
    print(
        f"{model_name} Global Test Results"
    )

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
        f"F1: "
        f"{metrics['f1_weighted']:.4f}"
    )

    print()
    print("Confusion Matrix:")

    for row in metrics["confusion_matrix"]:
        print(row)

    return {
        "accuracy":
            metrics["accuracy"],
        "precision":
            metrics["precision_weighted"],
        "recall":
            metrics["recall_weighted"],
        "f1":
            metrics["f1_weighted"],
        "confusion_matrix":
            metrics["confusion_matrix"],
        "per_class":
            metrics["per_class"],
    }


def main():

    set_random_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 70)
    print(
        "FedXAI-Med — Quick Architecture Comparison"
    )
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    print(
        f"Federated rounds: {ROUNDS}"
    )

    print(
        f"Local epochs: {LOCAL_EPOCHS}"
    )

    test_loader, classes = (
        create_test_dataloader(
            DATA_ROOT / "global_test"
        )
    )

    results = {}

    started = time.perf_counter()

    for model_name, create_model in MODELS.items():

        results[model_name] = run_model(
            model_name,
            create_model,
            device,
            test_loader,
            classes,
        )

    elapsed = (
        time.perf_counter()
        - started
    )

    output_dir = (
        REPO_ROOT
        / "artifacts"
        / "model_comparison"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "quick_comparison.json"
    )

    output_path.write_text(
        json.dumps(
            {
                "rounds": ROUNDS,
                "local_epochs":
                    LOCAL_EPOCHS,
                "results":
                    results,
                "training_time_seconds":
                    elapsed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print(
        "ARCHITECTURE COMPARISON COMPLETED"
    )
    print("=" * 70)

    print()

    for name, metrics in results.items():

        print(
            f"{name}: "
            f"Accuracy="
            f"{metrics['accuracy']:.4f}, "
            f"F1="
            f"{metrics['f1']:.4f}"
        )

    print()
    print(
        f"Results saved to: "
        f"{output_path}"
    )

    print(
        f"Total time: "
        f"{elapsed:.2f} seconds"
    )


if __name__ == "__main__":
    main()