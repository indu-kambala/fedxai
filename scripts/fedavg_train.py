from pathlib import Path
import json
import copy
import time

import torch
from torch.optim import Adam

from ml.datasets import (
    BATCH_SIZE,
    RANDOM_SEED,
    create_hospital_dataloaders,
    create_test_dataloader,
)
from ml.models.densenet.model import create_densenet
from ml.training.trainer import (
    evaluate,
    set_random_seed,
    train_one_epoch,
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
LOCAL_EPOCHS = 2
LEARNING_RATE = 1e-4

OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts"
    / "federated"
    / "fedavg"
)


def average_weights(
    state_dicts,
    sample_counts,
):
    total_samples = sum(sample_counts)

    global_state = copy.deepcopy(
        state_dicts[0]
    )

    for key in global_state:

        global_state[key] = (
            state_dicts[0][key]
            * (
                sample_counts[0]
                / total_samples
            )
        )

        for index in range(
            1,
            len(state_dicts)
        ):
            global_state[key] += (
                state_dicts[index][key]
                * (
                    sample_counts[index]
                    / total_samples
                )
            )

    return global_state


def main():

    set_random_seed(RANDOM_SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 70)
    print("FedXAI-Med — FedAvg Training")
    print("=" * 70)

    print(f"Device: {device}")
    print(f"Hospitals: {len(HOSPITAL_IDS)}")
    print(f"Federated rounds: {ROUNDS}")
    print(f"Local epochs: {LOCAL_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")

    hospital_data = {}

    for hospital_id in HOSPITAL_IDS:

        path = DATA_ROOT / hospital_id

        train_loader, validation_loader, classes = (
            create_hospital_dataloaders(path)
        )

        sample_count = len(
            train_loader.dataset
        )

        hospital_data[hospital_id] = {
            "train_loader": train_loader,
            "validation_loader": validation_loader,
            "classes": classes,
            "sample_count": sample_count,
        }

        print(
            f"{hospital_id}: "
            f"{sample_count} training samples"
        )

    global_test_loader, classes = (
        create_test_dataloader(
            DATA_ROOT / "global_test"
        )
    )

    global_model = create_densenet()
    global_model = global_model.to(device)

    history = []

    started = time.perf_counter()

    for round_number in range(
        1,
        ROUNDS + 1
    ):

        print()
        print("-" * 70)
        print(
            f"FEDAVG ROUND "
            f"{round_number}/{ROUNDS}"
        )
        print("-" * 70)

        local_states = []
        sample_counts = []

        for hospital_id in HOSPITAL_IDS:

            print()
            print(
                f"Local training: "
                f"{hospital_id}"
            )

            local_model = create_densenet()
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

            loader = hospital_data[
                hospital_id
            ]["train_loader"]

            for local_epoch in range(
                LOCAL_EPOCHS
            ):

                (
                    train_loss,
                    train_accuracy,
                    batches,
                    samples,
                ) = train_one_epoch(
                    local_model,
                    loader,
                    optimizer,
                    device,
                )

                print(
                    f"Epoch "
                    f"{local_epoch + 1}/"
                    f"{LOCAL_EPOCHS} | "
                    f"Loss={train_loss:.4f} | "
                    f"Accuracy="
                    f"{train_accuracy:.4f}"
                )

            local_states.append(
                copy.deepcopy(
                    local_model.state_dict()
                )
            )

            sample_counts.append(
                hospital_data[
                    hospital_id
                ]["sample_count"]
            )

        global_state = average_weights(
            local_states,
            sample_counts,
        )

        global_model.load_state_dict(
            global_state
        )

        global_metrics = evaluate(
            global_model,
            global_test_loader,
            device,
            class_names=list(classes),
        )

        round_record = {
            "round": round_number,
            "accuracy":
                global_metrics["accuracy"],
            "precision":
                global_metrics[
                    "precision_weighted"
                ],
            "recall":
                global_metrics[
                    "recall_weighted"
                ],
            "f1":
                global_metrics[
                    "f1_weighted"
                ],
            "confusion_matrix":
                global_metrics[
                    "confusion_matrix"
                ],
            "per_class":
                global_metrics[
                    "per_class"
                ],
        }

        history.append(round_record)

        print()
        print(
            f"Global Accuracy: "
            f"{global_metrics['accuracy']:.4f}"
        )

        print(
            f"Global Precision: "
            f"{global_metrics['precision_weighted']:.4f}"
        )

        print(
            f"Global Recall: "
            f"{global_metrics['recall_weighted']:.4f}"
        )

        print(
            f"Global F1: "
            f"{global_metrics['f1_weighted']:.4f}"
        )

    elapsed = (
        time.perf_counter()
        - started
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        OUTPUT_DIR
        / "fedavg_global_model.pt"
    )

    torch.save(
        {
            "model_name":
                "densenet121",
            "algorithm":
                "FedAvg",
            "rounds":
                ROUNDS,
            "local_epochs":
                LOCAL_EPOCHS,
            "learning_rate":
                LEARNING_RATE,
            "class_names":
                list(classes),
            "state_dict":
                global_model.state_dict(),
            "history":
                history,
            "training_time_seconds":
                elapsed,
        },
        checkpoint_path,
    )

    metrics_path = (
        OUTPUT_DIR
        / "fedavg_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            {
                "algorithm": "FedAvg",
                "rounds": ROUNDS,
                "local_epochs":
                    LOCAL_EPOCHS,
                "history": history,
                "training_time_seconds":
                    elapsed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("FEDAVG TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )

    print(
        f"Metrics: "
        f"{metrics_path}"
    )

    print(
        f"Total time: "
        f"{elapsed:.2f} seconds"
    )


if __name__ == "__main__":
    main()