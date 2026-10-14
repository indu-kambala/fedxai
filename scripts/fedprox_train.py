from pathlib import Path
import copy
import json
import time

import torch
from torch.optim import Adam

from ml.datasets import (
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
MU = 0.01

OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts"
    / "federated"
    / "fedprox"
)


def fedprox_loss(
    model,
    global_state,
    classification_loss,
):
    proximal_term = torch.tensor(
        0.0,
        device=next(model.parameters()).device,
    )

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        global_parameter = global_state[
            name
        ].to(parameter.device)

        proximal_term += torch.sum(
            (parameter - global_parameter) ** 2
        )

    return (
        classification_loss
        + (MU / 2.0) * proximal_term
    )


def train_one_epoch_fedprox(
    model,
    loader,
    optimizer,
    device,
    global_state,
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0
    batches = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        classification_loss = (
            torch.nn.functional.cross_entropy(
                outputs,
                labels,
            )
        )

        loss = fedprox_loss(
            model,
            global_state,
            classification_loss,
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() * labels.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        batches += 1

    average_loss = (
        total_loss / total
        if total > 0
        else 0.0
    )

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    return (
        average_loss,
        accuracy,
        batches,
        total,
    )


def average_weights(
    state_dicts,
    sample_counts,
):
    total_samples = sum(
        sample_counts
    )

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
            len(state_dicts),
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

    set_random_seed(
        RANDOM_SEED
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 70)
    print("FedXAI-Med — FedProx Training")
    print("=" * 70)

    print(f"Device: {device}")
    print(
        f"Hospitals: "
        f"{len(HOSPITAL_IDS)}"
    )
    print(
        f"Federated rounds: "
        f"{ROUNDS}"
    )
    print(
        f"Local epochs: "
        f"{LOCAL_EPOCHS}"
    )
    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )
    print(
        f"FedProx mu: {MU}"
    )

    hospital_data = {}

    for hospital_id in HOSPITAL_IDS:

        path = DATA_ROOT / hospital_id

        train_loader, _, classes = (
            create_hospital_dataloaders(
                path
            )
        )

        sample_count = len(
            train_loader.dataset
        )

        hospital_data[hospital_id] = {
            "train_loader":
                train_loader,
            "sample_count":
                sample_count,
            "classes":
                classes,
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
        ROUNDS + 1,
    ):

        print()
        print("-" * 70)
        print(
            f"FEDPROX ROUND "
            f"{round_number}/{ROUNDS}"
        )
        print("-" * 70)

        global_state = copy.deepcopy(
            global_model.state_dict()
        )

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
                global_state
            )

            local_model = local_model.to(
                device
            )

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
                    loss,
                    accuracy,
                    batches,
                    samples,
                ) = train_one_epoch_fedprox(
                    local_model,
                    loader,
                    optimizer,
                    device,
                    global_state,
                )

                print(
                    f"Epoch "
                    f"{local_epoch + 1}/"
                    f"{LOCAL_EPOCHS} | "
                    f"Loss={loss:.4f} | "
                    f"Accuracy="
                    f"{accuracy:.4f}"
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

        metrics = evaluate(
            global_model,
            global_test_loader,
            device,
            class_names=list(classes),
        )

        record = {
            "round":
                round_number,
            "accuracy":
                metrics["accuracy"],
            "precision":
                metrics[
                    "precision_weighted"
                ],
            "recall":
                metrics[
                    "recall_weighted"
                ],
            "f1":
                metrics[
                    "f1_weighted"
                ],
            "confusion_matrix":
                metrics[
                    "confusion_matrix"
                ],
            "per_class":
                metrics[
                    "per_class"
                ],
        }

        history.append(record)

        print()
        print(
            f"Global Accuracy: "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Global Precision: "
            f"{metrics['precision_weighted']:.4f}"
        )

        print(
            f"Global Recall: "
            f"{metrics['recall_weighted']:.4f}"
        )

        print(
            f"Global F1: "
            f"{metrics['f1_weighted']:.4f}"
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
        / "fedprox_global_model.pt"
    )

    torch.save(
        {
            "model_name":
                "densenet121",
            "algorithm":
                "FedProx",
            "rounds":
                ROUNDS,
            "local_epochs":
                LOCAL_EPOCHS,
            "learning_rate":
                LEARNING_RATE,
            "mu":
                MU,
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
        / "fedprox_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            {
                "algorithm":
                    "FedProx",
                "rounds":
                    ROUNDS,
                "local_epochs":
                    LOCAL_EPOCHS,
                "learning_rate":
                    LEARNING_RATE,
                "mu":
                    MU,
                "history":
                    history,
                "training_time_seconds":
                    elapsed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("FEDPROX TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )

    print(
        f"Metrics: "
        f"{metrics_path}"
    )


if __name__ == "__main__":
    main()