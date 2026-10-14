import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import torch

from ml.datasets import (
    create_hospital_dataloaders,
    create_test_dataloader
)
from ml.models.vit.model import create_vit
from ml.training.trainer import (
    evaluate,
    set_random_seed
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "federated"
    / "fedavg_vit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


HOSPITAL_IDS = [
    "hospital_1",
    "hospital_2",
    "hospital_3",
    "hospital_4",
    "hospital_5"
]


ROUNDS = 1
LOCAL_EPOCHS = 1
LEARNING_RATE = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


CLASS_NAMES = [
    "COVID-19",
    "NORMAL",
    "PNEUMONIA"
]


def get_trainable_state(model):

    state = {}

    for name, parameter in model.named_parameters():

        if parameter.requires_grad:

            state[name] = (
                parameter
                .detach()
                .cpu()
                .clone()
            )

    return state


def load_trainable_state(
    model,
    state
):

    with torch.no_grad():

        for name, parameter in model.named_parameters():

            if parameter.requires_grad:

                parameter.copy_(
                    state[name].to(
                        parameter.device
                    )
                )


def fedavg(
    states,
    weights
):

    result = {}

    total_weight = sum(weights)

    for name in states[0]:

        result[name] = torch.zeros_like(
            states[0][name]
        )

        for state, weight in zip(
            states,
            weights
        ):

            result[name] += (
                state[name]
                * weight
                / total_weight
            )

    return result


def train_local_model(
    global_state,
    train_loader
):

    model = create_vit().to(DEVICE)

    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.heads.head.parameters():
        parameter.requires_grad = True

    load_trainable_state(
        model,
        global_state
    )

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=LEARNING_RATE
    )

    criterion = torch.nn.CrossEntropyLoss()

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for epoch in range(LOCAL_EPOCHS):

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            total_correct += (
                outputs.argmax(dim=1)
                == labels
            ).sum().item()

            total_samples += labels.size(0)

        accuracy = (
            total_correct
            / total_samples
        )

        average_loss = (
            total_loss
            / total_samples
        )

        print(
            f"    Epoch {epoch + 1}/{LOCAL_EPOCHS} "
            f"loss={average_loss:.4f} "
            f"acc={accuracy:.4f}"
        )

    return (
        get_trainable_state(model),
        total_samples
    )


def main():

    set_random_seed(42)

    print(
        f"Device: {DEVICE}"
    )

    print(
        "Starting ViT FedAvg"
    )

    print(
        f"Rounds: {ROUNDS}"
    )

    print(
        f"Local epochs: {LOCAL_EPOCHS}"
    )

    hospital_loaders = {}

    print()
    print(
        "Loading hospital datasets..."
    )

    for hospital_id in HOSPITAL_IDS:

        train_loader, validation_loader, class_names = (
            create_hospital_dataloaders(
                DATA_ROOT / hospital_id
            )
        )

        hospital_loaders[hospital_id] = (
            train_loader
        )

        print(
            f"{hospital_id}: "
            f"{len(train_loader.dataset)} "
            f"training samples"
        )

    test_loader, test_class_names = (
        create_test_dataloader(
            DATA_ROOT / "global_test"
        )
    )

    print()

    print(
        f"Global test samples: "
        f"{len(test_loader.dataset)}"
    )

    global_model = create_vit()

    for parameter in global_model.parameters():
        parameter.requires_grad = False

    for parameter in global_model.heads.head.parameters():
        parameter.requires_grad = True

    global_state = get_trainable_state(
        global_model
    )

    metrics = []

    for round_number in range(
        1,
        ROUNDS + 1
    ):

        print()
        print("=" * 60)

        print(
            f"FEDERATED ROUND "
            f"{round_number}/{ROUNDS}"
        )

        print("=" * 60)

        client_states = []
        client_weights = []

        for hospital_id in HOSPITAL_IDS:

            print()
            print(
                f"[{hospital_id}]"
            )

            state, samples = (
                train_local_model(
                    global_state,
                    hospital_loaders[
                        hospital_id
                    ]
                )
            )

            client_states.append(
                state
            )

            client_weights.append(
                samples
            )

            print(
                f"    Samples: {samples}"
            )

        print()
        print(
            "Aggregating hospital models..."
        )

        global_state = fedavg(
            client_states,
            client_weights
        )

        evaluation_model = (
            create_vit().to(DEVICE)
        )

        for parameter in evaluation_model.parameters():
            parameter.requires_grad = False

        for parameter in evaluation_model.heads.head.parameters():
            parameter.requires_grad = True

        load_trainable_state(
            evaluation_model,
            global_state
        )

        result = evaluate(
            evaluation_model,
            test_loader,
            DEVICE,
            class_names=test_class_names
        )

        round_metrics = {
            "round": round_number,
            "accuracy": result[
                "accuracy"
            ],
            "precision_weighted": result[
                "precision_weighted"
            ],
            "recall_weighted": result[
                "recall_weighted"
            ],
            "f1_weighted": result[
                "f1_weighted"
            ],
            "confusion_matrix": result[
                "confusion_matrix"
            ]
        }

        metrics.append(
            round_metrics
        )

        print()
        print("=" * 60)

        print(
            "GLOBAL ViT FEDAVG MODEL"
        )

        print("=" * 60)

        print(
            f"Accuracy : "
            f"{result['accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{result['precision_weighted']:.4f}"
        )

        print(
            f"Recall   : "
            f"{result['recall_weighted']:.4f}"
        )

        print(
            f"F1       : "
            f"{result['f1_weighted']:.4f}"
        )

        print()
        print(
            "Confusion Matrix:"
        )

        for row in result[
            "confusion_matrix"
        ]:

            print(row)

    model = create_vit()

    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.heads.head.parameters():
        parameter.requires_grad = True

    load_trainable_state(
        model,
        global_state
    )

    model_path = (
        OUTPUT_DIR
        / "vit_fedavg_global_model.pt"
    )

    metrics_path = (
        OUTPUT_DIR
        / "vit_fedavg_metrics.json"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2
        )

    print()
    print("=" * 60)

    print(
        "ViT FEDAVG COMPLETE"
    )

    print("=" * 60)

    print(
        f"Model saved: {model_path}"
    )

    print(
        f"Metrics saved: {metrics_path}"
    )


if __name__ == "__main__":
    main()

       