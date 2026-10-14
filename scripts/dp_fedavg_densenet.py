import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import torch
import torch.nn as nn
from opacus import PrivacyEngine

from ml.datasets import create_hospital_dataloaders, create_test_dataloader
from ml.models.densenet.model import create_densenet
from ml.training.trainer import evaluate, set_random_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "federated"
    / "dp_fedavg_densenet"
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

TARGET_EPSILON = 8.0
TARGET_DELTA = 1e-5
MAX_GRAD_NORM = 1.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def get_trainable_state(model):

    if hasattr(model, "_module"):
        model = model._module

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


def load_trainable_state(model, state):

    if hasattr(model, "_module"):
        model = model._module

    with torch.no_grad():

        for name, parameter in model.named_parameters():

            if parameter.requires_grad:

                if name in state:

                    parameter.copy_(
                        state[name].to(
                            parameter.device
                        )
                    )


def fedavg(states, weights):

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
                * (weight / total_weight)
            )

    return result


def train_private_client(
    global_state,
    train_loader,
    epochs
):

    model = create_densenet().to(DEVICE)

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    for parameter in model.classifier.parameters():
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

    privacy_engine = PrivacyEngine()

    model, optimizer, private_loader = (
        privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=TARGET_EPSILON,
            target_delta=TARGET_DELTA,
            epochs=epochs,
            max_grad_norm=MAX_GRAD_NORM
        )
    )

    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in private_loader:

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

        average_loss = (
            total_loss
            / total_samples
        )

        accuracy = (
            total_correct
            / total_samples
        )

        print(
            f"    Epoch {epoch + 1}/{epochs} "
            f"loss={average_loss:.4f} "
            f"acc={accuracy:.4f}"
        )

    epsilon = privacy_engine.get_epsilon(
        TARGET_DELTA
    )

    if hasattr(model, "_module"):
        base_model = model._module
    else:
        base_model = model

    trainable_state = get_trainable_state(
        base_model
    )

    return (
        trainable_state,
        total_samples,
        epsilon
    )


def main():

    set_random_seed(42)

    print(f"Device: {DEVICE}")
    print("Starting DenseNet DP-FedAvg")
    print(f"Epsilon target: {TARGET_EPSILON}")
    print(f"Delta: {TARGET_DELTA}")
    print(f"Rounds: {ROUNDS}")
    print(f"Local epochs: {LOCAL_EPOCHS}")

    hospital_loaders = {}

    print()
    print("Loading hospital datasets...")

    for hospital_id in HOSPITAL_IDS:

        train_loader, validation_loader, class_names = (
            create_hospital_dataloaders(
                DATA_ROOT / hospital_id
            )
        )

        hospital_loaders[hospital_id] = train_loader

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

    initial_model = create_densenet()

    for parameter in initial_model.features.parameters():
        parameter.requires_grad = False

    for parameter in initial_model.classifier.parameters():
        parameter.requires_grad = True

    global_state = get_trainable_state(
        initial_model
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
        client_epsilons = []

        for hospital_id in HOSPITAL_IDS:

            print()
            print(f"[{hospital_id}]")

            state, samples, epsilon = (
                train_private_client(
                    global_state,
                    hospital_loaders[hospital_id],
                    LOCAL_EPOCHS
                )
            )

            client_states.append(state)
            client_weights.append(samples)
            client_epsilons.append(epsilon)

            print(
                f"    Samples: {samples}"
            )

            print(
                f"    Privacy epsilon: "
                f"{epsilon:.4f}"
            )

        print()
        print("Aggregating hospital models...")

        global_state = fedavg(
            client_states,
            client_weights
        )

        evaluation_model = (
            create_densenet().to(DEVICE)
        )

        for parameter in evaluation_model.features.parameters():
            parameter.requires_grad = False

        for parameter in evaluation_model.classifier.parameters():
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

        final_epsilon = max(
            client_epsilons
        )

        round_metrics = {
            "round": round_number,
            "accuracy": result["accuracy"],
            "precision_weighted": result[
                "precision_weighted"
            ],
            "recall_weighted": result[
                "recall_weighted"
            ],
            "f1_weighted": result[
                "f1_weighted"
            ],
            "epsilon": final_epsilon,
            "delta": TARGET_DELTA,
            "confusion_matrix": result[
                "confusion_matrix"
            ]
        }

        metrics.append(
            round_metrics
        )

        print()
        print("=" * 60)
        print("GLOBAL DENSENET DP MODEL")
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

        print(
            f"Epsilon  : "
            f"{final_epsilon:.4f}"
        )

        print(
            f"Delta    : "
            f"{TARGET_DELTA}"
        )

        print()
        print("Confusion Matrix:")

        for row in result["confusion_matrix"]:
            print(row)

    model_path = (
        OUTPUT_DIR
        / "dp_fedavg_densenet_global_model.pt"
    )

    metrics_path = (
        OUTPUT_DIR
        / "dp_fedavg_densenet_metrics.json"
    )

    torch.save(
        global_state,
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
    print("DENSENET DP-FedAvg COMPLETE")
    print("=" * 60)

    print(
        f"Model saved: {model_path}"
    )

    print(
        f"Metrics saved: {metrics_path}"
    )


if __name__ == "__main__":
    main()