from pathlib import Path
import json
import time

import torch
from torch.optim import Adam

from ml.datasets import (
    BATCH_SIZE,
    RANDOM_SEED,
    VALIDATION_RATIO,
    create_hospital_dataloaders,
)
from ml.models.densenet.model import NUM_CLASSES, create_densenet
from ml.training.trainer import (
    count_trainable_parameters,
    evaluate,
    save_checkpoint,
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

MODEL_NAME = "densenet121"
DATASET_VERSION = "hospital_partitions_unique_4999"

EPOCHS = 20
MAX_TRAIN_BATCHES = None
LEARNING_RATE = 1e-4


def assert_frozen_backbone(model):
    for parameter in model.features.parameters():
        if parameter.requires_grad:
            raise RuntimeError(
                "DenseNet backbone is not frozen."
            )

    if not model.classifier.weight.requires_grad:
        raise RuntimeError(
            "DenseNet classifier is not trainable."
        )


def train_hospital(hospital_id, device):
    hospital_path = DATA_ROOT / hospital_id

    if not hospital_path.exists():
        raise FileNotFoundError(
            f"Hospital dataset directory not found: "
            f"{hospital_path}"
        )

    experiment_dir = (
        REPO_ROOT
        / "artifacts"
        / "experiments"
        / f"densenet121_{hospital_id}_frozen"
    )

    checkpoint_path = experiment_dir / "checkpoint.pt"
    metrics_path = experiment_dir / "metrics.json"

    print()
    print("=" * 70)
    print(f"Training Hospital: {hospital_id}")
    print("=" * 70)

    started = time.perf_counter()

    set_random_seed(RANDOM_SEED)

    train_loader, validation_loader, classes = (
        create_hospital_dataloaders(
            hospital_path
        )
    )

    print(f"Hospital: {hospital_id}")
    print(f"Model: {MODEL_NAME}")
    print("Backbone: frozen")
    print(f"Epochs: {EPOCHS}")
    print(
        f"Max train batches: "
        f"{MAX_TRAIN_BATCHES}"
    )
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Seed: {RANDOM_SEED}")

    print(
        f"Classes: {list(classes)}"
    )

    print(
        f"Train loader batches available: "
        f"{len(train_loader)}"
    )

    print(
        f"Validation loader batches available: "
        f"{len(validation_loader)}"
    )

    model = create_densenet()
    model = model.to(device)

    assert_frozen_backbone(model)

    trainable, frozen = (
        count_trainable_parameters(model)
    )

    print(
        f"Trainable parameters: {trainable}"
    )

    print(
        f"Frozen parameters: {frozen}"
    )

    optimizer = Adam(
        filter(
            lambda parameter:
            parameter.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE,
    )

    history = []

    for epoch in range(EPOCHS):

        (
            train_loss,
            train_accuracy,
            batches_processed,
            samples_processed,
        ) = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            max_batches=MAX_TRAIN_BATCHES,
        )

        validation_metrics = evaluate(
            model,
            validation_loader,
            device,
            class_names=list(classes),
        )

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_batches": batches_processed,
            "train_samples": samples_processed,
            "validation_loss":
                validation_metrics["loss"],
            "validation_accuracy":
                validation_metrics["accuracy"],
            "validation_precision":
                validation_metrics[
                    "precision_weighted"
                ],
            "validation_recall":
                validation_metrics[
                    "recall_weighted"
                ],
            "validation_f1_score":
                validation_metrics[
                    "f1_weighted"
                ],
            "validation_confusion_matrix":
                validation_metrics[
                    "confusion_matrix"
                ],
            "validation_per_class":
                validation_metrics[
                    "per_class"
                ],
        }

        history.append(epoch_record)

        print()
        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"Training batches processed: "
            f"{batches_processed}"
        )

        print(
            f"Training samples processed: "
            f"{samples_processed}"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Validation Loss: "
            f"{validation_metrics['loss']:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{validation_metrics['accuracy']:.4f}"
        )

        print(
            f"Validation Precision: "
            f"{validation_metrics['precision_weighted']:.4f}"
        )

        print(
            f"Validation Recall: "
            f"{validation_metrics['recall_weighted']:.4f}"
        )

        print(
            f"Validation F1-score: "
            f"{validation_metrics['f1_weighted']:.4f}"
        )

        print(
            "Validation Confusion Matrix:"
        )

        for row in validation_metrics[
            "confusion_matrix"
        ]:
            print(row)

        print()
        print("Per-class metrics:")

        for class_name, metrics in (
            validation_metrics[
                "per_class"
            ].items()
        ):
            print(
                f"{class_name}: "
                f"Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}, "
                f"F1={metrics['f1_score']:.4f}, "
                f"Support={metrics['support']}"
            )

    elapsed_seconds = (
        time.perf_counter() - started
    )

    last = history[-1]

    checkpoint_payload = {
        "model_name": MODEL_NAME,
        "dataset_version": DATASET_VERSION,
        "hospital_id": hospital_id,
        "random_seed": RANDOM_SEED,
        "backbone_frozen": True,
        "num_classes": NUM_CLASSES,
        "class_names": list(classes),

        "hyperparameters": {
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "max_train_batches":
                MAX_TRAIN_BATCHES,
            "image_size": 224,
            "num_workers": 0,
            "pretrained": True,
        },

        "splits": {
            "global_test_ratio": 0.20,
            "hospital_validation_ratio":
                VALIDATION_RATIO,
            "notes":
                "Hospital-local 90/10 stratified split; "
                "global 80/20 split was not modified.",
        },

        "metrics": last,
        "history": history,
        "training_time_seconds":
            elapsed_seconds,
        "device": str(device),

        "disclaimer":
            "AI-assisted research prototype. "
            "Model output is a prediction, "
            "not a medical diagnosis.",

        "state_dict": model.state_dict(),
    }

    save_checkpoint(
        checkpoint_path,
        checkpoint_payload,
    )

    metrics_record = {
        key: value
        for key, value
        in checkpoint_payload.items()
        if key != "state_dict"
    }

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path.write_text(
        json.dumps(
            metrics_record,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )

    print(
        f"Metrics: "
        f"{metrics_path}"
    )

    print(
        f"Training time (seconds): "
        f"{elapsed_seconds:.2f}"
    )

    print(
        f"{hospital_id} training completed."
    )


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "FedXAI-Med Phase 1 — "
        "Frozen DenseNet Training"
    )
    print("=" * 70)
    print(
        "AI-assisted research prototype; "
        "not a diagnosis."
    )
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")
    print("Hospitals: 5")
    print(f"Epochs per hospital: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Seed: {RANDOM_SEED}")

    total_started = time.perf_counter()

    for hospital_id in HOSPITAL_IDS:
        train_hospital(
            hospital_id,
            device,
        )

    total_elapsed = (
        time.perf_counter() - total_started
    )

    print()
    print("=" * 70)
    print("ALL HOSPITALS TRAINING COMPLETED")
    print("=" * 70)
    print(
        f"Total training time (seconds): "
        f"{total_elapsed:.2f}"
    )


if __name__ == "__main__":
    main()