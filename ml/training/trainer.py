import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_trainable_parameters(model):
    trainable = 0
    frozen = 0

    for parameter in model.parameters():
        count = parameter.numel()

        if parameter.requires_grad:
            trainable += count
        else:
            frozen += count

    return trainable, frozen


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
    max_batches=None,
):
    model.train()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0
    batches_processed = 0
    samples_processed = 0

    for batch_index, (images, labels) in enumerate(data_loader):

        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        batch_size = images.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        batches_processed += 1
        samples_processed += batch_size

    if total == 0:
        raise RuntimeError(
            "No training samples were processed."
        )

    average_loss = total_loss / total
    accuracy = correct / total

    return (
        average_loss,
        accuracy,
        batches_processed,
        samples_processed,
    )


def evaluate(
    model,
    data_loader,
    device,
    class_names=None,
):
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            batch_size = images.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total += batch_size

            predictions = outputs.argmax(
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

    if total == 0:
        raise RuntimeError(
            "No validation samples were processed."
        )

    average_loss = (
        total_loss / total
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0,
        )
    )

    if class_names is not None:
        class_labels = list(
            range(len(class_names))
        )
    else:
        class_labels = None

    class_precision, class_recall, class_f1, class_support = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            labels=class_labels,
            average=None,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=class_labels,
    )

    metrics = {
        "loss": float(average_loss),

        "accuracy": float(accuracy),

        "precision_weighted": float(
            precision
        ),

        "recall_weighted": float(
            recall
        ),

        "f1_weighted": float(
            f1
        ),

        "confusion_matrix":
            matrix.tolist(),
    }

    if class_names is not None:

        metrics["per_class"] = {}

        for index, class_name in enumerate(
            class_names
        ):

            metrics["per_class"][
                class_name
            ] = {

                "precision": float(
                    class_precision[index]
                ),

                "recall": float(
                    class_recall[index]
                ),

                "f1_score": float(
                    class_f1[index]
                ),

                "support": int(
                    class_support[index]
                ),
            }

    return metrics


def save_checkpoint(path, payload):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        payload,
        path,
    )