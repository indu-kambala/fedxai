"""Classification metrics. Values must come from actual predictions."""

from __future__ import annotations

from typing import Sequence

import math


CLASS_NAMES = ["COVID-19", "NORMAL", "PNEUMONIA"]


def _safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d else 0.0


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int = 3) -> list[list[int]]:
    matrix = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            matrix[t][p] += 1
    return matrix


def per_class_metrics(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int = 3) -> list[dict]:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    rows = []
    for i in range(n_classes):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n_classes) if r != i)
        fn = sum(cm[i][c] for c in range(n_classes) if c != i)
        tn = sum(
            cm[r][c]
            for r in range(n_classes)
            for c in range(n_classes)
            if r != i and c != i
        )
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        specificity = _safe_div(tn, tn + fp)
        rows.append(
            {
                "class": CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i),
                "precision": precision,
                "recall": recall,
                "sensitivity": recall,
                "specificity": specificity,
                "f1": f1,
                "support": sum(cm[i]),
            }
        )
    return rows


def overall_metrics(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int = 3) -> dict:
    if not y_true:
        return {
            "status": "Not evaluated",
            "accuracy": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "note": "No predictions were collected.",
        }
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    accuracy = _safe_div(correct, len(y_true))
    per_class = per_class_metrics(y_true, y_pred, n_classes)
    return {
        "status": "measured",
        "n_samples": len(y_true),
        "accuracy": accuracy,
        "macro_precision": sum(r["precision"] for r in per_class) / n_classes,
        "macro_recall": sum(r["recall"] for r in per_class) / n_classes,
        "macro_f1": sum(r["f1"] for r in per_class) / n_classes,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, n_classes),
        "class_names": CLASS_NAMES[:n_classes],
    }


def roc_auc_ovr(y_true: Sequence[int], probabilities: Sequence[Sequence[float]], n_classes: int = 3) -> dict:
    """One-vs-rest ROC-AUC using trapezoid integration. Returns Not evaluated if degenerate."""
    if not y_true or not probabilities:
        return {"macro_roc_auc": None, "per_class": [], "status": "Not evaluated"}

    per_class = []
    scores = []
    for cls in range(n_classes):
        pairs = sorted(
            ((float(prob[cls]), 1 if label == cls else 0) for label, prob in zip(y_true, probabilities)),
            key=lambda x: x[0],
            reverse=True,
        )
        positives = sum(p[1] for p in pairs)
        negatives = len(pairs) - positives
        if positives == 0 or negatives == 0:
            per_class.append({"class": CLASS_NAMES[cls], "roc_auc": None, "status": "Not evaluated"})
            continue
        tps = fps = 0
        xs = [0.0]
        ys = [0.0]
        for _, is_pos in pairs:
            if is_pos:
                tps += 1
            else:
                fps += 1
            xs.append(fps / negatives)
            ys.append(tps / positives)
        auc = 0.0
        for i in range(1, len(xs)):
            auc += (xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2.0
        per_class.append({"class": CLASS_NAMES[cls], "roc_auc": auc, "status": "measured"})
        scores.append(auc)
    return {
        "macro_roc_auc": (sum(scores) / len(scores)) if scores else None,
        "per_class": per_class,
        "status": "measured" if scores else "Not evaluated",
    }


def expected_calibration_error(
    y_true: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    n_bins: int = 10,
) -> dict:
    if not y_true or not probabilities:
        return {"ece": None, "status": "Not evaluated"}
    confidences = [max(p) for p in probabilities]
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in probabilities]
    bins = [[] for _ in range(n_bins)]
    for conf, pred, label in zip(confidences, preds, y_true):
        idx = min(n_bins - 1, int(conf * n_bins))
        bins[idx].append((conf, int(pred == label)))
    ece = 0.0
    reliability = []
    n = len(y_true)
    for i, bucket in enumerate(bins):
        if not bucket:
            reliability.append({"bin": i, "count": 0, "confidence": None, "accuracy": None})
            continue
        mean_conf = sum(b[0] for b in bucket) / len(bucket)
        acc = sum(b[1] for b in bucket) / len(bucket)
        ece += (len(bucket) / n) * abs(acc - mean_conf)
        reliability.append(
            {"bin": i, "count": len(bucket), "confidence": mean_conf, "accuracy": acc}
        )
    return {"ece": ece, "reliability": reliability, "status": "measured"}


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
