"""Uncertainty and OOD flags. Softmax is not diagnostic certainty."""

from __future__ import annotations

from typing import Sequence


def max_softmax_ood(probabilities: Sequence[float], threshold: float = 0.55) -> dict:
    if not probabilities:
        return {
            "max_probability": None,
            "ood_flag": True,
            "message": "No class probabilities were produced. Treat as out-of-distribution / unusable.",
        }
    peak = max(probabilities)
    ood = peak < threshold
    return {
        "max_probability": peak,
        "ood_flag": ood,
        "threshold": threshold,
        "message": (
            "Low maximum softmax score. The image may be out-of-distribution or the model is uncertain. "
            "This is not diagnostic certainty."
            if ood
            else "Maximum softmax score is above the research OOD threshold. This is still not medical certainty."
        ),
    }
