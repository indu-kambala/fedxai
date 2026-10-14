"""Checkpoint helpers. Metadata only — never patient pixels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: Path, model: torch.nn.Module, metadata: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata,
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path, model: torch.nn.Module, map_location: str | torch.device = "cpu"):
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    return model, metadata
