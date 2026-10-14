"""JSON experiment records. Never invent metrics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def artifacts_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    path = root / "artifacts" / "experiments"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_record(name: str, payload: dict[str, Any]) -> Path:
    record = {
        **payload,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": (
            "AI-assisted research prototype. Metrics are from this run only. "
            "Not a medical diagnosis. Not clinically validated."
        ),
    }
    path = artifacts_dir() / f"{name}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def read_record(name: str) -> dict[str, Any] | None:
    path = artifacts_dir() / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_records() -> list[dict[str, Any]]:
    items = []
    for path in sorted(artifacts_dir().glob("*.json")):
        try:
            items.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return items
