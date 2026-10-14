from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = REPO_ROOT / "configs"


def load_yaml(name: str) -> dict:
    path = CONFIGS_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config {name} must be a mapping")
    return data
