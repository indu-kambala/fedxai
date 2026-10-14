"""Phase 0 structural and contract tests. No fabricated experiment metrics."""

from pathlib import Path

REQUIRED_DIRS = [
    "backend/app",
    "frontend/src",
    "ml/models/densenet",
    "ml/models/efficientnet",
    "ml/models/vit",
    "ml/models/swin",
    "ml/federated",
    "ml/diffusion",
    "ml/explainability",
    "rag",
    "llm",
    "configs",
    "docs/architecture",
    "tests",
    "docker",
]

REQUIRED_CONFIGS = [
    "configs/dataset.yaml",
    "configs/models.yaml",
    "configs/training.yaml",
    "configs/federated.yaml",
    "configs/privacy.yaml",
    "configs/diffusion.yaml",
    "configs/llm.yaml",
    "configs/rag.yaml",
    "configs/experiments.yaml",
]


def test_repository_layout_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in REQUIRED_DIRS:
        assert (root / relative).is_dir(), relative


def test_required_configs_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in REQUIRED_CONFIGS:
        assert (root / relative).is_file(), relative
