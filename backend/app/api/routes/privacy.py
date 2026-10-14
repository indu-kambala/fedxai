from fastapi import APIRouter

from backend.app.core.config import settings
from backend.app.core.yaml_loader import load_yaml

router = APIRouter()


@router.get("/privacy")
def privacy() -> dict:
    cfg = load_yaml("privacy.yaml")
    return {
        "raw_patient_images_centralized": False,
        "federated_training": True,
        "differential_privacy_enabled": settings.dp_enabled,
        "dp_epsilon_target": settings.dp_epsilon_target,
        "dp_delta": settings.dp_delta,
        "perfect_privacy_claimed": False,
        "config": cfg,
    }
