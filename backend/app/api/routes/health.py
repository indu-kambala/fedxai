from fastapi import APIRouter

from backend.app.core.config import settings
from backend.app.core.disclaimers import CLINICAL_DISCLAIMER, PRIVACY_PRINCIPLE

router = APIRouter()


@router.get("/status")
def api_status() -> dict:
    return {
        "phase": 0,
        "dev_mode": settings.dev_mode,
        "disclaimer": CLINICAL_DISCLAIMER,
        "privacy": PRIVACY_PRINCIPLE,
        "implemented": ["health", "config_contracts"],
        "deferred": [
            "inference",
            "federated_training",
            "llm",
            "rag",
            "reports",
        ],
    }
