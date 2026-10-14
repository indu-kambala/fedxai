from fastapi import APIRouter

from backend.app.core.yaml_loader import load_yaml

router = APIRouter()


@router.get("/models")
def list_models() -> dict:
    return {
        "configured": load_yaml("models.yaml"),
        "results": "Not evaluated",
        "note": "Do not invent metrics. Run experiments before reporting numbers.",
    }
