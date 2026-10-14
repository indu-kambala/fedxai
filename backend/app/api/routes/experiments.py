from fastapi import APIRouter

from backend.app.core.yaml_loader import load_yaml

router = APIRouter()


@router.get("/experiments")
def list_experiments() -> dict:
    spec = load_yaml("experiments.yaml")
    return {
        "tracking": spec.get("tracking"),
        "matrix": spec.get("matrix"),
        "unrun_display": spec.get("unrun_display", "Not evaluated"),
    }
