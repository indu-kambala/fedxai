from fastapi import APIRouter

from backend.app.core.yaml_loader import load_yaml

router = APIRouter()


@router.get("/federated/status")
def federated_status() -> dict:
    cfg = load_yaml("federated.yaml")
    return {
        "algorithm": cfg.get("algorithms", {}).get("default"),
        "clients": cfg.get("clients"),
        "raw_images_on_wire": cfg.get("raw_images_on_wire", False),
        "training": "Not evaluated",
    }


@router.get("/federated/rounds")
def federated_rounds() -> dict:
    return {"rounds": [], "status": "Not evaluated"}
