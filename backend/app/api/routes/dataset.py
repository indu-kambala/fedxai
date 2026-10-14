from fastapi import APIRouter
from backend.app.services.dataset_validator import validate_dataset

router = APIRouter(prefix="/dataset", tags=["Dataset"])


@router.get("/validate")
def validate_dataset_route():
    return validate_dataset()