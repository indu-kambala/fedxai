from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.post("/reports")
def create_report() -> dict:
    raise HTTPException(status_code=501, detail="Report generation is Phase 18.")
