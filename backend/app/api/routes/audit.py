from fastapi import APIRouter

router = APIRouter()


@router.get("/audit")
def audit() -> dict:
    return {"events": [], "note": "Audit persistence is implemented in later phases."}
