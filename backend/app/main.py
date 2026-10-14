"""FedXAI-Med FastAPI application (Phase 0: health, config, contracts)."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.router import api_router
from backend.app.core.config import settings
from backend.app.core.disclaimers import CLINICAL_DISCLAIMER

app = FastAPI(
    title=settings.app_name,
    version="0.0.1",
    description=CLINICAL_DISCLAIMER,
    docs_url="/docs" if settings.dev_mode else None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(api_router, prefix="/api")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "phase": "0",
        "app": settings.app_name,
        "disclaimer": "AI-assisted research prototype; not a diagnosis.",
    }
