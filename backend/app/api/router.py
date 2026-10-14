from fastapi import APIRouter
from backend.app.api.routes import dataset
from backend.app.api.routes import (
    analyze,
    audit,
    chat,
    experiments,
    explain,
    federated,
    health,
    models,
    privacy,
    rag,
    reports,
)

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(analyze.router, tags=["inference"])
api_router.include_router(explain.router, tags=["explainability"])
api_router.include_router(chat.router, tags=["cds"])
api_router.include_router(rag.router, tags=["rag"])
api_router.include_router(models.router, tags=["models"])
api_router.include_router(experiments.router, tags=["experiments"])
api_router.include_router(federated.router, tags=["federated"])
api_router.include_router(privacy.router, tags=["privacy"])
api_router.include_router(audit.router, tags=["audit"])
api_router.include_router(reports.router, tags=["reports"])
api_router.include_router(dataset.router)
