from backend.app.api import router
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

__all__ = ["router"]
