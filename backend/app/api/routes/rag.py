from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.app.services.rag_service import (
    retrieve_knowledge,
)


router = APIRouter()


class RAGQueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
    )


@router.post("/rag/query")
def rag_query(
    payload: RAGQueryRequest,
) -> dict:

    results = retrieve_knowledge(
        payload.query,
        payload.top_k,
    )

    return {
        "query": payload.query,
        "results": results,
    }