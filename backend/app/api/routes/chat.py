from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.app.services.rag_service import retrieve_knowledge
from llm.client import generate


router = APIRouter()


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    prediction: str | None = None
    probability: float | None = None
    explanation: str | None = None
    top_k: int = Field(default=3, ge=1, le=10)


@router.post("/chat")
def chat(payload: ChatRequest) -> dict:
    query = payload.question

    retrieved = retrieve_knowledge(
        query,
        payload.top_k,
    )

    context = {
        "prediction": payload.prediction,
        "probability": payload.probability,
        "explanation": (
            payload.explanation
            or "No XAI attribution information was provided."
        ),
        "retrieved": retrieved,
    }

    result = generate(
        question=payload.question,
        context=context,
    )

    return {
        "question": payload.question,
        "prediction": payload.prediction,
        "probability": payload.probability,
        "retrieved": retrieved,
        "clinical_support": result["content"],
        "provider": result["provider"],
        "model": result["model"],
        "is_diagnosis": False,
        "disclaimer": (
            "AI-assisted research prototype; "
            "this output is not a diagnosis. "
            "A qualified clinician remains the decision-maker."
        ),
    }