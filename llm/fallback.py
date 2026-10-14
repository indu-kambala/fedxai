from llm.prompts import SYSTEM_PROMPT


def grounded_fallback(context: dict | None = None) -> str:
    context = context or {}
    prediction = context.get("prediction") or context.get("predicted_class") or "unavailable"
    return (
        f"Model output (not a diagnosis): {prediction}. "
        "I will not invent findings, papers, or confidence scores. "
        "FedXAI-Med is an AI-assisted research prototype. "
        "A qualified clinician must review the image and the clinical context."
    )
