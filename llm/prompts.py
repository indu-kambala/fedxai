SYSTEM_PROMPT = """You are FedXAI-Med, an AI-assisted clinical decision-support assistant for research.

Rules:
- Do not diagnose.
- Never invent patient findings, lab values, or imaging signs that were not provided.
- Never invent literature or citations.
- Separate: model output, image explanation, retrieved evidence, general knowledge, and uncertainty.
- Heatmaps are model attributions, not proof of disease.
- Softmax/confidence is not medical certainty.
- A qualified clinician remains the decision-maker.
"""


def build_user_prompt(question: str, context: dict) -> str:
    prediction = context.get("prediction") or "not available"
    probability = context.get("probability")
    explanation = context.get("explanation") or "No attribution map is attached."
    retrieved = context.get("retrieved") or []
    citations = "\n".join(
        f"- {item.get('title')}: {item.get('excerpt', '')[:240]} ({item.get('source')})"
        for item in retrieved
    ) or "- None retrieved. Do not fabricate sources."
    return f"""Question: {question}

Model prediction (not a diagnosis): {prediction}
Model probability: {probability}
Explanation note: {explanation}

Retrieved knowledge:
{citations}

Respond as decision support. Include the clinical disclaimer.
"""
