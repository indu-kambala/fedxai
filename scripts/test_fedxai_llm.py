from pathlib import Path
import json

import faiss
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from llm.client import generate


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_PATH = PROJECT_ROOT / "artifacts" / "rag" / "medical_knowledge.faiss"
METADATA_PATH = PROJECT_ROOT / "artifacts" / "rag" / "metadata.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def retrieve_knowledge(query, top_k=3):
    model = SentenceTransformer(EMBEDDING_MODEL)

    index = faiss.read_index(str(INDEX_PATH))

    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = query_embedding.astype("float32")

    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal),
    )

    results = []

    for score, index_id in zip(scores[0], indices[0]):
        if index_id == -1:
            continue

        item = metadata[index_id]

        results.append(
            {
                "title": item["source"],
                "excerpt": item["text"],
                "source": item["source"],
                "score": float(score),
            }
        )

    return results


def main():
    predicted_class = "COVID-19"
    confidence = 0.6605

    class_probabilities = {
        "COVID-19": 0.6605,
        "NORMAL": 0.0882,
        "PNEUMONIA": 0.2513,
    }

    query = (
        "Clinical decision-support information for a chest "
        "radiograph predicted as COVID-19."
    )

    print("Loading RAG knowledge...")
    retrieved = retrieve_knowledge(query)

    print()
    print("Retrieved knowledge:")
    print()

    for item in retrieved:
        print(
            f"{item['title']} "
            f"(similarity: {item['score']:.4f})"
        )

    context = {
        "prediction": predicted_class,
        "probability": confidence,
        "explanation": (
            "ViT attention visualization generated. "
            "The attention map is a model attribution and "
            "must not be treated as proof of disease."
        ),
        "class_probabilities": class_probabilities,
        "retrieved": retrieved,
    }

    question = (
        "Provide a concise research-oriented clinical "
        "decision-support summary for this AI prediction."
    )

    print()
    print("Generating clinical decision support...")

    result = generate(
        question=question,
        context=context,
    )

    print()
    print("=" * 70)
    print("FEDXAI-MED CLINICAL DECISION SUPPORT")
    print("=" * 70)
    print()
    print(result["content"])
    print()
    print(f"Provider: {result['provider']}")
    print(f"Model: {result['model']}")


if __name__ == "__main__":
    main()