from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_PATH = PROJECT_ROOT / "artifacts" / "rag" / "medical_knowledge.faiss"
METADATA_PATH = PROJECT_ROOT / "artifacts" / "rag" / "metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"


def load_rag():
    model = SentenceTransformer(MODEL_NAME)

    index = faiss.read_index(str(INDEX_PATH))

    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    return model, index, metadata


def retrieve(query, top_k=3):
    model, index, metadata = load_rag()

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
                "score": float(score),
                "source": item["source"],
                "chunk_id": item["chunk_id"],
                "text": item["text"],
            }
        )

    return results


def main():
    query = "How should chest radiographs be interpreted for pneumonia?"

    print(f"Query: {query}")
    print()

    results = retrieve(query)

    for i, result in enumerate(results, start=1):
        print(f"Result {i}")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Source: {result['source']}")
        print(f"Chunk: {result['chunk_id']}")
        print()
        print(result["text"])
        print("-" * 70)


if __name__ == "__main__":
    main()