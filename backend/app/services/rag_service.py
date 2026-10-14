from pathlib import Path
import json

import faiss
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INDEX_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "rag"
    / "medical_knowledge.faiss"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "rag"
    / "metadata.json"
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_embedding_model = None
_index = None
_metadata = None


def load_rag():
    global _embedding_model
    global _index
    global _metadata

    if (
        _embedding_model is not None
        and _index is not None
        and _metadata is not None
    ):
        return (
            _embedding_model,
            _index,
            _metadata,
        )

    _embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    _index = faiss.read_index(
        str(INDEX_PATH)
    )

    _metadata = json.loads(
        METADATA_PATH.read_text(
            encoding="utf-8"
        )
    )

    return (
        _embedding_model,
        _index,
        _metadata,
    )


def retrieve_knowledge(
    query: str,
    top_k: int = 3,
):
    embedding_model, index, metadata = load_rag()

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = query_embedding.astype(
        "float32"
    )

    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal),
    )

    results = []

    for score, index_id in zip(
        scores[0],
        indices[0],
    ):
        if index_id == -1:
            continue

        item = metadata[index_id]

        results.append(
            {
                "title": item["source"],
                "excerpt": item["text"],
                "source": item["source"],
                "similarity": float(score),
            }
        )

    return results