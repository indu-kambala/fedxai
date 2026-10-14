from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"
INDEX_DIR = PROJECT_ROOT / "artifacts" / "rag"

MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def chunk_text(text):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end == len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    documents = []

    for file_path in KNOWLEDGE_DIR.glob("*.md"):
        if file_path.name.lower() == "readme.md":
            continue

        text = file_path.read_text(encoding="utf-8")

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            documents.append(
                {
                    "source": file_path.name,
                    "chunk_id": i,
                    "text": chunk,
                }
            )

    if not documents:
        raise RuntimeError("No knowledge documents found.")

    print(f"Documents/chunks found: {len(documents)}")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    texts = [item["text"] for item in documents]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    index_path = INDEX_DIR / "medical_knowledge.faiss"
    metadata_path = INDEX_DIR / "metadata.json"

    faiss.write_index(index, str(index_path))

    metadata_path.write_text(
        json.dumps(documents, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("RAG index built successfully.")
    print(f"Embedding dimension: {dimension}")
    print(f"Vectors stored: {index.ntotal}")
    print(f"FAISS index: {index_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()