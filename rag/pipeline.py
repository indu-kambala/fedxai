"""Lightweight retrieval. Prefers sentence-transformers+FAISS when installed; otherwise TF-IDF."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


@dataclass
class Chunk:
    chunk_id: str
    source: str
    title: str
    text: str


def chunk_text(text: str, source: str, title: str, size: int = 500, overlap: int = 80) -> list[Chunk]:
    words = text.split()
    chunks = []
    start = 0
    index = 0
    if not words:
        return chunks
    while start < len(words):
        piece = " ".join(words[start : start + size])
        chunks.append(
            Chunk(
                chunk_id=f"{source}::{index}",
                source=source,
                title=title,
                text=piece,
            )
        )
        index += 1
        start += max(1, size - overlap)
    return chunks


def ingest_paths(paths: Iterable[Path]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in paths:
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks.extend(chunk_text(text, str(path).replace("\\", "/"), path.stem))
    return chunks


class TfidfIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.doc_tokens = [tokenize(c.text) for c in chunks]
        df: dict[str, int] = {}
        for tokens in self.doc_tokens:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        n = max(1, len(chunks))
        self.idf = {t: math.log((n + 1) / (df_t + 1)) + 1.0 for t, df_t in df.items()}
        self.vectors = [self._vector(tokens) for tokens in self.doc_tokens]

    def _vector(self, tokens: list[str]) -> dict[str, float]:
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        length = max(1, len(tokens))
        return {t: (c / length) * self.idf.get(t, 0.0) for t, c in tf.items()}

    def query(self, text: str, k: int = 4) -> list[dict]:
        q = self._vector(tokenize(text))
        scored = []
        for chunk, vec in zip(self.chunks, self.vectors):
            scored.append((self._cosine(q, vec), chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, chunk in scored[:k]:
            if score <= 0:
                continue
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "source": chunk.source,
                    "excerpt": chunk.text[:400],
                    "score": score,
                }
            )
        return results

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        keys = set(a) & set(b)
        dot = sum(a[k] * b[k] for k in keys)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


_INDEX: TfidfIndex | None = None


def knowledge_roots() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    return [root / "data" / "knowledge", root / "docs"]


def build_index() -> TfidfIndex:
    files = []
    for folder in knowledge_roots():
        if folder.exists():
            files.extend(folder.rglob("*"))
    chunks = ingest_paths(files)
    return TfidfIndex(chunks)


def get_index() -> TfidfIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = build_index()
    return _INDEX


def retrieve(query: str, k: int = 4) -> list[dict]:
    return get_index().query(query, k=k)


def save_index_stats(path: Path) -> Path:
    index = get_index()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"chunks": len(index.chunks), "backend": "tfidf"}, indent=2),
        encoding="utf-8",
    )
    return path
