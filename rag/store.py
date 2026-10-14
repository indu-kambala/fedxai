from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence


class VectorStoreBackend(ABC):
    name: str

    @abstractmethod
    def upsert(self, documents: Sequence[Mapping[str, Any]]) -> None: ...

    @abstractmethod
    def query(self, embedding: Sequence[float], k: int) -> Sequence[Mapping[str, Any]]: ...


class FaissVectorStore(VectorStoreBackend):
    name = "faiss"


class QdrantVectorStore(VectorStoreBackend):
    name = "qdrant"
