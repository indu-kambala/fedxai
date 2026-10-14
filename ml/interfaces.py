"""Stable contracts for datasets, models, FL, XAI, RAG, and diffusion.

Phase 0 defines interfaces only. Implementations arrive in later phases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


class Split(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class DatasetSpec:
    class_names: Sequence[str]
    image_size: int
    hospital_id: str
    root: Path
    split: Split


class MedicalImageDataset(Protocol):
    """Local CXR dataset. Pixels stay on the client that owns `root`."""

    spec: DatasetSpec

    def __len__(self) -> int: ...

    def validate(self) -> Mapping[str, Any]: ...


class HospitalDataset(Protocol):
    """One simulated or real hospital client partition."""

    hospital_id: str
    data_root: Path

    def local_dataset(self, split: Split) -> MedicalImageDataset: ...


class FederatedHospitalDataset(Protocol):
    """Logical isolation across hospitals; never concatenates pixels centrally."""

    clients: Sequence[HospitalDataset]

    def client(self, hospital_id: str) -> HospitalDataset: ...


class ChestXrayClassifier(ABC):
    family: str  # cnn | transformer

    @abstractmethod
    def forward(self, images: Any) -> Any: ...

    @abstractmethod
    def parameter_update(self) -> Mapping[str, Any]:
        """Weights only. No image tensors."""


class DenseNet121Interface(ChestXrayClassifier):
    family = "cnn"


class EfficientNetB3Interface(ChestXrayClassifier):
    family = "cnn"


class ViTB16Interface(ChestXrayClassifier):
    family = "transformer"


class SwinTransformerInterface(ChestXrayClassifier):
    family = "transformer"


class FederatedStrategy(ABC):
    name: str

    @abstractmethod
    def aggregate(self, client_updates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        ...


class FedAvgStrategy(FederatedStrategy):
    name = "fedavg"


class FedProxStrategy(FederatedStrategy):
    name = "fedprox"
    mu: float = 0.01


class DifferentialPrivacyAccountant(Protocol):
    def report(self) -> Mapping[str, float]:
        """epsilon, delta, noise_multiplier, clipping_norm — not a perfect-privacy claim."""


class ExplanationEngine(ABC):
    """Architecture-aware XAI. Grad-CAM is for CNNs, not blindly for ViT/Swin."""

    @abstractmethod
    def explain(self, image: Any, model: ChestXrayClassifier) -> Mapping[str, Any]: ...


class DiffusionBackend(ABC):
    """MedFusion-inspired synthetic CXR generation. Not original MedFusion by default."""

    claim_original_medfusion: bool = False

    @abstractmethod
    def sample(self, class_name: str, n: int) -> Sequence[Any]: ...


class VectorStore(Protocol):
    def upsert(self, documents: Sequence[Mapping[str, Any]]) -> None: ...

    def query(self, embedding: Sequence[float], k: int) -> Sequence[Mapping[str, Any]]: ...


class LLMProvider(Protocol):
    def generate(self, prompt: str, context: Mapping[str, Any]) -> str: ...


@dataclass
class ExperimentRecord:
    experiment_id: str
    model: str
    status: str = "Not evaluated"
    metrics: dict[str, Any] = field(default_factory=dict)

    def public_metrics(self) -> Mapping[str, Any]:
        if self.status == "Not evaluated" or not self.metrics:
            return {"status": "Not evaluated"}
        return self.metrics
