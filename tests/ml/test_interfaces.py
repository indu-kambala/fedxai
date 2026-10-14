from ml.interfaces import ExperimentRecord, Split
from ml.interfaces import FedAvgStrategy, FedProxStrategy
from rag.store import FaissVectorStore, QdrantVectorStore
from llm.fallback import grounded_fallback
from llm.prompts import SYSTEM_PROMPT


def test_experiment_record_refuses_empty_metrics() -> None:
    record = ExperimentRecord(experiment_id="exp01", model="densenet121")
    assert record.public_metrics()["status"] == "Not evaluated"


def test_split_enum() -> None:
    assert Split.TRAIN.value == "train"
    assert Split.TEST.value == "test"


def test_fl_strategy_names() -> None:
    assert FedAvgStrategy.name == "fedavg"
    assert FedProxStrategy.name == "fedprox"


def test_vector_store_is_modular() -> None:
    assert FaissVectorStore.name == "faiss"
    assert QdrantVectorStore.name == "qdrant"


def test_llm_fallback_is_non_diagnostic() -> None:
    text = grounded_fallback({"prediction": "PNEUMONIA"})
    assert "PNEUMONIA" in text
    assert "not a diagnosis" in text.lower()
    assert "invent" not in text.lower()


def test_system_prompt_has_guardrails() -> None:
    assert "do not diagnose" in SYSTEM_PROMPT.lower()
    assert "never invent" in SYSTEM_PROMPT.lower()
