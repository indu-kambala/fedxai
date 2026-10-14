"""SQLAlchemy models will land with the database phase. Central DB stores metadata, not raw CXR pixels."""

ENTITIES = [
    "users",
    "hospitals",
    "cases",
    "predictions",
    "model_versions",
    "experiments",
    "federated_rounds",
    "client_metrics",
    "audit_logs",
    "chat_sessions",
    "chat_messages",
    "knowledge_documents",
    "citations",
    "privacy_metrics",
]
