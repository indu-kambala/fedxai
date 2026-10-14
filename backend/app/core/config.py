from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central environment configuration. Secrets never belong in the frontend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    app_name: str = "FedXAI-Med"
    app_env: str = "development"
    dev_mode: bool = True
    log_level: str = "INFO"
    secret_key: str = "change-me-in-production"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    frontend_origin: str = "http://localhost:5173"

    database_url: str = "postgresql+psycopg://fedxai:fedxai@localhost:5432/fedxai_med"

    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    model_name: str = "densenet121"
    image_size: int = 224
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 10
    num_workers: int = 0
    mixed_precision: bool = False
    device: str = "auto"

    fl_algorithm: str = "fedavg"
    num_clients: int = 5
    local_epochs: int = 1
    fl_rounds: int = 10
    fedprox_mu: float = 0.01
    client_fraction: float = 1.0

    dp_enabled: bool = False
    dp_epsilon_target: float = 8.0
    dp_delta: float = 1e-5
    dp_max_grad_norm: float = 1.0
    dp_noise_multiplier: float = 1.0

    diffusion_enabled: bool = False
    diffusion_backend: str = "placeholder"

    llm_provider: str = "huggingface"
    llm_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    hf_api_token: str = ""
    hf_api_timeout_seconds: int = 60

    vector_db: str = "faiss"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    qdrant_url: str = "http://localhost:6333"

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "fedxai-med"

    max_upload_mb: int = 20
    allowed_image_extensions: str = "png,jpg,jpeg,dcm,dicom"
    cors_allow_credentials: bool = True
    rate_limit_per_minute: int = 60


settings = Settings()
