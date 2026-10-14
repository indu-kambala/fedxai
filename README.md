# FedXAI-Med

**A privacy-preserving federated learning platform with explainable AI, MedFusion-inspired diffusion augmentation, and LLM-powered clinical decision support.**

Research-grade B.Tech / AIML prototype. **Not a medical device. Not a diagnosis system.** All predictions, attributions, and chat answers require review by a qualified clinician.

---

## Current status

**Phase 0 complete.** Architecture, repository, configuration, Docker skeleton, API contracts, and placeholders are in place. Training, inference, federated rounds, diffusion, RAG, and the full workstation UI are **not** implemented yet.

Do not proceed to later phases until you explicitly instruct: `Proceed to Phase 1`.

---

## Scientific narrative

Hospitals cannot freely share raw chest X-rays. Federated learning collaborates on **model updates** instead of pixels. FedAvg is the baseline aggregator; FedProx targets client heterogeneity; differential privacy adds a **quantified privacy–utility tradeoff** (not perfect privacy). CNN and transformer backbones are compared without assuming a winner. Diffusion augmentation is **MedFusion-inspired** unless a faithful original MedFusion backend is later wired in. XAI, calibration, RAG, and an LLM convert model outputs into **decision support**, with the physician remaining the decision-maker.

---

## Technology stack

| Layer | Choice | Why |
| --- | --- | --- |
| Frontend | React 18, Vite, TypeScript, Tailwind CSS, shadcn/ui (later), Recharts | Clinical workstation UX, not a generic admin theme |
| API | FastAPI, Pydantic v2 | Typed contracts, OpenAPI, async-ready |
| Auth | JWT + RBAC (Doctor, Researcher, Hospital Admin, System Admin) | Role-appropriate surfaces |
| DB | PostgreSQL (metadata only) | Cases, experiments, audit; **no central raw CXR store** |
| FL | Flower simulation, 5 hospital clients | FedAvg + FedProx, non-IID partitions |
| Models | PyTorch, torchvision, timm | DenseNet-121, EfficientNet-B3, ViT-B/16, Swin |
| DP | Opacus | ε, δ, clip, noise accounting |
| Diffusion | Replaceable backend (placeholder → pretrained conditional diffusion) | Explicitly **not** claimed as original MedFusion |
| XAI | Grad-CAM(++) for CNNs; attention rollout for transformers; optional SHAP | Architecture-appropriate attribution |
| RAG | FAISS default, Qdrant optional | Student-practical, modular |
| LLM | Hugging Face Inference API, configurable model | Tokens stay server-side; fallback if unavailable |
| Tracking | MLflow | Real metrics only; unrun experiments show **Not evaluated** |
| Ops | Docker Compose, pytest | Laptop-friendly `DEV_MODE` |

---

## Privacy invariant

```
Hospital local CXR  →  local train  →  weight update  →  FL server
Raw pixels never leave the hospital client for aggregation.
```

Inference for a logged-in doctor is a **hospital-side / authorized API** path with audit logging, not “upload all images to a public model zoo.”

---

## Repository layout

```
backend/          FastAPI app
frontend/         React workstation (Phase 0 shell)
ml/               Models, FL, diffusion, XAI, evaluation
rag/              Vector-store interfaces
llm/              Prompts and HF client stubs
configs/          YAML (dataset, models, FL, privacy, experiments)
data/hospital_*   Client-isolated datasets (gitignored images)
docs/             Architecture and research methodology
docker/           Containerfiles
tests/            Phase 0 contracts; later phases add suites
```

---

## Configuration

Copy `.env.example` → `.env`. Never commit `.env`, API keys, or patient images.

Primary YAML:

- `configs/dataset.yaml` — classes (NORMAL, PNEUMONIA, COVID-19), 5 clients, splits, conservative augmentations
- `configs/models.yaml` — four backbones, transfer-learning modes
- `configs/federated.yaml` — FedAvg / FedProx, no raw images on the wire
- `configs/privacy.yaml` — Opacus parameters
- `configs/diffusion.yaml` — MedFusion-inspired flag is `false` for original-model claims
- `configs/experiments.yaml` — ablation matrix, all **not_evaluated**

---

## Local development (Phase 0)

Recommend **Python 3.11 or 3.12** for later ML wheels (PyTorch, Flower, Opacus). Phase 0 tests run with the lightweight `requirements.txt` stack.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q
# Later phases: pip install -r requirements-backend.txt
# Later phases: pip install -r requirements-ml.txt

cd frontend
npm install
npm run build
```

API (after deps):

```bash
uvicorn backend.app.main:app --reload --port 8000
```

Optional Compose: `docker compose up postgres mlflow`

GPU training is **not** required for Phase 0. Later phases detect CUDA and fall back to CPU with small batches and `DEV_MODE=true`.

---

## Classes

Initial radiographic targets: **NORMAL**, **PNEUMONIA**, **COVID-19**.  
Common cold is **not** a chest X-ray class. The class list is configurable and extensible.

---

## Documentation

- [Architecture](docs/architecture/overview.md)
- [Implementation plan](TODO.md)
- [Limitations](docs/research/limitations.md)
- [Augmentations](docs/methodology/augmentations.md)
- [Data leakage policy](docs/methodology/data_leakage.md)

---

## License / ethics

For academic use. Do not deploy as an autonomous diagnostic product. Do not train on newly provided hospital data until Phase 1 validation passes.
