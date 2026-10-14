# System architecture (Phase 0)

## Clinical scope

FedXAI-Med is a **clinical decision-support research prototype**. It classifies chest radiographs into a configurable label set (initially NORMAL, PNEUMONIA, COVID-19), explains model attributions, and retrieves literature. It does **not** independently diagnose or replace a clinician.

## Logical architecture

```
Doctor UI (React + Tailwind)
        │ HTTPS/TLS (production)
        ▼
FastAPI ── Inference API ── Global FL checkpoint (weights)
       └── LLM/RAG engine ── Vector store + HF model

Federated training (Flower):
  Global model
    ├── Hospital A local data
    ├── Hospital B local data
    ├── Hospital C local data
    ├── Hospital D local data
    └── Hospital E local data
  Aggregation ← model updates only
```

## Trust boundaries

| Asset | Location | Central server |
| --- | --- | --- |
| Raw CXR pixels | Hospital client disk | Never for FL aggregation |
| Labels / local metrics | Client, summarized | Aggregate statistics only |
| Model weights | Clients + server | Yes |
| Case metadata / audit | PostgreSQL | Yes, no pixel blobs by default |
| HF API token | Backend env | Never in frontend |

Inference uploads in the doctor UI are an **authorized clinical workflow**, audited, size- and type-limited, and distinct from federated aggregation.

## Component map

- **frontend/** — Medical workstation: analysis canvas, XAI views, CDS chat, FL and experiment dashboards, privacy center.
- **backend/** — AuthZ, orchestration, report generation, audit. Delegates ML to `ml/`.
- **ml/models** — DenseNet-121, EfficientNet-B3, ViT-B/16, Swin; transfer-learning modes.
- **ml/federated** — FedAvg, FedProx, non-IID, Opacus DP.
- **ml/diffusion** — Conditional synthetic CXR; replaceable backend; quality gates.
- **ml/explainability** — CNN CAM vs transformer attention; optional SHAP.
- **ml/calibration** + **uncertainty** — ECE, reliability, OOD flag.
- **rag/** + **llm/** — Grounded answers; fallback if the provider fails.

## Data flow (doctor analysis)

1. Authenticated doctor opens New Case.
2. Image is validated (type, size, basic integrity).
3. Preprocess (no test-time heavy medical-inappropriate aug).
4. Inference → class scores.
5. Calibration / uncertainty.
6. Architecture-aware XAI.
7. UI shows prediction, probability, calibration status, disclaimer.
8. Chat receives model output + XAI metadata + retrieved passages only.
9. Optional report with doctor review section.

## Federated data flow

1. Each client loads **only** its `data/hospital_n` tree.
2. Local epochs produce a weight delta.
3. Server aggregates (FedAvg or FedProx).
4. Round metrics logged to MLflow and `federated_rounds`.
5. Failure of a client does not ship its images as fallback.

## Model selection policy

Do not pick a backbone by accuracy alone. Rank by macro-F1, ROC-AUC, sensitivity/recall (per class), specificity, calibration (ECE), and cost (latency, parameters). Medical emphasis: **minority-class recall and false negatives**.

## Experiment integrity

`configs/experiments.yaml` lists the ablation matrix. APIs and UI must render **Not evaluated** until a run writes metrics. Git commit hash is stored when available.

## Security (target)

TLS termination in production, JWT RBAC, CORS allowlist, rate limits, upload caps, no secrets in the SPA, audit log of access and inference. DP is optional and reported honestly.

## Risks (Phase 0 identification)

| Risk | Mitigation |
| --- | --- |
| No GPU | Transfer learning, small image size, DEV_MODE subset, CPU fallback |
| Flower + Opacus complexity | Modular strategies; DP off by default |
| Diffusion cost | Placeholder + pretrained backend; never required for core FL demo |
| LLM hallucination | RAG citations, refusal to invent, deterministic fallback |
| Data leakage | Split before aug; patient-level split; no synthetic from test |
| Overclaimed privacy | UI copy: tradeoff, not perfect secrecy |
| Fake medical UI | Clinical typography, disclaimers, no neon “diagnosis” language |

## Definition of done (project-level)

See the user specification §50. Phase 0 only claims: repo, configs, docs, stubs, and passing contract tests.
