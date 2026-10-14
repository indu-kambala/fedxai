# FedXAI-Med implementation plan

Gate: **do not start a phase until the previous phase builds and tests pass**, and the user says to proceed.

## Phase 0 — Architecture and repository — DONE

- [x] Analyze architecture and stack
- [x] Monorepo folders
- [x] Config YAML + `.env.example`
- [x] README, architecture docs, TODO
- [x] requirements / frontend package
- [x] Docker Compose skeleton
- [x] Module interfaces and API 501 stubs
- [x] Phase 0 tests
- [ ] User review, then `Proceed to Phase 1`

## Phase 1 — Dataset ingestion and validation

- [ ] Hospital folder adapters (`HospitalDataset`, `FederatedHospitalDataset`, `MedicalImageDataset`)
- [ ] Configurable classes / formats
- [ ] Duplicate, corruption, distribution reports
- [ ] Patient-leakage checks when IDs exist
- [ ] **Do not train** on newly dropped hospital data automatically

## Phase 2 — Preprocessing

- [ ] DICOM/PNG/JPG validation
- [ ] De-identification hooks
- [ ] Resize, 1→3 channel handling
- [ ] Conservative, documented augmentations after split only

## Phase 3 — Baseline CNNs

- [ ] DenseNet-121, EfficientNet-B3
- [ ] Frozen / partial / full fine-tune
- [ ] MLflow metrics (no invented numbers)

## Phase 4 — ViT and Swin

- [ ] ViT-B/16, Swin
- [ ] CPU/GPU and `DEV_MODE` subsets

## Phase 5 — Model comparison

- [ ] Macro-F1, ROC-AUC, sensitivity, specificity, ECE, latency, size
- [ ] Selection policy (not accuracy-only)

## Phase 6 — Diffusion augmentation

- [ ] `ml/diffusion/` trainer, sampler, conditioning, quality checks
- [ ] Label as MedFusion-inspired unless faithfully reproduced
- [ ] Experiments A/B/C; do not assume synthetic data helps

## Phase 7 — Federated learning (Flower)

- [ ] 5 clients, FedAvg, round metrics
- [ ] Non-IID partitions + visualization
- [ ] Updates only on the wire

## Phase 8 — FedProx

- [ ] Configurable μ
- [ ] Centralized vs FedAvg vs FedProx comparison

## Phase 9 — Differential privacy

- [ ] Opacus, ε/δ accounting
- [ ] No-DP vs low vs higher noise; no perfect-privacy claims

## Phase 10 — Explainable AI

- [ ] Grad-CAM / Grad-CAM++ for CNNs
- [ ] Attention rollout for transformers
- [ ] Optional SHAP
- [ ] Non-clinical wording for heatmaps

## Phase 11 — Calibration and uncertainty

- [ ] Reliability diagrams, ECE, temperature scaling
- [ ] OOD / low-confidence messaging

## Phase 12 — RAG

- [ ] FAISS (default) / Qdrant
- [ ] Citations required; no invented sources

## Phase 13 — Hugging Face LLM

- [ ] `LLM_PROVIDER`, `LLM_MODEL`, `HF_API_TOKEN`
- [ ] Guardrails + fallbacks (timeout, rate limit, down)

## Phase 14 — FastAPI product APIs

- [ ] Multipart analyze, auth, rate limits, audit persistence

## Phase 15 — React medical UI

- [ ] Workstation layout, image viewer, chat, themes
- [ ] Routes: login, dashboard, analyze, cases, chat, models, experiments, federated, privacy, knowledge, reports, audit, settings

## Phase 16 — Federated dashboard

- [ ] Live rounds, client status, communication cost

## Phase 17 — Experiment dashboard

- [ ] Comparison tables; **Not evaluated** if unrun

## Phase 18 — Reports

- [ ] PDF/HTML case reports with disclaimer and doctor review block

## Phase 19 — Security

- [ ] RBAC, TLS-ready, encrypted secrets, CORS, upload limits

## Phase 20 — Testing

- [ ] Dataset, preprocess, inference, FL, DP, RAG, LLM fallback, XAI, reports, UI critical paths

## Phase 21 — Docker / deployment

- [ ] Compose profiles: cpu-dev vs gpu-train
- [ ] Health checks

## Phase 22 — Research documentation

- [ ] Methodology, ablations, limitations, future work
- [ ] Fill result tables **only** from MLflow

---

Definition of done is listed in `docs/architecture/overview.md`. Experiment tables must never contain invented metrics.
