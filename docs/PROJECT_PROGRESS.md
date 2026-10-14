# FedXAI-Med project progress

Research prototype. Predictions are **not a diagnosis**.

| Phase | Status | Tests | Notes |
| --- | --- | --- | --- |
| 0 Architecture / FastAPI foundation | COMPLETED | pass | Existing Phase 0 contracts retained |
| 1 Frozen DenseNet local training | IN PROGRESS | pending | Capped CPU run, checkpoint + metrics |
| 2 Evaluation protocol | NOT STARTED | | |
| 3 EfficientNet | NOT STARTED | | |
| 4 ViT | NOT STARTED | | |
| 5 Swin | NOT STARTED | | |
| 6 FedAvg (PyTorch) | NOT STARTED | | No Flower |
| 7 FedProx | NOT STARTED | | |
| 8 Differential privacy | NOT STARTED | | Honest Opacus limits |
| 9 Explainability | NOT STARTED | | |
| 10 Calibration / OOD | NOT STARTED | | |
| 11 Medical RAG | NOT STARTED | | Existing `data/knowledge` |
| 12 LLM CDS | NOT STARTED | | HF + grounded fallback |
| 13 PostgreSQL / SQLAlchemy | NOT STARTED | | |
| 14 JWT / RBAC | NOT STARTED | | |
| 15 Security hardening | NOT STARTED | | |
| 16 React wiring | NOT STARTED | | Keep Vite/Tailwind |
| 17 Reports / audit | NOT STARTED | | |
| 18 Testing | NOT STARTED | | Update 501 tests when real |
| 19 Docker | NOT STARTED | | |
| 20 Research documentation | NOT STARTED | | Measured results only |
| 21 Research evaluation tables | NOT STARTED | | No fabricated numbers |
| Diffusion / MedFusion-inspired | NOT STARTED | | Deferred prototype after FL |

**Overall completion: 24%** (starting point from audit). Percentages are updated only after implementation and tests.
