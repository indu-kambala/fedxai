# Methodology outline (paper-facing)

To be expanded as phases complete. Do not insert numeric results here until MLflow runs exist.

## Dataset

Configurable CXR multiclass task. Initial labels: NORMAL, PNEUMONIA, COVID-19. Five simulated hospitals, non-IID label and quantity skew. Real hospital data will be attached via adapters, not by rewriting the pipeline.

## Models

DenseNet-121, EfficientNet-B3, ViT-B/16, Swin Transformer with ImageNet (or equivalent) pretrained weights. Transfer-learning modes: frozen backbone, partial fine-tune, full fine-tune.

## Federated learning

FedAvg vs FedProx vs centralized baseline. Communication rounds, client drift, and participation are logged.

## Privacy

Optional DP-SGD (Opacus). Report ε, δ, noise, clip. Analyze utility loss. Do not claim perfect privacy.

## Diffusion

MedFusion-inspired class-conditional generation for minority-class experiments, with quality filters. Improvement is a **hypothesis**, not a given.

## Explainability

CNN: Grad-CAM / Grad-CAM++. Transformers: attention rollout / attention maps. SHAP where feasible. Heatmaps are attributions, not findings.

## CDS

RAG over user-provided or licensed documents plus configurable HF LLM with guardrails.

## Evaluation

Accuracy, precision, recall, macro-F1, ROC-AUC, PR-AUC, sensitivity, specificity, per-class scores, ECE, latency, model size. Sensitivity and per-class recall are emphasized.
