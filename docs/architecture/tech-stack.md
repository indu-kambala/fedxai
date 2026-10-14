# Technology decisions

## Why FastAPI

Typed Python, native async, easy OpenAPI for the paper’s system section, same language as PyTorch.

## Why Flower

Standard FL research stack; simulation of five clients on one laptop; strategies map cleanly to FedAvg and FedProx.

## Why timm + torchvision

DenseNet from torchvision; EfficientNet, ViT, Swin from timm with pretrained weights for transfer learning without training from scratch.

## Why Opacus

PyTorch-native DP-SGD and accountants so ε, δ, clip, and noise are first-class experiment dimensions.

## Why FAISS default

Zero extra infra for a student demo. Qdrant remains a drop-in `VectorStoreBackend`.

## Why Hugging Face Inference API

No local 7B GPU requirement. Model ID is env-configurable. Offline fallback is mandatory.

## Why PostgreSQL for metadata

Relational audit, RBAC, experiment pointers. Pixel storage is out of scope for the central DB.

## Why not claim MedFusion

Unless architecture, training recipe, and weights match the paper, the module is **MedFusion-inspired diffusion augmentation**.

## Why shadcn later, Tailwind now

Phase 0 ships Tailwind tokens for a clinical look. shadcn/ui is added when Phase 15 implements real components, to avoid generating unused UI chrome.
