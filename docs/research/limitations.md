# Limitations (must appear in UI, reports, and paper)

- Small and/or simulated hospital datasets; ~2500 images is a research scale, not a clinical trial.
- Domain shift across scanners, protocols, and hospitals.
- Label noise and non-IID client distributions.
- Synthetic diffusion images can contain artifacts and may not improve metrics.
- Pretrained backbones carry natural-image bias.
- Grad-CAM and attention maps can highlight non-causal regions.
- Softmax probability is not diagnostic certainty; calibration may be poor under shift.
- LLMs can hallucinate if guardrails fail; retrieval can miss or rank poorly.
- Differential privacy reduces utility; ε is not zero-leakage.
- No prospective clinical validation.
- **AI output is not a medical diagnosis.**
