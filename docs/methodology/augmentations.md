# Conservative chest X-ray augmentations

Augmentation runs **only on the training split**, after patient-/hospital-aware splitting.

| Operation | Default | Rationale |
| --- | --- | --- |
| Rotation | ≤10° | Small positioning variation |
| Horizontal flip | **Off** | Laterality can be clinically meaningful; enable only with written justification |
| Brightness / contrast | ±10% | Acquisition variation, not pathology simulation |
| Affine | ≤5°, translate 2% | Mild geometric jitter |
| Random crop | scale 0.90–1.00 | Avoid deleting lung fields |
| Gaussian noise | σ=0.01 | Mild sensor noise |
| Mixup / CutMix / heavy elastic | Not used | Can create non-physical anatomy |

Medically aggressive pipelines (random erasing of large lung regions, vertical flips, extreme color jitter) are disabled.

Each experiment records the augmentation set in MLflow so ablations stay comparable.
