# Data leakage prevention

1. **Split first.** Train / validation / test (and hospital partitions) are defined before any augmentation or diffusion sampling.
2. **Patient identity.** If identifiers exist, all images of a patient stay in one split. Never mix a patient across train and test.
3. **Hospital isolation.** Federated clients do not share raw files. Centralized baselines, if run, still respect patient-level splits.
4. **No test-time tuning.** Validation may guide early stopping and hyperparameters. Test is locked until final reporting.
5. **No synthetic test leakage.** Diffusion models train on **train** images only. Synthetic images never originate from validation/test.
6. **No pre-split oversampling of the full corpus** that could duplicate a patient into two splits.
7. **Model selection** uses validation (and pre-declared criteria), not test metrics.

Phase 1 will implement automated checks and refuse training if validation has not been run (`validation_required_before_training: true`).
