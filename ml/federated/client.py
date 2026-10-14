"""Hospital client: trains only on its local folder. Never uploads images."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import Adam

from ml.datasets import create_hospital_dataloaders
from ml.training.trainer import train_one_epoch


class HospitalClient:
    def __init__(self, hospital_id: str, data_root: Path, batch_size: int = 8):
        self.hospital_id = hospital_id
        self.data_root = Path(data_root)
        self.batch_size = batch_size

    def local_train(
        self,
        model: torch.nn.Module,
        device: torch.device,
        *,
        local_epochs: int = 1,
        max_batches: int | None = 4,
        learning_rate: float = 1e-4,
        proximal_mu: float = 0.0,
        global_params=None,
    ) -> dict:
        if not self.data_root.exists():
            return {
                "hospital_id": self.hospital_id,
                "status": "skipped",
                "reason": f"Missing local data directory {self.data_root}",
                "n_samples": 0,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            }

        train_loader, _, _ = create_hospital_dataloaders(self.data_root, batch_size=self.batch_size)
        optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
        )
        last_loss = 0.0
        last_acc = 0.0
        samples = 0
        for _ in range(local_epochs):
            last_loss, last_acc, _, samples = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                max_batches=max_batches,
                proximal_mu=proximal_mu,
                global_params=global_params,
            )
        return {
            "hospital_id": self.hospital_id,
            "status": "trained",
            "train_loss": last_loss,
            "train_accuracy": last_acc,
            "n_samples": samples,
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        }
