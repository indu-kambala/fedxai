"""Temperature scaling. Softmax after scaling is still not diagnostic certainty."""

from __future__ import annotations

import torch
import torch.nn as nn


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, steps: int = 50, lr: float = 0.01) -> float:
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=steps)
        criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except Exception:
            optimizer = torch.optim.Adam([self.temperature], lr=lr)
            for _ in range(steps):
                optimizer.zero_grad()
                loss = criterion(self.forward(logits), labels)
                loss.backward()
                optimizer.step()
        return float(self.temperature.detach().cpu().item())
