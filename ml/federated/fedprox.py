"""FedProx: FedAvg aggregation plus a proximal term during local training."""

from __future__ import annotations

from typing import Mapping

import torch


def copy_global_params(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def fedprox_loss_term(
    model: torch.nn.Module,
    global_params: Mapping[str, torch.Tensor],
    mu: float,
) -> torch.Tensor:
    proximal = None
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        delta = parameter - global_params[name]
        term = torch.sum(delta * delta)
        proximal = term if proximal is None else proximal + term
    if proximal is None:
        return torch.tensor(0.0)
    return (mu / 2.0) * proximal
