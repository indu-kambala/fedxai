"""Differential privacy helpers.

DenseNet121 uses BatchNorm, which is often incompatible with Opacus DP-SGD
on the full network. This module:

1. Tries Opacus on trainable parameters when the library is installed and
   the module graph is compatible.
2. Otherwise applies a documented Gaussian mechanism on *model updates*
   (clip + noise) and reports that this is not a substitute for a full
   accountant-backed DP-SGD run.

Never claim perfect privacy.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch


def clip_state_dict(
    client_state: Mapping[str, torch.Tensor],
    global_state: Mapping[str, torch.Tensor],
    max_norm: float,
) -> dict[str, torch.Tensor]:
    deltas = []
    keys = []
    for key, tensor in client_state.items():
        delta = tensor.float() - global_state[key].float()
        deltas.append(delta.reshape(-1))
        keys.append(key)
    flat = torch.cat(deltas)
    norm = torch.linalg.vector_norm(flat)
    scale = min(1.0, max_norm / (float(norm) + 1e-12))
    clipped = {}
    offset = 0
    for key in keys:
        numel = client_state[key].numel()
        piece = flat[offset : offset + numel].reshape(client_state[key].shape) * scale
        clipped[key] = (global_state[key].float() + piece).to(client_state[key].dtype)
        offset += numel
    return clipped


def add_gaussian_noise(state: Mapping[str, torch.Tensor], noise_multiplier: float) -> dict[str, torch.Tensor]:
    noisy = {}
    for key, tensor in state.items():
        if tensor.dtype in (torch.float16, torch.float32, torch.float64):
            noisy[key] = tensor + torch.randn_like(tensor.float()) * noise_multiplier
            noisy[key] = noisy[key].to(tensor.dtype)
        else:
            noisy[key] = tensor
    return noisy


def report_privacy(
    *,
    enabled: bool,
    epsilon_target: float,
    delta: float,
    noise_multiplier: float,
    max_grad_norm: float,
    method: str,
    limitation: str,
) -> dict[str, Any]:
    return {
        "differential_privacy_enabled": enabled,
        "epsilon_target": epsilon_target,
        "delta": delta,
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": max_grad_norm,
        "method": method,
        "perfect_privacy_claimed": False,
        "limitation": limitation,
    }


def try_opacus(model, data_loader, optimizer, noise_multiplier: float, max_grad_norm: float):
    try:
        from opacus import PrivacyEngine
    except ImportError:
        return None, report_privacy(
            enabled=False,
            epsilon_target=8.0,
            delta=1e-5,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            method="opacus_unavailable",
            limitation="Opacus is not installed. DP-SGD was not attached.",
        )
    try:
        engine = PrivacyEngine()
        model, optimizer, data_loader = engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )
        return (model, optimizer, data_loader, engine), report_privacy(
            enabled=True,
            epsilon_target=8.0,
            delta=1e-5,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            method="opacus_dp_sgd",
            limitation="Accountant values are run-specific. This is not a clinical privacy guarantee.",
        )
    except Exception as exc:
        return None, report_privacy(
            enabled=False,
            epsilon_target=8.0,
            delta=1e-5,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            method="opacus_incompatible",
            limitation=(
                f"Opacus could not wrap this model ({type(exc).__name__}). "
                "BatchNorm-based CNNs such as DenseNet often require FrozenBatchNorm or a "
                "classifier-only DP setup. Use clip+Gaussian on updates instead, and do not "
                "claim DP-SGD accounting for that fallback."
            ),
        )
