"""FedAvg: weighted average of client parameter dicts. Images never leave clients."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch


def fedavg_aggregate(
    client_states: Sequence[Mapping[str, torch.Tensor]],
    client_sizes: Sequence[int] | None = None,
) -> dict[str, torch.Tensor]:
    if not client_states:
        raise ValueError("No client updates to aggregate.")
    weights = client_sizes or [1] * len(client_states)
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Client sizes must be positive.")
    aggregated: dict[str, torch.Tensor] = {}
    keys = client_states[0].keys()
    for key in keys:
        acc = None
        for state, size in zip(client_states, weights):
            tensor = state[key].detach().float()
            scaled = tensor * (size / total)
            acc = scaled if acc is None else acc + scaled
        assert acc is not None
        aggregated[key] = acc.to(dtype=client_states[0][key].dtype)
    return aggregated
