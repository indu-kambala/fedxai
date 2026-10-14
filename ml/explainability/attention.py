"""Transformer attention visualization. Not a clinical finding."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def attention_rollout_from_weights(attn_weights: list[torch.Tensor]) -> torch.Tensor:
    """attn_weights: list of (batch, heads, tokens, tokens)."""
    result = None
    for layer in attn_weights:
        avg = layer.mean(dim=1)
        identity = torch.eye(avg.size(-1), device=avg.device).unsqueeze(0)
        aug = avg + identity
        aug = aug / aug.sum(dim=-1, keepdim=True)
        result = aug if result is None else torch.bmm(aug, result)
    assert result is not None
    # CLS token attention to patches when present
    cls_map = result[:, 0, 1:]
    side = int(cls_map.size(-1) ** 0.5)
    if side * side == cls_map.size(-1):
        grid = cls_map.reshape(cls_map.size(0), 1, side, side)
        return F.interpolate(grid, size=(224, 224), mode="bilinear", align_corners=False)
    return cls_map


def vit_attention_map(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Best-effort last-layer attention. Falls back to input-gradient saliency."""
    captured: list[torch.Tensor] = []

    def hook(_module, _inp, out):
        if isinstance(out, tuple) and len(out) > 1 and torch.is_tensor(out[1]):
            captured.append(out[1].detach())

    handles = []
    for module in model.modules():
        if module.__class__.__name__.endswith("Attention"):
            handles.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            _ = model(images)
        if captured:
            return attention_rollout_from_weights(captured[-1:])
    finally:
        for handle in handles:
            handle.remove()
    images = images.detach().requires_grad_(True)
    logits = model(images)
    pred = logits.argmax(dim=1)
    logits[range(logits.size(0)), pred].sum().backward()
    sal = images.grad.abs().mean(dim=1, keepdim=True)
    sal = sal - sal.min()
    return sal / (sal.max() + 1e-8)
