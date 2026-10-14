"""Grad-CAM for CNN backbones. Heatmaps are attributions, not radiographic proof."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


def _target_layer(model: torch.nn.Module, family: str):
    if hasattr(model, "features"):
        return model.features[-1]
    if hasattr(model, "layer4"):
        return model.layer4
    raise RuntimeError("Could not locate a convolutional target layer for Grad-CAM.")


def generate_gradcam(model: torch.nn.Module, images: torch.Tensor, class_index: int | None = None) -> torch.Tensor:
    model.eval()
    activations = []
    gradients = []

    def fwd_hook(_module, _inp, out):
        activations.append(out)

    def bwd_hook(_module, _grad_in, grad_out):
        gradients.append(grad_out[0])

    layer = _target_layer(model, "cnn")
    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)
    try:
        images = images.detach().requires_grad_(True)
        logits = model(images)
        if class_index is None:
            class_index = int(logits.argmax(dim=1)[0].item())
        score = logits[:, class_index].sum()
        model.zero_grad()
        score.backward()
        acts = activations[-1]
        grads = gradients[-1]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach()
    finally:
        h1.remove()
        h2.remove()


def save_heatmap_png(cam: torch.Tensor, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (cam[0, 0].cpu().numpy() * 255).clip(0, 255).astype("uint8")
    try:
        from PIL import Image

        Image.fromarray(array).save(path)
    except ImportError:
        path.with_suffix(".pt").write_bytes(b"")
        torch.save(cam.cpu(), path.with_suffix(".pt"))
        return path.with_suffix(".pt")
    return path
