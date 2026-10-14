"""MedFusion-inspired diffusion module.

This is a research prototype backend, not a faithful original MedFusion
reproduction. Generated images MUST be labeled synthetic and must never
be mixed into the held-out global test set.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter
import random


def generate_synthetic_placeholder(
    source: Path,
    destination: Path,
    *,
    seed: int = 42,
    label: str = "synthetic",
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    image = Image.open(source).convert("L")
    image = ImageEnhance.Brightness(image).enhance(0.9 + rng.random() * 0.2)
    image = ImageEnhance.Contrast(image).enhance(0.9 + rng.random() * 0.2)
    if rng.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.6))
    image.save(destination)
    sidecar = destination.with_suffix(destination.suffix + ".synthetic.json")
    sidecar.write_text(
        '{"synthetic": true, "backend": "medfusion_inspired_placeholder", '
        f'"claim_original_medfusion": false, "label": "{label}"}}\n',
        encoding="utf-8",
    )
    return destination
