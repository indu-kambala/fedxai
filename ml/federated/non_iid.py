"""Non-IID notes. Hospital folders already exist; this module does not reshuffle them."""

from pathlib import Path


def client_roots() -> list[tuple[str, Path]]:
    return [
        (f"hospital_{i}", Path("data") / f"hospital_{i}")
        for i in range(1, 6)
    ]
