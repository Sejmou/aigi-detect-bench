"""Folder-based datasets: data/real/** = label 0, data/fake/** = label 1."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)


@dataclass
class Split:
    paths: list[Path]
    labels: list[int]  # 0 = real, 1 = fake

    def __len__(self) -> int:
        return len(self.paths)


def make_splits(
    real_dir: str | Path,
    fake_dir: str | Path,
    train_frac: float = 0.6,
    seed: int = 42,
    max_images_per_class: int | None = None,
) -> dict[str, Split]:
    """Deterministic train / calib / test splits.

    Remainder after train is divided evenly between calibration and test.
    """
    rng = random.Random(seed)
    splits: dict[str, tuple[list[Path], list[int]]] = {
        "train": ([], []),
        "calib": ([], []),
        "test": ([], []),
    }
    for label, root in ((0, real_dir), (1, fake_dir)):
        paths = list_images(root)
        rng.shuffle(paths)
        if max_images_per_class:
            paths = paths[:max_images_per_class]
        n = len(paths)
        n_train = int(n * train_frac)
        n_calib = (n - n_train) // 2
        chunks = {
            "train": paths[:n_train],
            "calib": paths[n_train : n_train + n_calib],
            "test": paths[n_train + n_calib :],
        }
        for name, chunk in chunks.items():
            splits[name][0].extend(chunk)
            splits[name][1].extend([label] * len(chunk))
    return {k: Split(paths=v[0], labels=v[1]) for k, v in splits.items()}


def load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")
