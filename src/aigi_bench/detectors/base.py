"""Detector interface. Implement `scores`; higher = more likely AI-generated."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from PIL import Image

REGISTRY: dict[str, type["Detector"]] = {}


def register(name: str):
    def _wrap(cls: type["Detector"]) -> type["Detector"]:
        REGISTRY[name] = cls
        cls.name = name
        return cls

    return _wrap


class Detector(ABC):
    """Minimal contract so heterogeneous detectors can be swept and ensembled."""

    name: str = "base"

    @abstractmethod
    def scores(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Return one raw score per image (higher = more fake). Shape (N,)."""

    def fit(self, images: Sequence[Image.Image], labels: Sequence[int]) -> None:
        """Optional: train a head on user data. Default: no-op (zero-shot)."""

    def probs(self, images: Sequence[Image.Image]) -> np.ndarray:
        """P(fake) via a sigmoid over raw scores. Override if scores are already
        probabilities."""
        return 1.0 / (1.0 + np.exp(-self.scores(images)))


class MeanEnsemble(Detector):
    """Score-level ensemble: mean of per-detector z-normalized scores.

    Simple but effective for combining heterogeneous families
    (frequency + CLIP-semantic + reconstruction).
    """

    name = "mean_ensemble"

    def __init__(self, detectors: Sequence[Detector]):
        self.detectors = list(detectors)

    def scores(self, images: Sequence[Image.Image]) -> np.ndarray:
        zs = []
        for d in self.detectors:
            s = d.scores(images).astype(np.float64)
            std = s.std() or 1.0
            zs.append((s - s.mean()) / std)
        return np.mean(zs, axis=0)
