"""UniversalFakeDetect-style baseline: frozen CLIP ViT-L/14 + logistic probe.

Ojha et al. (CVPR 2023) showed a linear probe on frozen CLIP features
generalizes across generators far better than end-to-end CNNs. This is the
cheapest strong baseline: feature extraction is the only GPU work, and the
probe trains in seconds on CPU.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from .base import Detector, register


@register("clip_linear")
class ClipLinearProbe(Detector):
    def __init__(
        self,
        clip_model: str = "ViT-L-14",
        clip_pretrained: str = "openai",
        batch_size: int = 64,
        device: str = "cuda",
        **_: object,
    ):
        import open_clip

        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained
        )
        self.model.eval().to(self.device)
        self.head: LogisticRegression | None = None

    @torch.no_grad()
    def _features(self, images: Sequence[Image.Image]) -> np.ndarray:
        feats = []
        for i in tqdm(range(0, len(images), self.batch_size), desc="CLIP features", leave=False):
            batch = torch.stack(
                [self.preprocess(im) for im in images[i : i + self.batch_size]]
            ).to(self.device)
            f = self.model.encode_image(batch)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.float().cpu().numpy())
        return np.concatenate(feats, axis=0)

    def fit(self, images: Sequence[Image.Image], labels: Sequence[int]) -> None:
        x = self._features(images)
        y = np.asarray(labels)
        self.head = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        self.head.fit(x, y)

    def scores(self, images: Sequence[Image.Image]) -> np.ndarray:
        if self.head is None:
            raise RuntimeError("Probe not fitted. Call fit() or load() first.")
        x = self._features(images)
        return self.head.decision_function(x)

    # -- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump(self.head, path)

    def load(self, path: str | Path) -> None:
        import joblib

        self.head = joblib.load(path)
