"""Adapter for third-party detectors (NPR, AIDE, FreqNet, DRCT, ...).

Published detectors ship as (model definition, checkpoint, preprocessing).
Wrap them once here and they participate in every sweep and ensemble.

Example:

    from aigi_bench.detectors.external import TorchModuleDetector
    import torchvision.transforms as T

    # `build_npr()` would come from the authors' repo, vendored under third_party/
    det = TorchModuleDetector(
        module=build_npr(),
        checkpoint="checkpoints/npr.pth",
        preprocess=T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()]),
        logit_index=None,  # scalar output; use int for multi-class logits
    )
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .base import Detector, register


@register("torch_module")
class TorchModuleDetector(Detector):
    def __init__(
        self,
        module: torch.nn.Module,
        preprocess: Callable[[Image.Image], torch.Tensor],
        checkpoint: str | Path | None = None,
        logit_index: int | None = None,
        batch_size: int = 32,
        device: str = "cuda",
        strict: bool = True,
        **_: object,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.module = module
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu")
            state = state.get("model", state.get("state_dict", state))
            self.module.load_state_dict(state, strict=strict)
        self.module.eval().to(self.device)
        self.preprocess = preprocess
        self.logit_index = logit_index
        self.batch_size = batch_size

    @torch.no_grad()
    def scores(self, images: Sequence[Image.Image]) -> np.ndarray:
        out = []
        for i in range(0, len(images), self.batch_size):
            batch = torch.stack(
                [self.preprocess(im) for im in images[i : i + self.batch_size]]
            ).to(self.device)
            logits = self.module(batch)
            if logits.ndim > 1 and logits.shape[1] > 1:
                idx = self.logit_index if self.logit_index is not None else 1
                logits = logits[:, idx]
            out.append(logits.flatten().float().cpu().numpy())
        return np.concatenate(out, axis=0)
