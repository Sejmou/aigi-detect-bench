"""NPR — Neighboring Pixel Relationships (Tan et al., CVPR 2024).

A deliberately *low-level* counterpart to the CLIP probe. Where CLIP looks at
semantics, NPR looks at the upsampling fingerprint every convolutional
generator leaves behind: it feeds the network

    NPR = x - upsample(downsample(x, 1/2), 2)          [nearest neighbour]

which is near-zero wherever an image is locally smooth in the way a real sensor
is, and structured wherever a decoder synthesised detail. The backbone is a
ResNet-50 truncated after layer2 — the artifact is local, so depth buys
nothing.

Pairing it with the CLIP probe is the point: the two families fail
*differently*, which is what makes an ensemble worth building (see
`MeanEnsemble`). It is also the family most exposed to the tier-2
`noise_denoise` laundering, since median filtering directly attacks the pixel
residual this model reads.

The architecture is reconstructed here from torchvision primitives rather than
vendoring the authors' `networks/resnet.py`, so this repo carries no third-party
code. The checkpoint is theirs and keeps its own license — download it from
github.com/chuangchuangtan/NPR-DeepfakeDetection (`NPR.pth`, shipped in-repo).

Preprocessing matches the authors' `validate.py`: ImageNet normalisation and a
224 center crop, with no resize — resizing would resample the very signal the
model reads.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from tqdm import tqdm

from .base import Detector, register

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CROP = 224


class NPRNet(nn.Module):
    """ResNet-50 truncated after layer2, fed the NPR residual.

    Submodule names deliberately match the authors' checkpoint (`conv1`, `bn1`,
    `layer1`, `layer2`, `fc1`) so their state_dict loads strictly.
    """

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import resnet50

        r = resnet50(weights=None, num_classes=1)
        # The authors replace torchvision's 7x7 stem with a 3x3 stride-2 conv.
        # That is not incidental: the NPR residual is a high-frequency signal at
        # the 2-pixel scale, and a 7x7 receptive field would smear it before the
        # first nonlinearity. Keeping torchvision's stem silently breaks the
        # checkpoint (64x3x7x7 vs 64x3x3x3).
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = r.bn1
        self.relu, self.maxpool = r.relu, r.maxpool
        self.layer1, self.layer2 = r.layer1, r.layer2
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 1)

    @staticmethod
    def _npr(x: torch.Tensor) -> torch.Tensor:
        down = F.interpolate(x, scale_factor=0.5, mode="nearest", recompute_scale_factor=True)
        up = F.interpolate(down, scale_factor=2.0, mode="nearest", recompute_scale_factor=True)
        return x - up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The *2/3 rescale is the authors' — it puts the residual, whose range is
        # much smaller than an image's, back into the range the BN stats expect.
        x = self.conv1(self._npr(x) * 2.0 / 3.0)
        x = self.maxpool(self.relu(self.bn1(x)))
        x = self.layer2(self.layer1(x))
        return self.fc1(self.avgpool(x).flatten(1))


def _strip_prefix(sd: dict) -> dict:
    """Checkpoints saved under DataParallel carry a 'module.' prefix."""
    return {k.removeprefix("module."): v for k, v in sd.items()}


@register("npr")
class NPRDetector(Detector):
    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cuda",
        batch_size: int = 64,
        crop: int = CROP,
        **_: object,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.crop = crop
        self.model = NPRNet()

        sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        missing, unexpected = self.model.load_state_dict(_strip_prefix(sd), strict=False)
        # layer3/layer4 are absent by design; anything else missing is a real
        # architecture mismatch and should be loud.
        real_missing = [k for k in missing if not k.startswith(("layer3", "layer4"))]
        real_unexpected = [k for k in unexpected if not k.startswith(("layer3", "layer4"))]
        if real_missing or real_unexpected:
            raise RuntimeError(
                f"NPR checkpoint mismatch. missing={real_missing[:5]} "
                f"unexpected={real_unexpected[:5]}"
            )
        self.model.eval().to(self.device)

        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    def _prep(self, im: Image.Image) -> torch.Tensor:
        im = im.convert("RGB")
        w, h = im.size
        # Center crop with no resize: resampling would destroy the residual.
        if w > self.crop or h > self.crop:
            left, top = max(0, (w - self.crop) // 2), max(0, (h - self.crop) // 2)
            im = im.crop((left, top, left + min(w, self.crop), top + min(h, self.crop)))
        a = np.asarray(im, dtype=np.float32) / 255.0
        return torch.from_numpy(a).permute(2, 0, 1)

    @torch.no_grad()
    def scores(self, images: Sequence[Image.Image]) -> np.ndarray:
        out = []
        for i in tqdm(
            range(0, len(images), self.batch_size), desc="NPR", leave=False, unit="batch"
        ):
            batch = torch.stack([self._prep(im) for im in images[i : i + self.batch_size]])
            batch = ((batch.to(self.device) - self._mean) / self._std)
            out.append(self.model(batch).flatten().float().cpu().numpy())
        # Raw logits: higher = more fake, matching the Detector contract. The
        # authors apply sigmoid for accuracy; AUROC is invariant to it.
        return np.concatenate(out)
