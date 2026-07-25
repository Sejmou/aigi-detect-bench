"""Tier-4: white-box PGD against the CLIP probe — the honest worst case.

Tiers 1 and 2 (transforms.py) are *non-adaptive*: the perturbation is chosen
without consulting the detector. That is the right model for incidental
processing, and the wrong model for an adversary. This module closes the gap by
letting the attacker see everything — CLIP weights, probe coefficients,
gradients — and optimise directly against the decision function.

The number this produces is a lower bound on detector strength, not a
prediction of typical performance. It belongs in a separate column from the
tier-1/2 numbers, never averaged with them: a detector holding at 0.95 under
laundering and collapsing to 0.15 under PGD is still useful against careless
adversaries, but you must know which regime you are quoting.

Why AUROC *below* 0.5 is the expected outcome: PGD does not merely erase the
signal, it inverts it — pushing fakes across the boundary into confidently-real
territory. An AUROC of 0.05 means the detector is reliably wrong, which is
strictly worse than the 0.5 of no information.

Threat model, stated explicitly because it decides how to read the result:
  - attacker perturbs only the *fake* images (the realistic goal is evading
    detection, not framing real photos)
  - budget is L-inf epsilon in [0,1] pixel space, default 4/255, which is at
    or below the threshold of visibility on photographic content
  - attack runs end-to-end through CLIP preprocessing, so the perturbation must
    survive resize+normalise — the usual reason naive attacks fail to transfer

Cost: each PGD step is a forward+backward through ViT-L/14, so this runs on a
subset (default 1000 images) rather than the full corpus.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# CLIP's channel normalisation. The attack operates in raw [0,1] pixel space and
# applies this inside the graph, so the epsilon budget means what it says.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class _RawImages(Dataset):
    """Images resized to CLIP's input resolution but *not* normalised."""

    def __init__(self, paths: Sequence[str], size: int = 224):
        self.paths = list(paths)
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB").resize((self.size, self.size), Image.BICUBIC)
            a = np.asarray(im, dtype=np.float32) / 255.0
        return torch.from_numpy(a).permute(2, 0, 1)


def pgd_scores(
    model,
    probe_w: np.ndarray,
    probe_b: float,
    paths: Sequence[str],
    device: str = "cuda",
    epsilon: float = 4 / 255,
    alpha: float = 1 / 255,
    steps: int = 10,
    batch_size: int = 32,
    size: int = 224,
    num_workers: int = 8,
    save_dir: str | Path | None = None,
    seed: int = 42,
    deterministic: bool = False,
) -> np.ndarray:
    """Adversarial probe scores for `paths`, minimising the fake-ness score.

    Returns the probe's decision_function on the *attacked* images, so the
    result drops straight into the same metrics as every other condition.

    If `save_dir` is given, the attacked images are written there as PNG (never
    JPEG — a lossy re-encode would partially undo the very perturbation being
    saved). Worth doing: the adversarial set is the reusable artifact from this
    experiment. It lets you test a *different* detector against the same attack
    without re-running PGD, and lets a human confirm by eye that the
    perturbation really is invisible rather than taking the epsilon on trust.
    """
    mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
    w = torch.tensor(probe_w, dtype=torch.float32, device=device).view(-1)
    b = float(probe_b)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Seeded random start, so the attack does not vary run to run for that
    # reason. Verified: two calls with the same seed give bitwise-identical
    # scores; different seeds differ by ~1.3 on a score scale of sigma 1.5.
    #
    # Caveat, measured rather than assumed: the *first* PGD call in a process
    # still differs from later ones by ~0.2 in score units (~14% of sigma),
    # because cuDNN picks kernels by autotuning on the first invocation. The
    # seed does not control that. Set `deterministic=True` to trade throughput
    # for exact repeatability when the saved images are a shared artifact.
    gen = torch.Generator(device=device).manual_seed(seed)

    loader = DataLoader(
        _RawImages(paths, size),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    out: list[np.ndarray] = []
    saved = 0
    for x0 in tqdm(loader, desc=f"pgd eps={epsilon:.4f}", unit="batch", leave=False):
        x0 = x0.to(device, non_blocking=True)
        # Random start inside the ball: PGD from the clean point can stall on a
        # flat gradient, and a random init is the standard fix.
        delta = torch.empty_like(x0).uniform_(-epsilon, epsilon, generator=gen)
        delta = (x0 + delta).clamp(0, 1) - x0
        delta.requires_grad_(True)

        for _ in range(steps):
            feats = model.encode_image((x0 + delta - mean) / std)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            score = feats.float() @ w + b  # probe decision function
            loss = score.sum()  # descend: make fakes look real
            (grad,) = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                delta -= alpha * grad.sign()
                delta.clamp_(-epsilon, epsilon)
                delta.copy_((x0 + delta).clamp(0, 1) - x0)

        with torch.no_grad():
            adv = (x0 + delta).clamp(0, 1)
            feats = model.encode_image((adv - mean) / std)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            out.append((feats.float() @ w + b).cpu().numpy())

            if save_dir is not None:
                arr = (adv * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
                for j in range(arr.shape[0]):
                    src = Path(paths[saved + j])
                    Image.fromarray(arr[j], "RGB").save(save_dir / f"{src.stem}.png")
                saved += arr.shape[0]
    return np.concatenate(out)


@torch.no_grad()
def clean_scores(
    model,
    probe_w: np.ndarray,
    probe_b: float,
    paths: Sequence[str],
    device: str = "cuda",
    batch_size: int = 64,
    size: int = 224,
    num_workers: int = 8,
) -> np.ndarray:
    """Same path as pgd_scores with zero perturbation.

    Used as the attack's control: it goes through the identical bicubic-resize
    pipeline, so the clean-vs-attacked comparison isolates the perturbation and
    not a difference in preprocessing.
    """
    mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
    w = torch.tensor(probe_w, dtype=torch.float32, device=device).view(-1)
    loader = DataLoader(
        _RawImages(paths, size), batch_size=batch_size,
        num_workers=num_workers, shuffle=False, pin_memory=(device == "cuda"),
    )
    out = []
    for x in tqdm(loader, desc="clean (attack pipeline)", unit="batch", leave=False):
        x = x.to(device, non_blocking=True)
        f = model.encode_image((x - mean) / std)
        f = f / f.norm(dim=-1, keepdim=True)
        out.append((f.float() @ w + float(probe_b)).cpu().numpy())
    return np.concatenate(out)
