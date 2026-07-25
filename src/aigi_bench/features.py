"""Cache CLIP features for every (image, perturbation condition) pair.

Everything downstream of this file — the leave-one-generator-out matrix,
robustness curves, calibration, ensembling — is a *linear* operation on frozen
CLIP features. Only the feature extraction touches the GPU, and it is by far
the dominant cost. Extracting once and reusing turns the rest into seconds of
NumPy.

Concretely: the leave-one-generator-out experiment trains six probes and
evaluates each under 16
perturbation conditions. Done naively that is 6 x 16 GPU passes over the
corpus. Because the probe is a logistic regression on frozen features, the
features do not depend on which fold is being trained, so one pass over
(corpus x conditions) serves all six folds.

Cost here: ~25k images x 16 conditions = ~400k forward passes, ~1.2 GB of
float32 at 768 dims. Perturbations are applied on CPU in a DataLoader worker
pool, because JPEG round-trips and Gaussian blur over 400k images are
themselves expensive enough to starve the GPU if done inline.

Layout on disk (one .npz per condition, so a partial run is still usable and
adding a condition does not invalidate the rest):

    <cache_dir>/<condition>.npz   feats (N, D) float32, paths (N,) str

Usage:
    uv run aigi-bench features --config configs/default.yaml
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .transforms import Transform, build_grid


def condition_key(name: str, intensity: float | None) -> str:
    """Filesystem-safe key for a (perturbation, intensity) pair."""
    if intensity is None:
        return name
    return f"{name}_{intensity:g}".replace(".", "p").replace("-", "m")


class _PerturbedImages(Dataset):
    """Loads an image, applies one perturbation, applies CLIP preprocessing.

    Lives in a DataLoader worker so the JPEG/blur/resize work overlaps with GPU
    compute instead of serialising against it.
    """

    def __init__(self, paths: Sequence[str], transform: Transform, preprocess):
        self.paths = list(paths)
        self.transform = transform
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
            return self.preprocess(self.transform(im))


@torch.no_grad()
def extract_condition(
    model,
    preprocess,
    paths: Sequence[str],
    transform: Transform,
    device: str,
    batch_size: int = 64,
    num_workers: int = 8,
    desc: str = "",
) -> np.ndarray:
    """L2-normalised CLIP image embeddings for `paths` under one perturbation."""
    loader = DataLoader(
        _PerturbedImages(paths, transform, preprocess),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        shuffle=False,
    )
    out = []
    use_amp = device == "cuda"
    for batch in tqdm(loader, desc=desc, unit="batch", leave=False):
        batch = batch.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            f = model.encode_image(batch)
        f = f.float()
        f = f / f.norm(dim=-1, keepdim=True)
        out.append(f.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def build_feature_cache(
    manifest_path: str | Path,
    cache_dir: str | Path,
    perturbations: dict[str, list[float]],
    clip_model: str = "ViT-L-14",
    clip_pretrained: str = "openai",
    device: str = "cuda",
    batch_size: int = 64,
    num_workers: int = 8,
    path_column: str = "normalized_path",
) -> Path:
    """Extract and cache features for every condition. Skips conditions already done."""
    import open_clip

    manifest_path, cache_dir = Path(manifest_path), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(manifest_path)
    paths = df[path_column].to_list()

    device = device if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model, pretrained=clip_pretrained
    )
    model.eval().to(device)

    grid = build_grid(perturbations)
    written = []
    for name, intensity, tf in grid:
        key = condition_key(name, intensity)
        dst = cache_dir / f"{key}.npz"
        if dst.exists():
            written.append(key)
            continue
        feats = extract_condition(
            model, preprocess, paths, tf, device,
            batch_size=batch_size, num_workers=num_workers,
            desc=f"{key} ({len(grid)} conds)",
        )
        # Paths are stored alongside so a cache file is self-describing and can
        # be re-aligned to a manifest that has since been re-sorted.
        np.savez(dst, feats=feats, paths=np.array(paths, dtype=object))
        written.append(key)

    meta = {
        "built_at": datetime.now(UTC).isoformat(),
        "manifest": str(manifest_path),
        "path_column": path_column,
        "clip_model": clip_model,
        "clip_pretrained": clip_pretrained,
        "device": device,
        "n_images": len(paths),
        "conditions": written,
        "n_conditions": len(written),
    }
    (cache_dir / "cache_meta.json").write_text(json.dumps(meta, indent=2))
    return cache_dir


class _PerturbedRaw(Dataset):
    """Like _PerturbedImages but hands back a PIL image (no CLIP preprocessing).

    Detectors other than the CLIP probe bring their own preprocessing, so the
    perturbation still belongs in a worker but the normalisation does not.
    """

    def __init__(self, paths: Sequence[str], transform: Transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            return self.transform(im.convert("RGB"))


def build_score_cache(
    detector,
    manifest_path: str | Path,
    cache_dir: str | Path,
    perturbations: dict[str, list[float]],
    batch_size: int = 256,
    path_column: str = "normalized_path",
) -> Path:
    """Cache scalar detector scores per condition, mirroring the feature cache.

    Used for detectors that are not a linear head on shared features (NPR), so
    the ensemble and the robustness sweep can read them back without re-running
    the model once per experiment.
    """
    manifest_path, cache_dir = Path(manifest_path), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(manifest_path)
    paths = df[path_column].to_list()

    for name, intensity, tf in build_grid(perturbations):
        key = condition_key(name, intensity)
        dst = cache_dir / f"{key}.npz"
        if dst.exists():
            continue
        scores = []
        for i in tqdm(
            range(0, len(paths), batch_size), desc=f"{detector.name} {key}", leave=False
        ):
            chunk = paths[i : i + batch_size]
            imgs = []
            for p in chunk:
                with Image.open(p) as im:
                    imgs.append(tf(im.convert("RGB")))
            scores.append(detector.scores(imgs))
        np.savez(
            dst,
            scores=np.concatenate(scores).astype(np.float32),
            paths=np.array(paths, dtype=object),
        )
    return cache_dir


def load_scores(cache_dir: str | Path, key: str) -> tuple[np.ndarray, list[str]]:
    d = np.load(Path(cache_dir) / f"{key}.npz", allow_pickle=True)
    return d["scores"], list(d["paths"])


def load_condition(cache_dir: str | Path, key: str) -> tuple[np.ndarray, list[str]]:
    """Load one cached condition as (features, paths)."""
    d = np.load(Path(cache_dir) / f"{key}.npz", allow_pickle=True)
    return d["feats"], list(d["paths"])


def available_conditions(cache_dir: str | Path) -> list[str]:
    return sorted(p.stem for p in Path(cache_dir).glob("*.npz"))
