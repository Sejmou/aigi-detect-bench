"""Robustness sweep: score a test split under each perturbation condition."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .data import Split, load_image
from .detectors.base import Detector
from .metrics import summarize
from .transforms import build_grid


def robustness_sweep(
    detector: Detector,
    split: Split,
    perturbations: dict[str, list[float]],
    fprs: list[float] | None = None,
    out_dir: str | Path = "outputs",
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(split.labels)
    grid = build_grid(perturbations)

    rows = []
    score_rows = []
    for name, intensity, tf in tqdm(grid, desc="conditions"):
        images = [tf(load_image(p)) for p in split.paths]
        scores = detector.scores(images)
        m = summarize(labels, scores, fprs)
        rows.append({"perturbation": name, "intensity": intensity, **m})
        for p, y, s in zip(split.paths, labels, scores):
            score_rows.append(
                {"path": str(p), "label": int(y), "perturbation": name,
                 "intensity": intensity, "score": float(s)}
            )

    curves = pd.DataFrame(rows)
    curves.to_csv(out_dir / "robustness_curves.csv", index=False)
    pd.DataFrame(score_rows).to_csv(out_dir / "scores.csv", index=False)
    _plot(curves, out_dir / "robustness_curves.png")
    return curves


def _plot(curves: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric = "auroc"
    clean = curves.loc[curves.perturbation == "clean", metric]
    clean_val = float(clean.iloc[0]) if len(clean) else float("nan")
    perts = [p for p in curves.perturbation.unique() if p != "clean"]
    n = max(1, len(perts))
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.2), squeeze=False)
    for ax, pert in zip(axes[0], perts):
        sub = curves[curves.perturbation == pert].sort_values("intensity")
        ax.plot(sub.intensity, sub[metric], marker="o", label=pert)
        if not np.isnan(clean_val):
            ax.axhline(clean_val, ls="--", c="gray", lw=1, label="clean")
        ax.set_title(pert)
        ax.set_xlabel("intensity")
        ax.set_ylabel(metric.upper())
        ax.set_ylim(0.4, 1.02)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
