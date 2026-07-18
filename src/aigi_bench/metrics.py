"""Evaluation metrics. Convention: label 1 = fake, higher score = more fake."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    """TPR at the largest threshold whose FPR <= target_fpr."""
    fpr, tpr, _ = roc_curve(labels, scores)
    ok = fpr <= target_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def threshold_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    """Score threshold achieving FPR closest to (and not above) target_fpr."""
    fpr, _, thr = roc_curve(labels, scores)
    ok = np.where(fpr <= target_fpr)[0]
    idx = ok[-1] if len(ok) else 0
    return float(thr[idx])


def balanced_accuracy(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    return float(balanced_accuracy_score(labels, (scores >= threshold).astype(int)))


def expected_calibration_error(
    labels: np.ndarray, probs: np.ndarray, n_bins: int = 15
) -> float:
    """ECE over predicted P(fake). `probs` must be in [0, 1]."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        if not mask.any():
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


def summarize(
    labels: np.ndarray, scores: np.ndarray, fprs: list[float] | None = None
) -> dict[str, float]:
    fprs = fprs or [0.01, 0.05]
    out = {
        "auroc": auroc(labels, scores),
        "balanced_acc@0.5": balanced_accuracy(labels, scores, 0.5),
    }
    for f in fprs:
        out[f"tpr@{f:g}fpr"] = tpr_at_fpr(labels, scores, f)
    return out
