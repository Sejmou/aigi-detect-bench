"""Confidence calibration and operating-point selection.

Two artifacts come out of calibration:
  1. a temperature T so that sigmoid(score / T) is a calibrated P(fake), and
  2. a decision threshold achieving a target false-positive rate on the
     calibration split (which should match the deployment distribution,
     including recompressed/resized copies).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .metrics import expected_calibration_error, threshold_at_fpr, tpr_at_fpr


@dataclass
class Calibration:
    temperature: float
    threshold: float
    target_fpr: float
    achieved_tpr: float
    ece_before: float
    ece_after: float

    def to_dict(self) -> dict:
        return asdict(self)

    def prob(self, scores: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-scores / self.temperature))

    def decide(self, scores: np.ndarray) -> np.ndarray:
        return (scores >= self.threshold).astype(int)


def fit_temperature(labels: np.ndarray, scores: np.ndarray) -> float:
    """1-D temperature scaling by NLL grid + golden-section refinement."""

    def nll(t: float) -> float:
        z = np.clip(scores / t, -60.0, 60.0)
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean())

    ts = np.geomspace(0.05, 50.0, 60)
    t = float(ts[int(np.argmin([nll(t) for t in ts]))])
    lo, hi = t / 2, t * 2
    for _ in range(40):
        m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        if nll(m1) < nll(m2):
            hi = m2
        else:
            lo = m1
    return (lo + hi) / 2


def calibrate(labels: np.ndarray, scores: np.ndarray, target_fpr: float = 0.05) -> Calibration:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    t = fit_temperature(labels, scores)
    raw_p = 1.0 / (1.0 + np.exp(-scores))
    cal_p = 1.0 / (1.0 + np.exp(-scores / t))
    thr = threshold_at_fpr(labels, scores, target_fpr)
    return Calibration(
        temperature=float(t),
        threshold=float(thr),
        target_fpr=float(target_fpr),
        achieved_tpr=tpr_at_fpr(labels, scores, target_fpr),
        ece_before=expected_calibration_error(labels, raw_p),
        ece_after=expected_calibration_error(labels, cal_p),
    )
