import numpy as np

from aigi_bench.calibrate import calibrate
from aigi_bench.metrics import auroc, threshold_at_fpr, tpr_at_fpr


def _fake_data(n=2000, sep=2.0, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
    scores = np.concatenate([rng.normal(0, 1, n // 2), rng.normal(sep, 1, n // 2)])
    return labels, scores


def test_auroc_orders_separability():
    l1, s1 = _fake_data(sep=0.5)
    l2, s2 = _fake_data(sep=3.0)
    assert auroc(l2, s2) > auroc(l1, s1) > 0.5


def test_threshold_respects_target_fpr():
    labels, scores = _fake_data()
    thr = threshold_at_fpr(labels, scores, 0.05)
    fpr = ((scores >= thr) & (labels == 0)).sum() / (labels == 0).sum()
    assert fpr <= 0.05 + 1e-9
    assert tpr_at_fpr(labels, scores, 0.05) > 0.5


def test_calibration_reduces_ece_for_miscalibrated_scores():
    labels, scores = _fake_data(sep=2.0)
    result = calibrate(labels, scores * 10, target_fpr=0.05)  # overconfident scores
    assert result.ece_after <= result.ece_before + 1e-6
    assert 0 < result.temperature
