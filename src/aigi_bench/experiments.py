"""Experiments over cached CLIP features: robustness, held-out generator, matrix.

All of these are logistic regressions on frozen features, so once features.py
has run they cost seconds rather than GPU-hours. Three experiments:

  robustness      one probe trained on the train split, evaluated on the test
                  split under every perturbation condition. Answers "how much
                  does benign processing cost me?"

  leave-one-generator-out
                  train on reals + five generators, test on the held-out
                  sixth. This is the deployment-realistic number — you never
                  have training data for tomorrow's model.

  matrix          the full 6x6: train on one generator, test on each. Answers
                  which generator *families* share artifacts. Evaluated on the
                  paired core (the 2000 covers every model reconstructed), so
                  every cell sees the identical set of source images and a cell
                  difference cannot be a content difference.

Class balance: probes are fit with class_weight="balanced" because the corpus
is 1:1.53 real:fake overall and far more skewed once a single generator is
selected (reals stay ~9.9k while one generator contributes ~2k).

Usage:
    uv run aigi-bench experiments --config configs/default.yaml
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from .features import available_conditions, load_condition
from .metrics import summarize


@dataclass
class Corpus:
    """A manifest aligned to a feature matrix, row for row."""

    df: pl.DataFrame
    feats: np.ndarray

    def mask(self, expr: pl.Expr) -> np.ndarray:
        """Boolean row mask. Nulls become False.

        Necessary because `generator` is null for reals, so a predicate like
        `generator == "sdxl-turbo"` evaluates to null on every real row rather
        than False, and null does not support `~`.
        """
        return self.df.select(expr.fill_null(False).alias("m"))["m"].to_numpy().astype(bool)

    def subset(self, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.feats[m], self.df["label"].to_numpy()[m]


def load_corpus(manifest_path: str | Path, cache_dir: str | Path, condition: str) -> Corpus:
    """Load the manifest and one cached condition, aligned by path.

    The cache stores its own path vector, so alignment survives the manifest
    being re-sorted or filtered between the two runs.
    """
    df = pl.read_parquet(manifest_path)
    feats, paths = load_condition(cache_dir, condition)
    index = {p: i for i, p in enumerate(paths)}
    missing = [p for p in df["normalized_path"] if p not in index]
    if missing:
        raise KeyError(
            f"{len(missing)} manifest rows absent from cache '{condition}' "
            f"(first: {missing[0]}). Re-run `aigi-bench features`."
        )
    order = np.array([index[p] for p in df["normalized_path"]], dtype=np.int64)
    return Corpus(df=df, feats=feats[order])


def fit_probe(x: np.ndarray, y: np.ndarray, c: float = 1.0) -> LogisticRegression:
    return LogisticRegression(max_iter=2000, C=c, class_weight="balanced").fit(x, y)


def _scores(probe: LogisticRegression, x: np.ndarray) -> np.ndarray:
    return probe.decision_function(x)


def run_robustness(
    manifest_path: str | Path,
    cache_dir: str | Path,
    fprs: list[float] | None = None,
) -> pl.DataFrame:
    """Train once on clean train split; evaluate on test split under every condition.

    The probe is trained on *clean* features only. This is the honest setup: it
    measures how a detector trained on pristine data degrades in the wild, which
    is the question operators actually have. Training on perturbed data is a
    mitigation to evaluate separately, not the baseline.
    """
    clean = load_corpus(manifest_path, cache_dir, "clean")
    tr = clean.mask(pl.col("split") == "train")
    probe = fit_probe(clean.feats[tr], clean.df["label"].to_numpy()[tr])

    rows = []
    for cond in available_conditions(cache_dir):
        c = load_corpus(manifest_path, cache_dir, cond)
        te = c.mask(pl.col("split") == "test")
        y = c.df["label"].to_numpy()[te]
        s = _scores(probe, c.feats[te])
        rows.append({"condition": cond, "n": int(te.sum()), **summarize(y, s, fprs)})
    return pl.DataFrame(rows)


def run_leave_one_generator_out(
    manifest_path: str | Path,
    cache_dir: str | Path,
    condition: str = "clean",
    fprs: list[float] | None = None,
) -> pl.DataFrame:
    """Leave-one-generator-out. Train on reals + 5 generators, test on the 6th.

    Both the in-distribution number (test-split fakes of the five seen
    generators) and the held-out number are reported, because the gap between
    them *is* the generalization result — a held-out AUROC of 0.85 means
    something different if in-distribution is 0.87 than if it is 0.99.
    """
    c = load_corpus(manifest_path, cache_dir, condition)
    y_all = c.df["label"].to_numpy()
    gens = sorted(c.df["generator"].drop_nulls().unique().to_list())
    is_train = c.mask(pl.col("split") == "train")
    is_test = c.mask(pl.col("split") == "test")
    is_real = y_all == 0

    rows = []
    for held in gens:
        is_held = c.mask(pl.col("generator") == held)
        tr = is_train & (is_real | ~is_held)
        probe = fit_probe(c.feats[tr], y_all[tr])

        # held-out generator vs all test reals
        m_out = is_test & (is_real | is_held)
        s_out = _scores(probe, c.feats[m_out])
        out = summarize(y_all[m_out], s_out, fprs)

        # seen generators vs all test reals, same probe
        m_in = is_test & (is_real | ~is_held)
        s_in = _scores(probe, c.feats[m_in])
        ind = summarize(y_all[m_in], s_in, fprs)

        rows.append(
            {
                "held_out": held,
                "n_train": int(tr.sum()),
                "n_test_heldout_fake": int((m_out & (y_all == 1)).sum()),
                "auroc_heldout": out["auroc"],
                "auroc_indist": ind["auroc"],
                "gap": ind["auroc"] - out["auroc"],
                # Both sides of every metric, not just AUROC: the seen-vs-unseen
                # TPR gap at a fixed FPR is the number an operator feels.
                **{f"heldout_{k}": v for k, v in out.items() if k != "auroc"},
                **{f"indist_{k}": v for k, v in ind.items() if k != "auroc"},
            }
        )
    return pl.DataFrame(rows)


def run_matrix(
    manifest_path: str | Path,
    cache_dir: str | Path,
    condition: str = "clean",
    paired_core_only: bool = True,
    fprs: list[float] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Train on one generator, test on each. Returns (wide_auroc, long_metrics).

    Restricted to the paired core by default so every cell is scored on the
    same 2000 source covers — a difference between cells is then a generator
    difference, never a content difference.

    Two shapes come back because they answer different questions. The wide frame
    is one AUROC per cell, which is what a heatmap wants. The long frame keeps
    *every* metric per cell — TPR at each target FPR especially, since AUROC is
    a ranking summary and says nothing about the catch rate at a usable
    operating point. A cell can hold AUROC 0.95 while catching two thirds of
    fakes, and only the long frame shows that.
    """
    c = load_corpus(manifest_path, cache_dir, condition)
    y_all = c.df["label"].to_numpy()
    gens = sorted(c.df["generator"].drop_nulls().unique().to_list())
    is_real = y_all == 0
    is_train = c.mask(pl.col("split") == "train")
    is_test = c.mask(pl.col("split") == "test")
    core = c.mask(pl.col("is_paired_core")) if paired_core_only else np.ones_like(is_test)

    wide, long = [], []
    for g_tr in gens:
        m_tr = is_train & (is_real | c.mask(pl.col("generator") == g_tr))
        probe = fit_probe(c.feats[m_tr], y_all[m_tr])
        row = {"train_on": g_tr, "n_train": int(m_tr.sum())}
        for g_te in gens:
            m_te = is_test & core & (is_real | c.mask(pl.col("generator") == g_te))
            metrics = summarize(y_all[m_te], _scores(probe, c.feats[m_te]), fprs)
            row[g_te] = metrics["auroc"]
            long.append({
                "train_on": g_tr,
                "tested_on": g_te,
                "same_generator": g_tr == g_te,
                "n_train": int(m_tr.sum()),
                "n_test": int(m_te.sum()),
                "n_test_fake": int((m_te & ~is_real).sum()),
                **metrics,
            })
        wide.append(row)
    return pl.DataFrame(wide), pl.DataFrame(long)


def run_ensemble(
    manifest_path: str | Path,
    cache_dir: str | Path,
    score_cache_dir: str | Path,
    condition: str = "clean",
    fprs: list[float] | None = None,
) -> pl.DataFrame:
    """CLIP probe vs NPR vs their z-normalised mean, on the same test split.

    The ensemble is only interesting if the members disagree, so the Spearman
    correlation between the two score vectors is reported alongside: a mean of
    two highly-correlated detectors buys nothing, and a *negative* correlation
    means averaging actively destroys signal.
    """
    from scipy.stats import spearmanr

    from .features import load_scores

    c = load_corpus(manifest_path, cache_dir, condition)
    tr = c.mask(pl.col("split") == "train")
    te = c.mask(pl.col("split") == "test")
    y_all = c.df["label"].to_numpy()

    probe = fit_probe(c.feats[tr], y_all[tr])
    clip_s = _scores(probe, c.feats[te])

    raw, paths = load_scores(score_cache_dir, condition)
    index = {p: i for i, p in enumerate(paths)}
    order = np.array([index[p] for p in c.df["normalized_path"]], dtype=np.int64)
    npr_s = raw[order][te]

    y = y_all[te]

    def z(v: np.ndarray) -> np.ndarray:
        return (v - v.mean()) / (v.std() or 1.0)

    rho = float(spearmanr(clip_s, npr_s).statistic)
    rows = [
        {"detector": "clip_linear", **summarize(y, clip_s, fprs)},
        {"detector": "npr", **summarize(y, npr_s, fprs)},
        {"detector": "mean_ensemble", **summarize(y, z(clip_s) + z(npr_s), fprs)},
        # NPR runs below chance here, so the sign-corrected variant is reported
        # too: it is what an operator who *measured* the inversion would deploy.
        {"detector": "ensemble_npr_flipped", **summarize(y, z(clip_s) - z(npr_s), fprs)},
    ]
    return pl.DataFrame(rows).with_columns(pl.lit(rho).alias("spearman_clip_npr"))


def run_roc_curves(
    manifest_path: str | Path,
    cache_dir: str | Path,
    condition: str = "clean",
    paired_core_only: bool = True,
    n_grid: int = 120,
) -> pl.DataFrame:
    """ROC curve per (train_on, tested_on) cell, on a shared log-spaced FPR grid.

    Why a grid rather than raw ROC vertices: each cell's curve has as many
    vertices as there are distinct scores (thousands), the vertices fall at
    different FPRs per cell so curves cannot be compared row-wise, and the
    resulting frame would be tens of MB. Interpolating onto one grid makes the
    curves directly comparable and the output small enough to ship.

    The grid is **log-spaced from 1e-4**, because that is where the decision
    lives. Half the linear x-axis of an ROC plot covers FPR > 0.5 — operating
    points nobody would ever choose. Spacing by decade instead gives the
    low-FPR region the room it deserves; this is the DET-style convention from
    the detection and biometrics literature.

    Returns long-form: train_on, tested_on, fpr, tpr.
    """
    from sklearn.metrics import roc_curve

    c = load_corpus(manifest_path, cache_dir, condition)
    y_all = c.df["label"].to_numpy()
    gens = sorted(c.df["generator"].drop_nulls().unique().to_list())
    is_real = y_all == 0
    is_train = c.mask(pl.col("split") == "train")
    is_test = c.mask(pl.col("split") == "test")
    core = c.mask(pl.col("is_paired_core")) if paired_core_only else np.ones_like(is_test)

    # Include the two reporting points exactly so the curve and the table agree.
    grid = np.unique(
        np.concatenate([np.logspace(-4, 0, n_grid), np.array([0.01, 0.05])])
    )

    rows = []
    for g_tr in gens:
        m_tr = is_train & (is_real | c.mask(pl.col("generator") == g_tr))
        probe = fit_probe(c.feats[m_tr], y_all[m_tr])
        for g_te in gens:
            m_te = is_test & core & (is_real | c.mask(pl.col("generator") == g_te))
            fpr, tpr, _ = roc_curve(y_all[m_te], _scores(probe, c.feats[m_te]))
            # Step interpolation: at a given FPR budget the achievable TPR is the
            # curve's value at the last vertex at or below it, which is what
            # tpr_at_fpr reports. Linear interpolation would overstate it.
            idx = np.searchsorted(fpr, grid, side="right") - 1
            rows.append(
                pl.DataFrame({
                    "train_on": g_tr,
                    "tested_on": g_te,
                    "fpr": grid,
                    "tpr": tpr[np.clip(idx, 0, len(tpr) - 1)],
                })
            )
    return pl.concat(rows, how="vertical")


def run_pr_curves(
    manifest_path: str | Path,
    cache_dir: str | Path,
    condition: str = "clean",
    paired_core_only: bool = True,
    n_grid: int = 120,
) -> pl.DataFrame:
    """Precision-recall curve per cell, on a shared recall grid.

    Complements run_roc_curves rather than duplicating it: precision is what an
    operator experiences — of the images flagged, how many are actually fake.

    Measured caveat, and it matters for how these curves read: on the paired
    core the cells are **balanced**, prevalence ≈ 0.501, because every cover in
    the core has exactly one reconstruction per generator. So the PR chance
    baseline here is 0.5, not the low value an imbalanced set would give, and
    average precision tracks AUROC fairly closely. These curves are *not*
    showing the rare-positive regime.

    That regime is the operationally important one, and it is recoverable
    without new data: precision at any assumed prevalence p follows from the
    same TPR/FPR pair,

        precision = TPR·p / (TPR·p + FPR·(1-p))

    which is what `precision_at_prevalence` below computes. `prevalence` is
    returned per cell so the curve is always readable against its own baseline.

    Recall is the x-grid because it is the axis shared with TPR — recall *is*
    TPR — so a point read off the ROC plot can be located on this one.
    """
    from sklearn.metrics import precision_recall_curve

    c = load_corpus(manifest_path, cache_dir, condition)
    y_all = c.df["label"].to_numpy()
    gens = sorted(c.df["generator"].drop_nulls().unique().to_list())
    is_real = y_all == 0
    is_train = c.mask(pl.col("split") == "train")
    is_test = c.mask(pl.col("split") == "test")
    core = c.mask(pl.col("is_paired_core")) if paired_core_only else np.ones_like(is_test)

    grid = np.linspace(0.0, 1.0, n_grid)

    rows = []
    for g_tr in gens:
        m_tr = is_train & (is_real | c.mask(pl.col("generator") == g_tr))
        probe = fit_probe(c.feats[m_tr], y_all[m_tr])
        for g_te in gens:
            m_te = is_test & core & (is_real | c.mask(pl.col("generator") == g_te))
            y, s = y_all[m_te], _scores(probe, c.feats[m_te])
            prec, rec, _ = precision_recall_curve(y, s)
            # precision_recall_curve returns recall descending; flip for interp.
            order = np.argsort(rec)
            # Best achievable precision at or above each recall level: the curve
            # is not monotone, so a plain interpolation would understate it.
            interp = np.maximum.accumulate(prec[order][::-1])[::-1]
            idx = np.searchsorted(rec[order], grid, side="left")
            rows.append(
                pl.DataFrame({
                    "train_on": g_tr,
                    "tested_on": g_te,
                    "recall": grid,
                    "precision": interp[np.clip(idx, 0, len(interp) - 1)],
                    "prevalence": float(np.mean(y)),
                })
            )
    return pl.concat(rows, how="vertical")


def precision_at_prevalence(
    metrics: pl.DataFrame,
    prevalences: list[float] | None = None,
    fpr_col: str = "tpr@0.05fpr",
    fpr_value: float = 0.05,
) -> pl.DataFrame:
    """Precision each cell would achieve if fakes were rarer than in the test set.

    The evaluation sets here are 50/50 by construction, which is convenient for
    measurement and unlike any deployment. If only a small fraction of incoming
    images are generated, a fixed false-positive rate is applied to a much larger
    pool of reals, so false alarms swamp true hits and precision collapses even
    though TPR and FPR — and therefore AUROC — are unchanged.

    Bayes, with the operating point fixed:

        precision = TPR·p / (TPR·p + FPR·(1-p))

    Nothing is re-fitted; this is a re-reading of the same measured numbers at a
    different base rate.
    """
    prevalences = prevalences or [0.5, 0.1, 0.01, 0.001]
    rows = []
    for p in prevalences:
        for r in metrics.to_dicts():
            tpr = r[fpr_col]
            prec = (tpr * p) / (tpr * p + fpr_value * (1 - p)) if tpr > 0 else 0.0
            rows.append({
                "train_on": r["train_on"],
                "tested_on": r["tested_on"],
                "same_generator": r["same_generator"],
                "assumed_prevalence": p,
                "tpr": tpr,
                "fpr": fpr_value,
                "precision": prec,
            })
    return pl.DataFrame(rows)


def plot_matrix(matrix: pl.DataFrame, path: str | Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gens = [c for c in matrix.columns if c not in ("train_on", "n_train")]
    m = matrix.select(gens).to_numpy()
    fig, ax = plt.subplots(figsize=(1.1 * len(gens) + 3, 1.0 * len(gens) + 2))
    im = ax.imshow(m, vmin=0.5, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(gens)), gens, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix)), matrix["train_on"].to_list(), fontsize=8)
    ax.set_xlabel("tested on")
    ax.set_ylabel("trained on")
    ax.set_title(title, fontsize=10)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(
                j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if m[i, j] < 0.8 else "black",
            )
    fig.colorbar(im, ax=ax, label="AUROC", shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_robustness(curves: pl.DataFrame, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clean = curves.filter(pl.col("condition") == "clean")["auroc"]
    clean_val = float(clean[0]) if len(clean) else float("nan")
    d = curves.filter(pl.col("condition") != "clean").sort("condition")
    fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(d)), 4))
    ax.bar(d["condition"].to_list(), d["auroc"].to_list(), color="#4C72B0")
    if not np.isnan(clean_val):
        ax.axhline(clean_val, ls="--", c="gray", lw=1, label=f"clean ({clean_val:.3f})")
        ax.legend(fontsize=8)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.02)
    ax.tick_params(axis="x", rotation=60, labelsize=7)
    ax.set_title("Robustness under benign processing (probe trained on clean)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_calibration(
    manifest_path: str | Path,
    cache_dir: str | Path,
    targets: list[float] | None = None,
    conditions: list[str] | None = None,
) -> pl.DataFrame:
    """Fit a threshold on calib, then report what it actually does on test.

    Two questions, both operational. First, does a target FPR chosen on the
    calibration split hold on unseen data (it should, given splits are grouped
    by cover). Second, does a threshold tuned on *clean* images survive being
    applied to processed ones without retuning — the README warns it will
    over-fire, and whether that warning binds is worth measuring rather than
    assuming.
    """
    from .calibrate import calibrate

    targets = targets or [0.01, 0.05]
    c = load_corpus(manifest_path, cache_dir, "clean")
    y = c.df["label"].to_numpy()
    tr, ca, te = (
        c.mask(pl.col("split") == s) for s in ("train", "calib", "test")
    )
    probe = fit_probe(c.feats[tr], y[tr])
    s_ca = _scores(probe, c.feats[ca])

    rows = []
    for target in targets:
        cal = calibrate(y[ca], s_ca, target)
        for cond in conditions or ["clean"]:
            cc = load_corpus(manifest_path, cache_dir, cond)
            s = _scores(probe, cc.feats[te])
            yt = y[te]
            rows.append({
                "target_fpr": target,
                "condition": cond,
                "threshold": cal.threshold,
                "temperature": cal.temperature,
                "ece_before": cal.ece_before,
                "ece_after": cal.ece_after,
                "realized_fpr": float(((s >= cal.threshold) & (yt == 0)).sum() / (yt == 0).sum()),
                "realized_tpr": float(((s >= cal.threshold) & (yt == 1)).sum() / (yt == 1).sum()),
            })
    return pl.DataFrame(rows)


def run_all(
    manifest_path: str | Path,
    cache_dir: str | Path,
    out_dir: str | Path,
    fprs: list[float] | None = None,
    score_cache_dir: str | Path | None = None,
) -> dict[str, pl.DataFrame]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rob = run_robustness(manifest_path, cache_dir, fprs)
    heldout = run_leave_one_generator_out(manifest_path, cache_dir, fprs=fprs)
    mat, mat_long = run_matrix(manifest_path, cache_dir, fprs=fprs)
    mat_long.write_csv(out_dir / "cross_generator_metrics.csv")
    roc = run_roc_curves(manifest_path, cache_dir)
    roc.write_parquet(out_dir / "roc_curves.parquet")
    prc = run_pr_curves(manifest_path, cache_dir)
    prc.write_parquet(out_dir / "pr_curves.parquet")
    prev = precision_at_prevalence(mat_long)
    prev.write_csv(out_dir / "precision_at_prevalence.csv")

    avail = available_conditions(cache_dir)
    cal_conds = [c for c in ("clean", "jpeg_40", "blur_2", "social_55") if c in avail]
    cal = run_calibration(manifest_path, cache_dir, fprs, cal_conds)
    cal.write_csv(out_dir / "calibration.csv")

    out = {
        "robustness": rob,
        "leave_one_generator_out": heldout,
        "matrix": mat,
        "matrix_metrics": mat_long,
        "roc_curves": roc,
        "pr_curves": prc,
        "precision_at_prevalence": prev,
        "calibration": cal,
    }
    if score_cache_dir and Path(score_cache_dir).exists():
        ens = run_ensemble(manifest_path, cache_dir, score_cache_dir, fprs=fprs)
        ens.write_csv(out_dir / "ensemble.csv")
        out["ensemble"] = ens

    rob.write_csv(out_dir / "robustness.csv")
    heldout.write_csv(out_dir / "leave_one_generator_out.csv")
    mat.write_csv(out_dir / "cross_generator_matrix.csv")
    plot_robustness(rob, out_dir / "robustness.png")
    plot_matrix(mat, out_dir / "cross_generator_matrix.png",
                "Cross-generator AUROC (paired core, clean)")

    (out_dir / "experiments_meta.json").write_text(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "cache_dir": str(cache_dir),
                "conditions": available_conditions(cache_dir),
                "n_conditions": len(available_conditions(cache_dir)),
            },
            indent=2,
        )
    )
    return out
