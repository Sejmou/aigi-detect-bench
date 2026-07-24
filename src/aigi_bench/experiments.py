"""Experiments over the cached CLIP features: robustness, LOGO, cross-generator.

All of these are logistic regressions on frozen features, so once features.py
has run they cost seconds rather than GPU-hours. Three experiments:

  robustness      one probe trained on the train split, evaluated on the test
                  split under every perturbation condition. Answers "how much
                  does benign processing cost me?"

  logo            leave-one-generator-out: train on reals + five generators,
                  test on the held-out sixth. This is the deployment-realistic
                  number — you never have training data for tomorrow's model.

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
        return self.df.select(expr.alias("m"))["m"].to_numpy()

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


def run_logo(
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
                "auroc_heldout": out["auroc"],
                "auroc_indist": ind["auroc"],
                "gap": ind["auroc"] - out["auroc"],
                **{f"heldout_{k}": v for k, v in out.items() if k != "auroc"},
            }
        )
    return pl.DataFrame(rows)


def run_matrix(
    manifest_path: str | Path,
    cache_dir: str | Path,
    condition: str = "clean",
    paired_core_only: bool = True,
) -> pl.DataFrame:
    """Train on one generator, test on each. Rows = train, columns = test.

    Restricted to the paired core by default so every cell is scored on the
    same 2000 source covers — a difference between cells is then a generator
    difference, never a content difference.
    """
    c = load_corpus(manifest_path, cache_dir, condition)
    y_all = c.df["label"].to_numpy()
    gens = sorted(c.df["generator"].drop_nulls().unique().to_list())
    is_real = y_all == 0
    is_train = c.mask(pl.col("split") == "train")
    is_test = c.mask(pl.col("split") == "test")
    core = c.mask(pl.col("is_paired_core")) if paired_core_only else np.ones_like(is_test)

    rows = []
    for g_tr in gens:
        m_tr = is_train & (is_real | c.mask(pl.col("generator") == g_tr))
        probe = fit_probe(c.feats[m_tr], y_all[m_tr])
        row = {"train_on": g_tr, "n_train": int(m_tr.sum())}
        for g_te in gens:
            m_te = is_test & core & (is_real | c.mask(pl.col("generator") == g_te))
            row[g_te] = summarize(y_all[m_te], _scores(probe, c.feats[m_te]))["auroc"]
        rows.append(row)
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


def run_all(
    manifest_path: str | Path,
    cache_dir: str | Path,
    out_dir: str | Path,
    fprs: list[float] | None = None,
) -> dict[str, pl.DataFrame]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rob = run_robustness(manifest_path, cache_dir, fprs)
    logo = run_logo(manifest_path, cache_dir, fprs=fprs)
    mat = run_matrix(manifest_path, cache_dir)

    rob.write_csv(out_dir / "robustness.csv")
    logo.write_csv(out_dir / "logo.csv")
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
            },
            indent=2,
        )
    )
    return {"robustness": rob, "logo": logo, "matrix": mat}
