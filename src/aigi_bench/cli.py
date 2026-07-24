"""CLI: `uv run aigi-bench {fit,eval,calibrate} --config configs/default.yaml`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from . import detectors
from .calibrate import calibrate as run_calibrate
from .data import load_image, make_splits
from .eval import robustness_sweep
from .experiments import run_all
from .features import build_feature_cache
from .manifest import build_manifest
from .normalize import normalize_corpus


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build(cfg: dict):
    dcfg = dict(cfg["detector"])
    name = dcfg.pop("name")
    det = detectors.build(name, **dcfg)
    return det


def _splits(cfg: dict):
    d = cfg["data"]
    return make_splits(
        d["real_dir"], d["fake_dir"],
        train_frac=d.get("train_frac", 0.6),
        seed=d.get("seed", 42),
        max_images_per_class=d.get("max_images_per_class"),
    )


def cmd_manifest(cfg: dict) -> None:
    m = cfg["manifest"]
    d = cfg["data"]
    df = build_manifest(
        real_dir=m["real_dir"],
        slop_root=m["slop_root"],
        meta_parquet=m["meta_parquet"],
        out_dir=m.get("out_dir", cfg["eval"]["out_dir"]),
        seed=d.get("seed", 42),
        train_frac=d.get("train_frac", 0.6),
        probe=m.get("probe_dimensions", True),
    )
    import polars as pl

    print(f"{df.height} rows -> {Path(m.get('out_dir', 'outputs')) / 'manifest.parquet'}")
    print(
        df.group_by("label", "generator", "prompt_variant")
        .agg(pl.len().alias("n"), pl.col("spotify_id").n_unique().alias("ids"))
        .sort("label", "generator", nulls_last=False)
    )
    print(df.group_by("split").agg(pl.len().alias("n")).sort("split"))


def cmd_normalize(cfg: dict) -> None:
    import polars as pl

    n = cfg["normalize"]
    out_dir = Path(cfg["manifest"].get("out_dir", cfg["eval"]["out_dir"]))
    df = normalize_corpus(
        manifest_path=out_dir / "manifest.parquet",
        out_root=n["out_root"],
        seed=cfg["data"].get("seed", 42),
        edge=n.get("edge", 512),
        workers=n.get("workers"),
    )
    print(f"{df.height} normalized -> {n['out_root']}")
    print(df.group_by("label").agg(pl.len().alias("n")).sort("label"))


def cmd_features(cfg: dict) -> None:
    f, d = cfg["features"], cfg["detector"]
    out_dir = Path(cfg["manifest"].get("out_dir", cfg["eval"]["out_dir"]))
    cache = build_feature_cache(
        manifest_path=out_dir / "manifest_normalized.parquet",
        cache_dir=f["cache_dir"],
        perturbations=cfg.get("perturbations", {}),
        clip_model=d.get("clip_model", "ViT-L-14"),
        clip_pretrained=d.get("clip_pretrained", "openai"),
        device=d.get("device", "cuda"),
        batch_size=d.get("batch_size", 64),
        num_workers=f.get("num_workers", 8),
    )
    print(f"feature cache -> {cache}")


def cmd_npr_scores(cfg: dict) -> None:
    """Cache NPR scores per condition so the ensemble does not re-run the model."""
    from .features import build_score_cache

    n = cfg["npr"]
    det = detectors.build(
        "npr", checkpoint=n["checkpoint"], device=cfg["detector"].get("device", "cuda")
    )
    out_dir = Path(cfg["manifest"].get("out_dir", cfg["eval"]["out_dir"]))
    cache = build_score_cache(
        det,
        out_dir / "manifest_normalized.parquet",
        n["score_cache_dir"],
        cfg.get("perturbations", {}) if n.get("all_conditions") else {},
    )
    print(f"NPR score cache -> {cache}")


def cmd_attack(cfg: dict) -> None:
    """Tier-4 white-box PGD against the fitted probe (see attacks.py)."""
    import open_clip
    import torch

    from .attacks import clean_scores, pgd_scores
    from .experiments import fit_probe, load_corpus
    from .metrics import summarize

    a, d = cfg["attack"], cfg["detector"]
    out_dir = Path(cfg["manifest"].get("out_dir", cfg["eval"]["out_dir"]))
    manifest = out_dir / "manifest_normalized.parquet"
    cache = cfg["features"]["cache_dir"]

    c = load_corpus(manifest, cache, "clean")
    y_all = c.df["label"].to_numpy()
    probe = fit_probe(c.feats[c.mask(pl_expr_split("train"))], y_all[c.mask(pl_expr_split("train"))])
    w, b = probe.coef_.ravel(), float(probe.intercept_[0])

    import polars as pl

    te = c.df.with_row_index("i").filter(pl.col("split") == "test")
    n = a.get("n_per_class", 500)
    sub = pl.concat([
        te.filter(pl.col("label") == 0).sample(n, seed=11),
        te.filter(pl.col("label") == 1).sample(n, seed=11),
    ])
    paths, y = sub["normalized_path"].to_list(), sub["label"].to_numpy()

    model, _, _ = open_clip.create_model_and_transforms(
        d.get("clip_model", "ViT-L-14-quickgelu"), pretrained=d.get("clip_pretrained", "openai")
    )
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    for p in model.parameters():
        p.requires_grad_(False)

    base = clean_scores(model, w, b, paths)
    rows = [{"condition": "clean", "epsilon_255": 0.0, **summarize(y, base)}]
    fake_paths = [p for p, lab in zip(paths, y, strict=True) if lab == 1]
    for eps255 in a.get("epsilons_255", [1, 2, 4, 8]):
        eps = eps255 / 255
        adv = pgd_scores(
            model, w, b, fake_paths, epsilon=eps,
            alpha=max(eps / 4, 0.5 / 255), steps=a.get("steps", 10),
            batch_size=a.get("batch_size", 24),
        )
        s = base.copy()
        s[y == 1] = adv
        rows.append({"condition": f"pgd_eps{eps255}", "epsilon_255": float(eps255),
                     **summarize(y, s)})

    df = pl.DataFrame(rows)
    df.write_csv(out_dir / "attack_pgd.csv")
    print(df)


def pl_expr_split(name: str):
    import polars as pl

    return pl.col("split") == name


def cmd_experiments(cfg: dict) -> None:
    out_dir = Path(cfg["manifest"].get("out_dir", cfg["eval"]["out_dir"]))
    res = run_all(
        manifest_path=out_dir / "manifest_normalized.parquet",
        cache_dir=cfg["features"]["cache_dir"],
        out_dir=out_dir,
        fprs=cfg["eval"].get("tpr_at_fpr"),
        score_cache_dir=cfg.get("npr", {}).get("score_cache_dir"),
    )
    for name, df in res.items():
        print(f"\n=== {name} ===")
        with pl_config():
            print(df)


def pl_config():
    import polars as pl

    return pl.Config(tbl_rows=40, tbl_cols=20, float_precision=4)


def cmd_fit(cfg: dict) -> None:
    det = _build(cfg)
    splits = _splits(cfg)
    train = splits["train"]
    print(f"Fitting {det.name} on {len(train)} images "
          f"({sum(train.labels)} fake / {len(train) - sum(train.labels)} real)")
    det.fit([load_image(p) for p in train.paths], train.labels)
    out = Path(cfg["eval"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if hasattr(det, "save"):
        det.save(out / "probe.joblib")
        print(f"Saved head -> {out / 'probe.joblib'}")


def _load_fitted(cfg: dict):
    det = _build(cfg)
    head = Path(cfg["eval"]["out_dir"]) / "probe.joblib"
    if hasattr(det, "load") and head.exists():
        det.load(head)
    return det


def cmd_eval(cfg: dict) -> None:
    det = _load_fitted(cfg)
    splits = _splits(cfg)
    curves = robustness_sweep(
        det, splits["test"], cfg.get("perturbations", {}),
        fprs=cfg["eval"].get("tpr_at_fpr"), out_dir=cfg["eval"]["out_dir"],
    )
    print(curves.to_string(index=False))


def cmd_calibrate(cfg: dict, target_fpr: float) -> None:
    det = _load_fitted(cfg)
    splits = _splits(cfg)
    calib = splits["calib"]
    scores = det.scores([load_image(p) for p in calib.paths])
    result = run_calibrate(np.asarray(calib.labels), scores, target_fpr)
    out = Path(cfg["eval"]["out_dir"]) / "calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2))
    print(json.dumps(result.to_dict(), indent=2))
    print(f"Saved -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="aigi-bench")
    ap.add_argument(
        "command",
        choices=[
            "manifest", "normalize", "features", "npr-scores", "experiments",
            "attack", "fit", "eval", "calibrate",
        ],
    )
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--target-fpr", type=float, default=0.05)
    args = ap.parse_args()
    cfg = _load_config(args.config)
    if args.command == "manifest":
        cmd_manifest(cfg)
    elif args.command == "normalize":
        cmd_normalize(cfg)
    elif args.command == "features":
        cmd_features(cfg)
    elif args.command == "npr-scores":
        cmd_npr_scores(cfg)
    elif args.command == "experiments":
        cmd_experiments(cfg)
    elif args.command == "attack":
        cmd_attack(cfg)
    elif args.command == "fit":
        cmd_fit(cfg)
    elif args.command == "eval":
        cmd_eval(cfg)
    else:
        cmd_calibrate(cfg, args.target_fpr)


if __name__ == "__main__":
    main()
