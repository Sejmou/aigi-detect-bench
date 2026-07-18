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
    ap.add_argument("command", choices=["fit", "eval", "calibrate"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--target-fpr", type=float, default=0.05)
    args = ap.parse_args()
    cfg = _load_config(args.config)
    if args.command == "fit":
        cmd_fit(cfg)
    elif args.command == "eval":
        cmd_eval(cfg)
    else:
        cmd_calibrate(cfg, args.target_fpr)


if __name__ == "__main__":
    main()
