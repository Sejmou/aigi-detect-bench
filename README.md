# aigi-detect-bench

Starter harness for **defensive evaluation of AI-generated-image (AIGI) detectors**:
cross-generator generalization, robustness curves under benign transformations
(NTIRE-2026-style: JPEG, resize, blur, crop), threshold calibration, and
score-level ensembling. Designed to run on a single consumer GPU (e.g. RTX 3090).

> Scope: this repo is for *evaluating and stress-testing detectors*, not for
> evading them. The transformation suite mirrors published evaluation protocols
> (benign, everyday processing) so you can reproduce robustness numbers.

## Quick start (uv)

```bash
# install uv if needed: https://docs.astral.sh/uv/
uv sync                      # creates .venv and installs deps from pyproject/uv.lock
uv sync --extra recon        # + diffusers stack for reconstruction-based detectors
uv run pytest                # sanity-check the transform/metric code (CPU-only, no data needed)
```

Put images in:

```
data/real/...   # real photos (any nested structure, jpg/png/webp)
data/fake/...   # generated images
```

Then:

```bash
# 1) Fit the CLIP linear-probe baseline on your own split
uv run aigi-bench fit --config configs/default.yaml

# 2) Robustness curves: score clean + perturbed copies, write CSV + plots
uv run aigi-bench eval --config configs/default.yaml

# 3) Calibrate a threshold at a target FPR on a held-out calibration split
uv run aigi-bench calibrate --config configs/default.yaml --target-fpr 0.05
```

Outputs land in `outputs/`: per-image scores, `robustness_curves.csv`,
`robustness_curves.png`, and a `calibration.json` with the chosen threshold +
temperature.

## What's included

| Piece | File | Notes |
|---|---|---|
| CLIP ViT-L/14 linear probe | `detectors/clip_linear.py` | UniversalFakeDetect-style baseline; trains in minutes on a 3090 |
| Generic checkpoint wrapper | `detectors/external.py` | Adapter for third-party detectors (NPR, AIDE, DRCT, ...) |
| Perturbation suite | `transforms.py` | JPEG QF 100→40, resize 0.5×→2.0×, Gaussian blur σ 0→2, center/random crop |
| Metrics | `metrics.py` | AUROC, balanced acc, TPR@k%FPR, ECE |
| Calibration | `calibrate.py` | Temperature scaling + threshold @ target FPR |
| Runner | `eval.py`, `cli.py` | Robustness-curve sweep, CSV/PNG artifacts |

## Plugging in published detectors

The out-of-the-box literature checkpoints are distributed by their authors; this
repo deliberately does not vendor them. Recommended external repos (all with
code + weights, all 3090-friendly):

- **UniversalFakeDetect** (Ojha et al., CVPR'23) — CLIP feature probe (our
  baseline reimplements the spirit of this; their repo has the paper weights)
- **NPR** (Tan et al., CVPR'24) — neighboring-pixel relationships
- **FreqNet** (Tan et al., AAAI'24) — frequency-domain
- **AIDE** (Yan et al.) — two-stream frequency + semantic
- **DRCT** (Chen et al., ICML'24) — reconstruction-contrastive CLIP/ConvNeXt
- **DIRE / FIRE** — diffusion-reconstruction error (`--extra recon`)
- **SAFE**, **C2P-CLIP**, **Effort** — recent CLIP-adaptation methods

To add one, subclass `Detector` in `src/aigi_bench/detectors/base.py` (one
method: `scores(images) -> np.ndarray`, higher = more likely fake) and register
it in `detectors/__init__.py`. `detectors/external.py` shows the pattern for
wrapping an arbitrary `torch.nn.Module` + checkpoint + preprocessing.

Benchmark datasets to point the harness at: GenImage, UniversalFakeDetect test
sets, Synthbuster, Chameleon, the NTIRE 2026 challenge data, and RAID (for
adversarial stress tests — evaluation only).

## Repo layout

```
├── pyproject.toml            # uv-managed project
├── configs/default.yaml      # paths, model, perturbation grid
├── src/aigi_bench/
│   ├── data.py               # folder datasets + splits
│   ├── transforms.py         # perturbation suite (pure-PIL, deterministic)
│   ├── metrics.py            # AUROC, TPR@FPR, ECE
│   ├── calibrate.py          # temperature scaling + thresholding
│   ├── eval.py               # robustness sweep
│   ├── cli.py                # fit / eval / calibrate entrypoints
│   └── detectors/
│       ├── base.py           # Detector interface + registry
│       ├── clip_linear.py    # CLIP ViT-L/14 + logistic head
│       └── external.py       # adapter for third-party checkpoints
├── scripts/checkpoints.md    # where to get published weights
└── tests/                    # CPU-only unit tests
```

## Method notes

- **Report TPR@5%FPR, not accuracy.** Accuracy at 0.5 is misleading once the
  real/fake base rate or the score distribution shifts.
- **Calibrate on data that matches deployment** — including recompressed and
  resized copies. A threshold tuned on pristine PNGs will over-fire on
  screenshots and social-media re-uploads.
- **Match compression/resolution between real and fake sets** when training,
  or the model learns "JPEG quality" instead of generator artifacts
  (Grommelt et al., "Fake or JPEG?", ECCV-W 2024).
- **Ensemble heterogeneous families** (low-level + CLIP-semantic +
  reconstruction) for stability; single-family detectors fail correlated.
- Detector scores are **triage signals**, to be combined with provenance
  (C2PA/Content Credentials) and human review for consequential decisions.

## License

MIT (this scaffold). External detector code/weights keep their own licenses.
