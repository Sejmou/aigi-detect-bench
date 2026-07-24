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
| Corpus ingest | `manifest.py` | Joins real covers ↔ generated reconstructions into one table |
| Normalization | `normalize.py` | Strips format/resolution as a class signal |

---

# The album-cover corpus

Findings from running this harness on a reconstruction corpus of Spotify album
art. **Read the caveats** — several affect how the numbers should be
interpreted.

## What the corpus is

10,000 real covers (all released pre-2014, before GANs, so "guaranteed real")
were densely captioned with Qwen3-VL, and the captions were fed back to six
text-to-image models via ComfyUI. Every fake therefore has a real counterpart
with matching content, keyed on Spotify album ID.

| Generator | n | native px | caption variant |
|---|---|---|---|
| qwen-image-2512 | 5,000 | 1024 | text |
| dreamshaper-8 | 2,150 | 512 | no-text |
| krea2-turbo | 2,000 | 1024 | text |
| pixeldit | 2,000 | 1024 | no-text |
| sdxl-turbo | 2,000 | 512 | no-text |
| z-image-turbo | 2,000 | 1024 | text |

5,000 distinct covers were used as sources; **2,000 were reconstructed by all
six models** (`is_paired_core`), which is the subset that supports clean
cross-generator comparison.

## Step 1 — manifest (`aigi-bench manifest`)

25,150 rows, ~9 s. Splits are assigned **per `spotify_id`, not per image**, so a
cover and its six reconstructions never straddle train/test — otherwise the
probe scores test images by recognising the album rather than by detecting
artifacts. Assignment is hash-based, so adding a seventh generator later does
not reshuffle existing assignments.

## Step 2 — normalization (`aigi-bench normalize`)

**The raw corpus is 100 % separable on metadata alone**: reals are JPEG
289–640 px and 17 % non-square; fakes are PNG, always square, 512 or 1024 px.
Training on that yields a ~0.99 AUROC detector that has learned "PNG"
(Grommelt et al., *Fake or JPEG?*, ECCV-W 2024). Normalization is therefore not
optional pre-processing — it is what makes any downstream number mean anything.

Pipeline: center-crop to square → resize to 512 → JPEG at a per-image random
quality in 85–95. 25,057 images in 84 s (12 cores).

**Assumptions and decisions, with their justification:**

| Decision | Why | Cost |
|---|---|---|
| Target **512 px** | short side of only 1.0 % of source reals falls below it, vs 20 % at 640 | — |
| **Drop** reals under 512 rather than upsample | upsampling stamps a resampling signature on one class only — the same species of confound being removed | 93 of 10,000 reals (0.9 %) |
| **Center-crop** to square, don't drop non-square | median off-square real is 1.008:1; the crop discards a few pixel rows | ~5 % have AR > 1.15 and lose real content |
| **Randomized** JPEG quality, not fixed | a fixed QF removes the format tell but leaves a constant-QF fingerprint | — |
| LANCZOS resampling | cheap filters imprint their own artifact, and the 1024 px models dominate the fake class | — |

**Verification gate (passed):** on a 4,000-image sample, both classes are
uniformly 512×512 JPEG with identical quality distributions (mean 89.99 vs
90.00, both spanning 85–95).

### Caveats that survive normalization

These affect interpretation and are **not** fixed by the pipeline above:

1. **Two generators are never resampled.** dreamshaper-8 and sdxl-turbo output
   512 px natively, so they alone pass through with no resize, while reals
   (640→512) and the four 1024 px models are LANCZOS-downsampled. Downsampling
   concentrates detail; those two models keep their native smoothness *and*
   lack a resampling signature. Their numbers are the least trustworthy in the
   set and should be read separately, not pooled.

2. **Residual file-size signal.** After normalization, file size alone still
   gives AUROC 0.564 pooled — but it is very uneven per generator:
   dreamshaper-8 0.786, sdxl-turbo 0.696, z-image-turbo 0.516,
   qwen-image-2512 0.443 (i.e. *larger* than real). This tracks caveat 1
   exactly. A pixel-space detector cannot read file size directly, so this is a
   proxy for content smoothness rather than a leak — but it sets a floor on how
   much of any per-generator gap is explained by blur alone.

3. **The text/no-text split is confounded with generator.** Three models were
   driven by `no-text` captions (which forbid mentioning writing) and three by
   `text`. Real album covers nearly always carry artist/title text, so "has
   text" is a usable shortcut — and because the variant is not crossed with
   generator, its effect cannot be separated from generator identity in this
   corpus. `prompt_variant` is a manifest column so it can at least be
   stratified on.

4. **Real covers are pre-2014; fakes are 2026 models.** The real class is
   biased toward older design conventions (typography, photographic style).
   Some of what a detector learns here is era, not generation.

5. **Class balance is 9,907 real vs 15,150 fake** (1:1.53), and
   qwen-image-2512 contributes a third of the fakes. Metrics are computed on
   raw scores; `balanced_acc` and TPR@FPR are robust to this, plain accuracy
   is not.

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
