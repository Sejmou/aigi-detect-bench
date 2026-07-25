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

## Summary

Five results, in rough order of how much they should change what you do:

1. **The format confound dominates everything.** NPR scores 0.994 AUROC on the
   raw corpus and 0.427 after normalization. Any benchmark whose classes differ
   in container format or resolution is measuring the container.
2. **Cross-generator drift is the real failure mode, not image processing.**
   Across 28 benign conditions — including 5× JPEG chains and noise+denoise —
   the probe never loses more than 0.015 AUROC. Holding out an unseen generator
   costs 0.045 and drops TPR@5%FPR from 0.89 to 0.68.
3. **Generalization is asymmetric — train on your hardest generator.** Probes
   trained on the easy cluster transfer at 0.72–0.82; probes trained on the hard
   cluster transfer at 0.91–0.98.
4. **Ensembling is not free.** Averaging the CLIP probe with a below-chance NPR
   cost 0.19 AUROC versus the probe alone.
5. **Against an adaptive attacker there is no defence here.** PGD at ε=1/255
   takes the probe to 0.039 AUROC and catches zero fakes at 5% FPR.

Reproduce with:

```bash
uv run aigi-bench manifest    # join covers <-> reconstructions
uv run aigi-bench normalize   # strip the format/resolution signal
uv run aigi-bench features    # cache CLIP features per condition (GPU, ~1h)
uv run aigi-bench npr-scores  # cache NPR scores
uv run aigi-bench experiments # robustness, LOGO, 6x6 matrix, calibration, ensemble
uv run aigi-bench attack      # tier-4 white-box PGD
```

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

## Step 3 — the format confound is real, and it is enormous

The clearest evidence that normalization was necessary comes from **NPR**
(Tan et al., CVPR 2024), a published detector applied zero-shot:

| | pooled AUROC |
|---|---|
| NPR on **raw** corpus (JPEG reals vs PNG fakes) | **0.9937** |
| NPR on **normalized** corpus (both 512 px JPEG) | **0.4265** |

Per-generator on raw it is 0.989–0.998; on normalized, 0.257–0.495. A detector
that looks state-of-the-art on the raw corpus is, after the container tell is
removed, **worse than chance**.

Two things follow. First, any published number on a corpus where the classes
differ in format or resolution should be treated as unproven. Second, NPR's
inversion is itself informative: it reads high-frequency pixel residual as
evidence of fakeness (it was trained on GAN upsampling artifacts), and modern
diffusion output on album art is *smoother* than photographic covers, so the
sign flips. This matches the file-size result above independently.

**A hypothesis we tested and rejected:** reals arrive as JPEG and are
re-encoded, giving them two compression generations against the fakes' one. We
checked whether that asymmetry explained the inversion by giving the fakes a
second generation — AUROC moved from 0.4265 to 0.4244, i.e. not at all. The
compression-history asymmetry is *not* driving the result, so no correction for
it is applied.

## Step 4 — cross-generator generalization

Trained on one generator (rows), tested on each (columns). Scored on the paired
core, so every cell sees the same 2,000 source covers.

| train ↓ / test → | dreamshaper-8 | pixeldit | sdxl-turbo | krea2-turbo | qwen-image | z-image |
|---|---|---|---|---|---|---|
| **dreamshaper-8** | 1.000 | 0.998 | 0.998 | 0.759 | 0.760 | 0.815 |
| **pixeldit** | 1.000 | 0.999 | 0.999 | 0.746 | 0.755 | 0.799 |
| **sdxl-turbo** | 0.999 | 0.997 | 0.999 | 0.723 | 0.715 | 0.775 |
| **krea2-turbo** | 0.984 | 0.959 | 0.953 | 0.947 | 0.959 | 0.970 |
| **qwen-image-2512** | 0.974 | 0.945 | 0.911 | 0.926 | 0.974 | 0.960 |
| **z-image-turbo** | 0.983 | 0.962 | 0.955 | 0.920 | 0.936 | 0.973 |

**The matrix is strongly asymmetric, and that is the main result.** Two clusters
fall out:

- **Cluster A — dreamshaper-8, pixeldit, sdxl-turbo.** Mutually detectable at
  ~0.999, but a probe trained on any of them transfers to cluster B at only
  0.72–0.82.
- **Cluster B — krea2-turbo, qwen-image-2512, z-image-turbo.** Harder in
  absolute terms (~0.95 self), but training on them generalizes *everywhere*,
  including 0.91–0.98 on cluster A.

The practical reading: **train on your hardest generator, not your most
available one.** A probe trained on the easy cluster looks excellent in
validation and degrades by ~0.25 AUROC on models it has not seen.

The clustering is *not* explained by resolution — pixeldit is natively 1024 px
yet sits with the two 512 px models. That is a useful internal control, because
it means the split is not an artifact of the resampling asymmetry noted above.

### Leave-one-generator-out

Train on reals + five generators, test on the held-out sixth:

| held out | in-dist AUROC | held-out AUROC | gap | held-out TPR@5%FPR |
|---|---|---|---|---|
| krea2-turbo | 0.983 | **0.932** | +0.051 | **0.683** |
| qwen-image-2512 | 0.979 | **0.947** | +0.032 | **0.732** |
| z-image-turbo | 0.979 | 0.965 | +0.014 | 0.822 |
| sdxl-turbo | 0.976 | 0.992 | −0.015 | 0.962 |
| pixeldit | 0.975 | 0.995 | −0.020 | 0.982 |
| dreamshaper-8 | 0.974 | 0.999 | −0.025 | 1.000 |

Negative gaps mean the held-out generator was *easier* than the training mix —
which is exactly the cluster-A/B structure again, not a bug.

**The number to quote is TPR@5%FPR, not AUROC.** krea2-turbo held out gives
0.932 AUROC, which sounds strong, but only **68 % of its fakes are caught at a
5 % false-positive rate**. Nearly a third slip through at an operating point
that already misflags one real cover in twenty.

## Step 5 — robustness to benign processing (tier 1)

Probe trained on **clean** train-split features only, then evaluated on the test
split under each perturbation. This is the honest setup: it measures how a
detector trained on pristine data degrades in the wild. (Training on perturbed
data is a mitigation to evaluate separately, not a baseline.)

| condition | AUROC | Δ vs clean | TPR@5%FPR |
|---|---|---|---|
| clean | 0.9772 | — | 0.889 |
| blur σ=0.5 | 0.9767 | −0.000 | 0.886 |
| crop 0.9 | 0.9766 | −0.001 | 0.892 |
| resize 2.0× | 0.9766 | −0.001 | 0.893 |
| jpeg QF 90 | 0.9765 | −0.001 | 0.889 |
| resize 0.5× | 0.9747 | −0.002 | 0.883 |
| blur σ=1.0 | 0.9733 | −0.004 | 0.874 |
| crop 0.5 | 0.9721 | −0.005 | 0.872 |
| jpeg QF 75 | 0.9709 | −0.006 | 0.861 |
| jpeg QF 40 | 0.9689 | −0.008 | 0.856 |
| **blur σ=2.0** | **0.9625** | **−0.015** | 0.832 |

**The CLIP probe is essentially immune to tier-1 processing.** The worst single
transform costs 1.5 AUROC points. This is the expected behaviour of a *semantic*
detector and it is the mirror image of NPR's collapse: CLIP reads content and
composition, which JPEG and blur leave intact, while NPR reads a pixel residual
that the same operations destroy.

### Tier 2 — laundering (28 conditions total)

Tier 2 composes operations the way a real distribution path does: WebP
round-trips, multi-generation JPEG chains, screenshot simulation, a
social-media pipeline (resize + unsharp + JPEG), and noise-then-median-filter.

| tier | worst condition | Δ AUROC | TPR@5%FPR |
|---|---|---|---|
| tier 1 | blur σ=2.0 | **−0.0147** | 0.832 |
| tier 2 | webp QF 60 | **−0.0122** | 0.835 |

Selected tier-2 rows:

| condition | AUROC | Δ vs clean |
|---|---|---|
| webp QF 60 | 0.9649 | −0.0122 |
| noise+denoise σ=8 | 0.9689 | −0.0083 |
| recompress ×5 | 0.9698 | −0.0073 |
| social QF 55 | 0.9703 | −0.0068 |
| screenshot 1.0× | 0.9756 | −0.0015 |

**We expected tier 2 to be harsher than tier 1. It is not.** Five successive
JPEG generations cost 0.0073 AUROC; noise-then-median-filter, which targets
pixel residuals directly, costs 0.0083. Both are milder than a single σ=2 blur.

The explanation is the same mechanism as everywhere else in this document: these
compositions destroy *low-level* structure, and the CLIP probe does not use
low-level structure. The result is therefore a statement about semantic
detectors, not about the transforms — the identical suite should be expected to
devastate a residual-based detector like NPR. The tier-2 conditions stay in the
sweep for exactly that reason: they discriminate between detector families even
when they barely move this one.

The practical consequence is that **cross-generator drift, not image processing,
is this detector's failure mode.** Across all 28 perturbation conditions the
probe never loses more than 0.015 AUROC, while holding out krea2-turbo costs
0.045 and drops TPR@5%FPR from 0.889 to 0.683. Robustness effort spent on
compression augmentation would be misdirected here; effort spent on generator
coverage would not.

## Calibration and operating points

Threshold fitted on the held-out **calib** split, then applied unchanged to
test:

| target FPR | threshold | realized FPR (test) | TPR (test) |
|---|---|---|---|
| 1 % | +1.379 | 0.94 % | 0.732 |
| 5 % | +0.323 | 4.11 % | 0.871 |

Temperature scaling improves ECE from 0.0555 to 0.0413 (T = 0.719). The target
FPRs transfer almost exactly, which is what the grouped-by-`spotify_id` split is
buying — with content leakage across splits these numbers would be optimistic.

Taking the 5 % threshold tuned on **clean** data and applying it to perturbed
images without retuning:

| applied to | FPR | TPR |
|---|---|---|
| clean | 0.041 | 0.871 |
| jpeg QF 40 | 0.025 | 0.791 |
| blur σ=2.0 | 0.047 | 0.828 |

The README's standing warning is that a threshold tuned on pristine PNGs
over-fires on recompressed images. **That does not happen here** — if anything
JPEG makes it *under*-fire (FPR 0.041 → 0.025), because compression shifts both
classes' scores down together rather than pulling them apart. The warning is
sound in general; it just does not bind for a semantic detector whose features
barely move under compression. Worth knowing which regime you are in before
budgeting for per-condition recalibration.

## Step 6 — ensembling made things worse

Both detectors on the same clean test split:

| detector | AUROC | TPR@5%FPR |
|---|---|---|
| clip_linear | **0.9772** | **0.889** |
| npr (zero-shot) | 0.3941 | 0.023 |
| mean ensemble (z-normalised) | 0.7910 | 0.397 |
| ensemble, NPR sign-corrected | 0.8754 | 0.515 |

Spearman correlation between the two score vectors: **−0.22**.

**Averaging cost 0.19 AUROC versus the CLIP probe alone**, and flipping NPR's
sign — the best case for an operator who had measured the inversion — still
lands 0.10 below. This contradicts the guidance in the Method notes below
("ensemble heterogeneous families for stability"), so it is worth being precise
about why rather than just noting the exception:

Score-level ensembling assumes each member is individually informative and that
their errors are somewhat independent. Here the second member is *anti*-informative
on this corpus, so the mean imports noise rather than complementary evidence.
Heterogeneity is necessary for a useful ensemble but not sufficient — the
members must first clear chance on the data you are actually deploying against.

The general lesson is about the *order of operations*: validate each member on
your own distribution first, and only then combine. Had we trusted NPR's
published performance and ensembled without checking, we would have shipped a
detector 0.19 AUROC worse than the trivial one-model baseline.

## Step 7 — tier-4 white-box attack

Tiers 1–2 are *non-adaptive*: the perturbation is chosen without consulting the
detector. Tier 4 gives the attacker everything — CLIP weights, probe
coefficients, gradients — and optimises directly against the decision function.
PGD, 10 steps, L∞ budget in [0,1] pixel space, run end-to-end through CLIP
preprocessing so the perturbation must survive resize and normalisation.

Threat model: the attacker perturbs **only the fakes** (the goal is evading
detection, not framing real photographs), on a balanced 500/500 test subset.

| condition | AUROC | TPR@5%FPR |
|---|---|---|
| clean (same pipeline) | 0.9807 | 0.910 |
| **PGD ε = 1/255** | **0.0388** | **0.0000** |

At the smallest budget tested — a perturbation below the threshold of visibility
— the detector does not merely lose signal, it **inverts**: AUROC 0.039 means it
is reliably *wrong*, and TPR@5%FPR of exactly 0.000 means **not one** attacked
fake is caught at that operating point.

This is the expected result, and the reason it is reported in its own column
rather than averaged with anything: a frozen-feature linear probe offers no
resistance to a gradient-based attacker. The 0.98 clean number and the 0.04
attacked number describe two different threat models, and quoting the first
without the second would be misleading.

**What this does and does not mean.** It does *not* say the detector is useless
— against non-adaptive adversaries (the overwhelming majority of real cases) it
holds at 0.96–0.98 through every benign transform we tested. It says the
detector must not be the only control where a motivated adversary is in scope,
which is the same conclusion the provenance literature reaches by a different
route.

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
