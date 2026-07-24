"""Normalize the corpus so that container format and resolution carry no signal.

This is the load-bearing preprocessing step, and skipping it invalidates every
number downstream. The raw corpus separates perfectly on metadata alone:

    real   JPEG, 289-640px, 17% non-square    (Spotify cover art)
    fake   PNG,  512 or 1024px, always square (ComfyUI output)

A detector trained on that learns "PNG" and reports ~0.99 AUROC while having
learned nothing about generation artifacts — the failure documented in
Grommelt et al., "Fake or JPEG?" (ECCV-W 2024). The fix is to put both classes
through an identical geometry and encoding pipeline:

  1. center-crop to square    kills the aspect-ratio tell (17% of reals, 0% of
                              fakes). Median off-square real is 1.008:1, so the
                              crop discards a few pixel rows for most of them.
  2. resize to EDGE (512)     512 is deliberate: the short side of only 1.0% of
                              source reals falls below it, versus 20% at 640.
                              sdxl-turbo/dreamshaper are natively 512, the four
                              1024 models downsample.
  3. JPEG at random quality   both classes, quality drawn per-image from
                              QUALITY_BAND. A *fixed* quality would remove the
                              format tell but leave a constant-QF fingerprint;
                              randomizing means the detector cannot condition on
                              compression level either. The draw is seeded per
                              image path, so re-runs are byte-identical.

Reals whose short side is under EDGE are dropped rather than upsampled (50 of
5000 source covers, 1.0%): upsampling stamps a resampling signature onto one
class only, which is the same species of confound this pass exists to remove.

Fakes are never dropped — they are all >= 512 by construction.

Output goes to <out_root>/{real,fake}/<spotify_id>[__<generator>].jpg and the
manifest gains `normalized_path` + `norm_quality`. Existing outputs are skipped,
so the pass is resumable.

Usage:
    uv run aigi-bench normalize --config configs/default.yaml
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import blake2b
from pathlib import Path

import polars as pl
from PIL import Image
from tqdm import tqdm

# Both endpoints inclusive. Chosen to span "typical web re-encode" without
# reaching visually-lossless (which would leave generator artifacts pristine)
# or destructive (which would erase them for both classes equally).
QUALITY_BAND = (85, 95)

EDGE = 512


@dataclass(frozen=True)
class NormJob:
    src: str
    dst: str
    quality: int


def quality_for(path: str, seed: int, band: tuple[int, int] = QUALITY_BAND) -> int:
    """Deterministic per-image JPEG quality drawn from `band`.

    Keyed on the *source* path so a given image always gets the same quality
    regardless of run order, parallelism, or which subset is being processed.
    """
    lo, hi = band
    digest = blake2b(f"{seed}:q:{path}".encode(), digest_size=8).digest()
    return lo + int.from_bytes(digest, "big") % (hi - lo + 1)


def center_square(im: Image.Image) -> Image.Image:
    """Crop to the largest centered square."""
    w, h = im.size
    if w == h:
        return im
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return im.crop((left, top, left + s, top + s))


def normalize_one(job: NormJob) -> tuple[str, bool, str | None]:
    """Process a single image. Returns (dst, ok, error)."""
    try:
        dst = Path(job.dst)
        if dst.exists():  # resumable
            return job.dst, True, None
        with Image.open(job.src) as im:
            im = im.convert("RGB")
            im = center_square(im)
            if im.size[0] != EDGE:
                # LANCZOS for downsampling: the four 1024px generators dominate
                # the fake class, and a cheap filter there would imprint its own
                # artifact on that class only.
                im = im.resize((EDGE, EDGE), Image.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".tmp.jpg")
            im.save(tmp, format="JPEG", quality=job.quality, subsampling="4:2:0")
            tmp.rename(dst)  # atomic: a killed run never leaves a partial file
        return job.dst, True, None
    except (OSError, ValueError) as e:
        return job.dst, False, f"{type(e).__name__}: {e}"


def plan_jobs(
    manifest: pl.DataFrame, out_root: Path, seed: int, edge: int = EDGE
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Decide the output path for every manifest row; split into keep / dropped.

    Fakes get `<id>__<generator>.jpg` so the six reconstructions of one cover do
    not collide; reals get `<id>.jpg`.
    """
    short_side = pl.min_horizontal("width", "height")
    df = manifest.with_columns(short_side.alias("short_side"))

    # Only reals can fall below the target edge; fakes are >= 512 by construction.
    dropped = df.filter(pl.col("short_side") < edge)
    keep = df.filter(pl.col("short_side") >= edge)

    keep = keep.with_columns(
        pl.when(pl.col("label") == 0)
        .then(
            pl.lit(str(out_root / "real")) + "/" + pl.col("spotify_id") + pl.lit(".jpg")
        )
        .otherwise(
            pl.lit(str(out_root / "fake"))
            + "/"
            + pl.col("spotify_id")
            + pl.lit("__")
            + pl.col("generator")
            + pl.lit(".jpg")
        )
        .alias("normalized_path")
    )
    keep = keep.with_columns(
        pl.col("image_path")
        .map_elements(lambda p: quality_for(p, seed), return_dtype=pl.Int32)
        .alias("norm_quality")
    )
    return keep, dropped


def normalize_corpus(
    manifest_path: str | Path,
    out_root: str | Path,
    seed: int = 42,
    edge: int = EDGE,
    workers: int | None = None,
) -> pl.DataFrame:
    """Run the normalization pass and rewrite the manifest with output paths."""
    manifest_path, out_root = Path(manifest_path), Path(out_root)
    manifest = pl.read_parquet(manifest_path)

    keep, dropped = plan_jobs(manifest, out_root, seed, edge)
    jobs = [
        NormJob(src=r["image_path"], dst=r["normalized_path"], quality=r["norm_quality"])
        for r in keep.select("image_path", "normalized_path", "norm_quality").to_dicts()
    ]

    (out_root / "real").mkdir(parents=True, exist_ok=True)
    (out_root / "fake").mkdir(parents=True, exist_ok=True)

    errors: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for dst, ok, err in tqdm(
            ex.map(normalize_one, jobs, chunksize=32),
            total=len(jobs),
            desc="normalizing",
            unit="img",
        ):
            if not ok:
                errors.append((dst, err or "unknown"))

    failed = {d for d, _ in errors}
    out = keep.filter(~pl.col("normalized_path").is_in(list(failed))) if failed else keep
    out = out.drop("short_side")
    out.write_parquet(manifest_path.parent / "manifest_normalized.parquet")

    meta = {
        "normalized_at": datetime.now(UTC).isoformat(),
        "source_manifest": str(manifest_path),
        "out_root": str(out_root),
        "edge": edge,
        "quality_band": list(QUALITY_BAND),
        "seed": seed,
        "n_in": manifest.height,
        "n_out": out.height,
        "n_dropped_below_edge": dropped.height,
        "n_dropped_real": int((dropped["label"] == 0).sum()),
        "n_dropped_fake": int((dropped["label"] == 1).sum()),
        "n_failed": len(errors),
        "errors": errors[:50],
    }
    (manifest_path.parent / "normalize_meta.json").write_text(json.dumps(meta, indent=2))
    return out
