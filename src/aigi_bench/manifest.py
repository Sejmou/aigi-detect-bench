"""Build the real/fake image manifest that every downstream stage reads.

The corpus this benchmark runs on is a *reconstruction* set, not two unrelated
piles of images. Each generated image traces back to a specific real album
cover: the cover was captioned (Qwen3-VL, `text` or `no-text` prompt variant),
and the caption was fed to a text-to-image model. So every fake has a real
counterpart with matching content, and the join key is the Spotify album ID.

That pairing is the reason this file exists. Two properties fall out of it that
a naive `list_images(real_dir) + list_images(fake_dir)` would destroy:

  grouped splits   A cover and all six of its reconstructions share content. If
                   the real lands in train and a fake in test, the probe can
                   score the test image by *recognising the album*, not by
                   detecting generation artifacts. Splits are therefore assigned
                   per `spotify_id`, never per image.

  generator held   Cross-generator generalization is the headline question, so
                   the manifest keeps `generator` as a first-class column and
                   flags the 2000 IDs reconstructed by all six models
                   (`is_paired_core`). Leave-one-generator-out evaluation is a
                   filter over this table, not a separate ingest.

The manifest also carries the two known confounds as explicit columns rather
than burying them, because both are large enough to dominate a naive AUROC:

  format/size    Reals are JPEG 640x640; fakes are PNG at 512x512 or 1024x1024.
                 A detector trained on this as-is learns "PNG" (Grommelt et al.,
                 "Fake or JPEG?", ECCV-W 2024). `width`/`height`/`img_format`
                 are recorded so the normalization pass can verify it actually
                 removed the gap, and so it stays auditable afterwards.

  text variant   Three generators were driven by `no-text` captions, which
                 forbid mentioning writing. Real covers nearly always carry
                 artist/title text, so "has text" leaks. `prompt_variant` lets
                 you stratify or hold out on it.

Reals are *all* emitted, including the 5000 never used as a caption source
(`is_source = false`), so downstream can choose its own class balance — the
6 generators produce ~15k fakes against 10k reals.

Output: <out>/manifest.parquet, one row per image, plus a manifest_meta.json
sidecar recording inputs, counts and the split seed (same contract as
run_meta.json in the sibling img-captioning projects).

Usage:
    uv run aigi-bench manifest --config configs/default.yaml
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from hashlib import blake2b
from pathlib import Path

import polars as pl
from PIL import Image
from tqdm import tqdm

# Generator run directories are named "<generator>_<captioner>_<variant>",
# e.g. "z-image-turbo_qwen3-vl_text" or "dreamshaper-8_qwen3-vl_no-text".
# The variant itself contains a hyphen, never an underscore, so a 3-way split
# is unambiguous.
RUN_DIR_PARTS = 3

SPLITS = ("train", "calib", "test")


def parse_run_dir(name: str) -> dict[str, str]:
    """Split a run directory name into generator / captioner / prompt variant."""
    parts = name.split("_")
    if len(parts) != RUN_DIR_PARTS:
        raise ValueError(
            f"Cannot parse run directory {name!r}: expected "
            f"'<generator>_<captioner>_<variant>', got {len(parts)} part(s)"
        )
    generator, captioner, variant = parts
    if variant not in ("text", "no-text"):
        raise ValueError(f"Unexpected prompt variant {variant!r} in {name!r}")
    return {"generator": generator, "captioner": captioner, "prompt_variant": variant}


def assign_split(key: str, seed: int, train_frac: float) -> str:
    """Deterministically map a grouping key to train / calib / test.

    Hash-based rather than shuffle-based so the assignment depends only on the
    key and the seed — adding a generator, or re-running with a different set of
    directories present, never reshuffles the images already assigned. The
    remainder after `train_frac` is split evenly between calib and test, keeping
    the convention in data.make_splits.
    """
    digest = blake2b(f"{seed}:{key}".encode(), digest_size=8).digest()
    u = int.from_bytes(digest, "big") / float(1 << 64)
    if u < train_frac:
        return "train"
    return "calib" if u < train_frac + (1.0 - train_frac) / 2 else "test"


def _probe(path: Path) -> tuple[int | None, int | None, str | None]:
    """Read image dimensions and format from the header only (no pixel decode).

    Unreadable files yield nulls rather than raising: a truncated download in a
    15k-image run should surface as a null row to filter on, not abort the
    ingest. `manifest_meta.json` records the count so they stay visible.
    """
    try:
        with Image.open(path) as im:
            return im.size[0], im.size[1], im.format
    except (OSError, ValueError):
        return None, None, None


def _probe_all(paths: Iterable[Path], desc: str) -> list[tuple]:
    paths = list(paths)
    return [_probe(p) for p in tqdm(paths, desc=desc, unit="img", leave=False)]


def collect_fakes(slop_root: Path, probe: bool = True) -> pl.DataFrame:
    """One row per generated image, read from each run's results.jsonl.

    results.jsonl is the source of truth for provenance (which caption, which
    workflow, which seed), but it records *attempts* — a row can exist for an
    image that failed to download. Rows are therefore filtered against what is
    actually on disk.
    """
    run_dirs = sorted(d for d in slop_root.iterdir() if d.is_dir() and (d / "images").is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No generator run directories under {slop_root}")

    frames = []
    for d in run_dirs:
        meta = parse_run_dir(d.name)
        results = d / "results.jsonl"
        if not results.exists():
            raise FileNotFoundError(f"{d.name}: missing results.jsonl")

        df = pl.read_ndjson(results, ignore_errors=True).select(
            pl.col("input_image_path"),
            pl.col("output_image_path").alias("image_path"),
            pl.col("seed").alias("gen_seed"),
            pl.col("generated_at"),
            pl.col("duration_s").alias("gen_duration_s"),
        )
        # The Spotify ID is the stem of the *source* cover; the generated file
        # is prefixed with a running index ("00042_<id>.png"), so deriving it
        # from input_image_path avoids depending on that prefix.
        df = df.with_columns(
            pl.col("input_image_path")
            .str.split("/")
            .list.last()
            .str.replace(r"\.jpg$", "")
            .alias("spotify_id"),
            pl.lit(d.name).alias("run_dir"),
            pl.lit(meta["generator"]).alias("generator"),
            pl.lit(meta["captioner"]).alias("captioner"),
            pl.lit(meta["prompt_variant"]).alias("prompt_variant"),
        )

        on_disk = {p.name for p in (d / "images").iterdir()}
        df = df.filter(
            pl.col("image_path").str.split("/").list.last().is_in(on_disk)
        ).unique(subset=["image_path"], keep="first")
        frames.append(df)

    fakes = pl.concat(frames, how="vertical")

    if probe:
        dims = _probe_all(
            (Path(p) for p in fakes["image_path"]), desc="probing fakes"
        )
        fakes = fakes.with_columns(
            pl.Series("width", [d[0] for d in dims], dtype=pl.Int32),
            pl.Series("height", [d[1] for d in dims], dtype=pl.Int32),
            pl.Series("img_format", [d[2] for d in dims], dtype=pl.Utf8),
        )
    return fakes.with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))


def collect_reals(real_dir: Path, probe: bool = True) -> pl.DataFrame:
    """One row per real cover on disk. The stem is the Spotify album ID."""
    paths = sorted(p for p in real_dir.iterdir() if p.suffix.lower() == ".jpg")
    if not paths:
        raise FileNotFoundError(f"No .jpg covers under {real_dir}")

    reals = pl.DataFrame(
        {
            "image_path": [str(p) for p in paths],
            "spotify_id": [p.stem for p in paths],
        }
    ).with_columns(
        pl.lit(0, dtype=pl.Int8).alias("label"),
        pl.lit(None, dtype=pl.Utf8).alias("generator"),
        pl.lit(None, dtype=pl.Utf8).alias("captioner"),
        pl.lit(None, dtype=pl.Utf8).alias("prompt_variant"),
        pl.lit(None, dtype=pl.Utf8).alias("run_dir"),
    )

    if probe:
        dims = _probe_all((Path(p) for p in reals["image_path"]), desc="probing reals")
        reals = reals.with_columns(
            pl.Series("width", [d[0] for d in dims], dtype=pl.Int32),
            pl.Series("height", [d[1] for d in dims], dtype=pl.Int32),
            pl.Series("img_format", [d[2] for d in dims], dtype=pl.Utf8),
        )
    return reals


def attach_album_metadata(df: pl.DataFrame, meta_parquet: Path) -> pl.DataFrame:
    """Left-join the Spotify album metadata (year, artist, popularity rank).

    Kept optional-ish: a missing file is fatal (it is part of the corpus), but a
    missing *row* is not — generated images are keyed on covers that are all in
    the top-10k table, so a null here means the ID drifted and is worth seeing.
    """
    meta = pl.read_parquet(
        meta_parquet,
        columns=[
            "album_id",
            "album_name",
            "artist_names",
            "release_year",
            "rank",
            "streams_de",
        ],
    ).rename({"album_id": "spotify_id"})
    return df.join(meta, on="spotify_id", how="left")


def build_manifest(
    real_dir: str | Path,
    slop_root: str | Path,
    meta_parquet: str | Path,
    out_dir: str | Path,
    seed: int = 42,
    train_frac: float = 0.6,
    probe: bool = True,
) -> pl.DataFrame:
    """Assemble and write manifest.parquet + manifest_meta.json."""
    real_dir, slop_root = Path(real_dir), Path(slop_root)
    meta_parquet, out_dir = Path(meta_parquet), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fakes = collect_fakes(slop_root, probe=probe)
    reals = collect_reals(real_dir, probe=probe)

    # How many distinct generators reconstructed each cover. 6 == the fully
    # paired core; 0 == a real never used as a caption source.
    per_id = fakes.group_by("spotify_id").agg(
        pl.col("generator").n_unique().alias("n_generators")
    )
    n_max = int(per_id["n_generators"].max()) if per_id.height else 0

    df = pl.concat([reals, fakes], how="diagonal")
    df = df.join(per_id, on="spotify_id", how="left").with_columns(
        pl.col("n_generators").fill_null(0).cast(pl.Int8)
    )
    df = df.with_columns(
        (pl.col("n_generators") > 0).alias("is_source"),
        (pl.col("n_generators") == n_max).alias("is_paired_core"),
    )

    # Grouped by spotify_id: a cover and all its reconstructions share a split.
    ids = df["spotify_id"].unique().to_list()
    split_map = {i: assign_split(i, seed, train_frac) for i in ids}
    df = df.with_columns(
        pl.col("spotify_id").replace_strict(split_map).alias("split")
    )

    df = attach_album_metadata(df, meta_parquet)
    df = df.select(
        "image_path", "label", "spotify_id", "split", "generator", "prompt_variant",
        "captioner", "run_dir", "n_generators", "is_source", "is_paired_core",
        "width", "height", "img_format", "release_year", "album_name",
        "artist_names", "rank", "streams_de", "gen_seed", "gen_duration_s",
        "generated_at", "input_image_path",
    ).sort("label", "generator", "spotify_id", nulls_last=False)

    out_path = out_dir / "manifest.parquet"
    df.write_parquet(out_path)

    meta = {
        "built_at": datetime.now(UTC).isoformat(),
        "real_dir": str(real_dir),
        "slop_root": str(slop_root),
        "meta_parquet": str(meta_parquet),
        "seed": seed,
        "train_frac": train_frac,
        "probed_dimensions": probe,
        "n_rows": df.height,
        "n_real": int((df["label"] == 0).sum()),
        "n_fake": int((df["label"] == 1).sum()),
        "n_ids": len(ids),
        "n_unreadable": int(df["width"].null_count()) if probe else None,
        "n_paired_core_ids": int(
            df.filter(pl.col("is_paired_core"))["spotify_id"].n_unique()
        ),
        "generators": sorted(df["generator"].drop_nulls().unique().to_list()),
        "per_generator": {
            r["generator"]: r["len"]
            for r in df.filter(pl.col("label") == 1)
            .group_by("generator")
            .len()
            .sort("generator")
            .to_dicts()
        },
        "per_split": {
            r["split"]: r["len"] for r in df.group_by("split").len().sort("split").to_dicts()
        },
    }
    (out_dir / "manifest_meta.json").write_text(json.dumps(meta, indent=2))
    return df
