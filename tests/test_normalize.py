"""CPU-only tests for the normalization pass."""
from __future__ import annotations

import polars as pl
import pytest
from PIL import Image

from aigi_bench.normalize import (
    QUALITY_BAND,
    NormJob,
    center_square,
    normalize_one,
    plan_jobs,
    quality_for,
)


def test_center_square_crops_landscape():
    assert center_square(Image.new("RGB", (640, 480))).size == (480, 480)


def test_center_square_crops_portrait():
    assert center_square(Image.new("RGB", (480, 640))).size == (480, 480)


def test_center_square_passes_square_through():
    assert center_square(Image.new("RGB", (512, 512))).size == (512, 512)


def test_quality_is_in_band_and_deterministic():
    q1 = quality_for("/a/b.jpg", 42)
    q2 = quality_for("/a/b.jpg", 42)
    assert q1 == q2
    assert QUALITY_BAND[0] <= q1 <= QUALITY_BAND[1]


def test_quality_varies_across_images():
    qs = {quality_for(f"/img/{i}.jpg", 42) for i in range(400)}
    # should actually spread over the band, not collapse to one value
    assert len(qs) == QUALITY_BAND[1] - QUALITY_BAND[0] + 1


def test_quality_depends_on_seed():
    a = [quality_for(f"/img/{i}.jpg", 1) for i in range(200)]
    b = [quality_for(f"/img/{i}.jpg", 2) for i in range(200)]
    assert a != b


def _manifest(rows):
    return pl.DataFrame(
        rows,
        schema={
            "image_path": pl.Utf8,
            "label": pl.Int8,
            "spotify_id": pl.Utf8,
            "generator": pl.Utf8,
            "width": pl.Int32,
            "height": pl.Int32,
        },
    )


def test_plan_jobs_drops_undersized_reals(tmp_path):
    m = _manifest(
        [
            {"image_path": "/r/a.jpg", "label": 0, "spotify_id": "a",
             "generator": None, "width": 400, "height": 400},
            {"image_path": "/r/b.jpg", "label": 0, "spotify_id": "b",
             "generator": None, "width": 640, "height": 633},
        ]
    )
    keep, dropped = plan_jobs(m, tmp_path, seed=42)
    assert dropped.height == 1 and dropped["spotify_id"][0] == "a"
    # 640x633 survives: its SHORT side (633) clears 512
    assert keep.height == 1 and keep["spotify_id"][0] == "b"


def test_plan_jobs_names_fakes_by_generator(tmp_path):
    m = _manifest(
        [
            {"image_path": "/f/x1.png", "label": 1, "spotify_id": "x",
             "generator": "sdxl-turbo", "width": 512, "height": 512},
            {"image_path": "/f/x2.png", "label": 1, "spotify_id": "x",
             "generator": "pixeldit", "width": 1024, "height": 1024},
        ]
    )
    keep, _ = plan_jobs(m, tmp_path, seed=42)
    names = sorted(p.split("/")[-1] for p in keep["normalized_path"])
    # same cover, two generators -> must not collide
    assert names == ["x__pixeldit.jpg", "x__sdxl-turbo.jpg"]


@pytest.mark.parametrize("size", [(1024, 1024), (640, 633), (512, 512)])
def test_normalize_one_emits_512_square_jpeg(tmp_path, size):
    src = tmp_path / "src.png"
    Image.new("RGB", size, (120, 60, 30)).save(src)
    dst = tmp_path / "out" / "img.jpg"
    _, ok, err = normalize_one(NormJob(str(src), str(dst), 90))
    assert ok, err
    with Image.open(dst) as im:
        assert im.size == (512, 512)
        assert im.format == "JPEG"


def test_normalize_one_is_resumable(tmp_path):
    src = tmp_path / "s.png"
    Image.new("RGB", (512, 512)).save(src)
    dst = tmp_path / "o.jpg"
    normalize_one(NormJob(str(src), str(dst), 90))
    mtime = dst.stat().st_mtime_ns
    normalize_one(NormJob(str(src), str(dst), 90))
    assert dst.stat().st_mtime_ns == mtime  # untouched on second pass


def test_normalize_one_reports_unreadable(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    _, ok, err = normalize_one(NormJob(str(bad), str(tmp_path / "o.jpg"), 90))
    assert not ok and err


def test_normalize_one_leaves_no_tmp_file(tmp_path):
    src = tmp_path / "s.png"
    Image.new("RGB", (600, 600)).save(src)
    dst = tmp_path / "out" / "o.jpg"
    normalize_one(NormJob(str(src), str(dst), 90))
    assert not list(dst.parent.glob("*.tmp.jpg"))
