import numpy as np
from PIL import Image

from aigi_bench.transforms import build_grid, center_crop, gaussian_blur, jpeg_compress, resize


def _im(w=64, h=48):
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def test_jpeg_roundtrip_changes_pixels_keeps_size():
    im = _im()
    out = jpeg_compress(50)(im)
    assert out.size == im.size
    assert np.any(np.asarray(out) != np.asarray(im))


def test_resize_scales_and_restores():
    im = _im(64, 48)
    assert resize(0.5)(im).size == (32, 24)
    assert resize(0.5, restore=True)(im).size == (64, 48)


def test_crop_area_fraction():
    im = _im(100, 100)
    out = center_crop(0.25)(im)
    assert abs(out.size[0] * out.size[1] / (100 * 100) - 0.25) < 0.02


def test_blur_noop_at_zero():
    im = _im()
    assert np.array_equal(np.asarray(gaussian_blur(0)(im)), np.asarray(im))


def test_grid_includes_clean_and_all_intensities():
    grid = build_grid({"jpeg": [90, 60], "blur": [1.0]})
    names = [(n, i) for n, i, _ in grid]
    assert names[0] == ("clean", None)
    assert ("jpeg", 90) in names and ("jpeg", 60) in names and ("blur", 1.0) in names
