import numpy as np
import pytest
from PIL import Image

from aigi_bench.transforms import (
    FACTORIES,
    build_grid,
    center_crop,
    gaussian_blur,
    jpeg_compress,
    resize,
)


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


# --- tier 2 laundering -------------------------------------------------------


def _img(size=(256, 256)):
    import numpy as np

    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (*size, 3), dtype=np.uint8), "RGB")


@pytest.mark.parametrize("name,intensity", [
    ("webp", 80), ("recompress", 3), ("screenshot", 1.0),
    ("social", 70), ("noise_denoise", 4.0),
])
def test_tier2_transforms_return_rgb_images(name, intensity):
    out = FACTORIES[name](intensity)(_img())
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"


def test_tier2_transforms_actually_change_pixels():
    import numpy as np

    im = _img()
    for name, intensity in [("webp", 80), ("recompress", 3), ("social", 70),
                            ("noise_denoise", 4.0)]:
        out = FACTORIES[name](intensity)(im)
        if out.size == im.size:
            assert not np.array_equal(np.asarray(out), np.asarray(im)), name


def test_noise_denoise_is_deterministic():
    import numpy as np

    im = _img()
    a = FACTORIES["noise_denoise"](4.0)(im)
    b = FACTORIES["noise_denoise"](4.0)(im)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_recompress_chain_is_monotonically_destructive():
    """More generations should not be closer to the original than fewer."""
    import numpy as np

    im = _img()
    ref = np.asarray(im, dtype=float)
    d1 = np.abs(np.asarray(FACTORIES["recompress"](1)(im), dtype=float) - ref).mean()
    d5 = np.abs(np.asarray(FACTORIES["recompress"](5)(im), dtype=float) - ref).mean()
    assert d5 >= d1


def test_build_grid_accepts_tier2_names():
    grid = build_grid({"webp": [80], "social": [70]})
    assert [g[0] for g in grid] == ["clean", "webp", "social"]
