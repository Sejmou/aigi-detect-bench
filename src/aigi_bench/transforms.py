"""Benign, everyday image transformations for robustness evaluation.

Mirrors the stress-test protocol common in the literature (e.g. NTIRE 2026,
HEDGE): JPEG recompression, resizing, Gaussian blur, and cropping at graded
intensities. Pure PIL, deterministic, no GPU needed.
"""
from __future__ import annotations

import io
from collections.abc import Callable

from PIL import Image, ImageFilter

Transform = Callable[[Image.Image], Image.Image]


def identity(im: Image.Image) -> Image.Image:
    return im


def jpeg_compress(quality: int) -> Transform:
    """Round-trip through JPEG at the given quality factor (100 = best)."""

    def _t(im: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    return _t


def resize(scale: float, restore: bool = False) -> Transform:
    """Bilinear resize by `scale`. If `restore`, resize back to original size
    (isolates resampling artifacts from resolution change)."""

    def _t(im: Image.Image) -> Image.Image:
        w, h = im.size
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        out = im.resize((nw, nh), Image.BILINEAR)
        if restore:
            out = out.resize((w, h), Image.BILINEAR)
        return out

    return _t


def gaussian_blur(sigma: float) -> Transform:
    def _t(im: Image.Image) -> Image.Image:
        if sigma <= 0:
            return im
        return im.filter(ImageFilter.GaussianBlur(radius=sigma))

    return _t


def center_crop(area_frac: float) -> Transform:
    """Center crop retaining `area_frac` of the pixels (aspect preserved)."""

    def _t(im: Image.Image) -> Image.Image:
        w, h = im.size
        s = area_frac**0.5
        cw, ch = max(1, round(w * s)), max(1, round(h * s))
        left, top = (w - cw) // 2, (h - ch) // 2
        return im.crop((left, top, left + cw, top + ch))

    return _t


FACTORIES: dict[str, Callable[[float], Transform]] = {
    "jpeg": lambda q: jpeg_compress(int(q)),
    "resize": lambda s: resize(float(s)),
    "resize_restore": lambda s: resize(float(s), restore=True),
    "blur": lambda s: gaussian_blur(float(s)),
    "crop": lambda a: center_crop(float(a)),
}


def build_grid(spec: dict[str, list[float]]) -> list[tuple[str, float | None, Transform]]:
    """Expand a config dict into [(name, intensity, transform), ...].

    Always prepends the clean condition.
    """
    grid: list[tuple[str, float | None, Transform]] = [("clean", None, identity)]
    for name, intensities in (spec or {}).items():
        if name not in FACTORIES:
            raise KeyError(f"Unknown perturbation '{name}'. Known: {sorted(FACTORIES)}")
        for x in intensities:
            grid.append((name, x, FACTORIES[name](x)))
    return grid
