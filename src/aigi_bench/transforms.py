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


# --- Tier 2: laundering ------------------------------------------------------
# Tier 1 above is single-operation and matches the published robustness
# protocols. Tier 2 models what actually happens to an image between a generator
# and the place a detector sees it: a chat app, a screenshot, a re-upload, a CDN
# transcode. These are *compositions* rather than single operations.
#
# None of these are adversarial: no detector is consulted while building them.
# They are the benign-but-realistic middle ground between tier 1 and the
# white-box attack in attacks.py.
#
# MEASURED, on the album-cover corpus with the CLIP ViT-L/14 probe: tier 2 is
# NOT harsher than tier 1. Worst tier-2 condition is webp QF60 at -0.0122 AUROC;
# worst tier-1 is blur sigma=2 at -0.0147. Five successive JPEG generations cost
# 0.0073, and noise+median-filter — which targets pixel residuals directly —
# costs 0.0083.
#
# That is a statement about *semantic* detectors, not about these transforms.
# CLIP reads content and composition, which survive every codec here; a
# residual-based detector like NPR reads exactly what these operations destroy,
# so the same suite should be expected to hurt it far more. Keep the tier-2
# conditions in the sweep for that reason — they discriminate between detector
# families even when they barely move this one.


def webp_roundtrip(quality: int) -> Transform:
    """Encode to WebP and back. Different transform/quantisation than JPEG, so
    it perturbs a detector keyed on JPEG-specific DCT traces."""

    def _t(im: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        im.save(buf, format="WEBP", quality=int(quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    return _t


def recompress_chain(n: int, quality: int = 75) -> Transform:
    """`n` successive JPEG round-trips. Each generation drives the image further
    toward the codec's fixed points, erasing high-frequency generator traces
    that survive a single pass."""

    def _t(im: Image.Image) -> Image.Image:
        for _ in range(int(n)):
            im = jpeg_compress(quality)(im)
        return im

    return _t


def screenshot(scale: float = 1.0) -> Transform:
    """Screenshot simulation: resample to a screen-ish size, save lossless, then
    re-crop slightly — no JPEG, but a full resampling pass and a framing shift.
    Models 'someone screenshotted it and sent me the PNG'."""

    def _t(im: Image.Image) -> Image.Image:
        w, h = im.size
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        out = im.resize((nw, nh), Image.BICUBIC)
        buf = io.BytesIO()
        out.save(buf, format="PNG")
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
        return center_crop(0.95)(out)

    return _t


def social_pipeline(quality: int = 70) -> Transform:
    """Instagram/WhatsApp-style: downscale to a fixed long edge, sharpen, then
    a fairly aggressive JPEG. The sharpen step matters — platforms apply it, and
    it partially *restores* high-frequency energy that resizing removed, which
    is why this is not equivalent to resize-then-jpeg."""

    def _t(im: Image.Image) -> Image.Image:
        w, h = im.size
        target = 640
        s = target / max(w, h)
        if s < 1:
            im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
        im = im.filter(ImageFilter.UnsharpMask(radius=1.0, percent=80, threshold=3))
        return jpeg_compress(quality)(im)

    return _t


def noise_denoise(sigma: float) -> Transform:
    """Add Gaussian noise, then median-filter it out. Neither step alone is
    unusual, but the pair is a cheap approximation of the denoising stage in a
    camera or upscaler pipeline, and it specifically attacks detectors that key
    on pixel-level noise residuals (the NPR/FreqNet family)."""

    def _t(im: Image.Image) -> Image.Image:
        import numpy as np

        rng = np.random.default_rng(0)  # deterministic: this is an eval, not training
        a = np.asarray(im, dtype=np.float32)
        a = a + rng.normal(0.0, float(sigma), a.shape).astype(np.float32)
        out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), mode="RGB")
        return out.filter(ImageFilter.MedianFilter(size=3))

    return _t


FACTORIES: dict[str, Callable[[float], Transform]] = {
    # tier 1 — single benign operations (published protocols)
    "jpeg": lambda q: jpeg_compress(int(q)),
    "resize": lambda s: resize(float(s)),
    "resize_restore": lambda s: resize(float(s), restore=True),
    "blur": lambda s: gaussian_blur(float(s)),
    "crop": lambda a: center_crop(float(a)),
    # tier 2 — laundering (compositions seen in real distribution paths)
    "webp": lambda q: webp_roundtrip(int(q)),
    "recompress": lambda n: recompress_chain(int(n)),
    "screenshot": lambda s: screenshot(float(s)),
    "social": lambda q: social_pipeline(int(q)),
    "noise_denoise": lambda s: noise_denoise(float(s)),
}

TIER1 = ("jpeg", "resize", "resize_restore", "blur", "crop")
TIER2 = ("webp", "recompress", "screenshot", "social", "noise_denoise")


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
