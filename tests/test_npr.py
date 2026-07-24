"""CPU-only tests for the NPR detector (architecture, not the checkpoint)."""
from __future__ import annotations

import torch

from aigi_bench.detectors.npr import NPRNet, _strip_prefix


def test_strip_prefix_removes_dataparallel_wrapper():
    assert _strip_prefix({"module.conv1.weight": 1, "bn1.bias": 2}) == {
        "conv1.weight": 1,
        "bn1.bias": 2,
    }


def test_stem_is_3x3_not_torchvision_7x7():
    """The authors replace torchvision's 7x7 stem; using 7x7 breaks the checkpoint."""
    assert NPRNet().conv1.weight.shape == (64, 3, 3, 3)


def test_forward_returns_one_logit_per_image():
    out = NPRNet().eval()(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 1)


def test_npr_residual_is_zero_on_locally_constant_input():
    """x - upsample(downsample(x)) must vanish where the image is flat: that is
    the whole premise of the feature."""
    x = torch.ones(1, 3, 32, 32)
    assert torch.allclose(NPRNet._npr(x), torch.zeros_like(x), atol=1e-6)


def test_npr_residual_is_nonzero_on_high_frequency_input():
    x = torch.zeros(1, 3, 32, 32)
    x[..., ::2] = 1.0  # alternating columns: pure 2-pixel-scale detail
    assert NPRNet._npr(x).abs().max() > 0.1
