# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""SE shape preservation, source formula, padding and gradient checks."""

import pytest
import torch

from rfdetr.models.se_layer import SELayer


@pytest.mark.parametrize("channels", [1, 8, 24, 256])
def test_shape_and_gradient(channels: int) -> None:
    """Every supported channel count retains shape and propagates gradients."""
    module = SELayer(channels)
    features = torch.randn(2, channels, 5, 7, requires_grad=True)
    original = features.detach().clone()
    output = module(features)
    assert output.shape == features.shape
    torch.testing.assert_close(features.detach(), original, rtol=0, atol=0)
    output.square().mean().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())


def test_source_formula() -> None:
    """With six valid groups, compute exactly the uploaded module's expression."""
    module = SELayer(24, num_context=6)
    features = torch.randn(2, 24, 5, 7)
    descriptor = module.avg_pool(features)
    context = module.context_attention(descriptor).repeat(1, 1, module.context_channel, 1)
    context = context.view(-1, module.channel, 1, 1)
    expected = features * (context * module.channel_attention(descriptor)).expand_as(features)
    torch.testing.assert_close(module(features), expected)


def test_256_channels_and_invalid_groups() -> None:
    """RF-DETR's 256 channels use four groups; invalid explicit counts fail early."""
    assert SELayer(256).num_context == 4
    with pytest.raises(ValueError, match="divide"):
        SELayer(256, num_context=6)


def test_padding_invariance() -> None:
    """Padding must neither affect the descriptor nor change padded values."""
    module = SELayer(8)
    image = torch.randn(1, 8, 3, 5)
    padded = torch.randn(1, 8, 7, 9) * 100
    padded[:, :, :3, :5] = image
    mask = torch.ones(1, 7, 9, dtype=torch.bool)
    mask[:, :3, :5] = False
    output = module(padded, mask)
    torch.testing.assert_close(output[:, :, :3, :5], module(image))
    torch.testing.assert_close(output.masked_select(mask[:, None]), padded.masked_select(mask[:, None]))


def test_export_trace() -> None:
    """Export-style unmasked inference includes the SE operation."""
    module = SELayer(8).eval()
    features = torch.randn(1, 8, 5, 7)
    traced = torch.jit.trace(module, features)
    torch.testing.assert_close(traced(features), module(features))
