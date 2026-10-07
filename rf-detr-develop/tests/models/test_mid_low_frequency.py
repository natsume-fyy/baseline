# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Behavioral checks for feature fusion, padding and optimization."""

import pytest
import torch

from rfdetr.models.mid_low_frequency import MidLowFrequencyFusion


@pytest.mark.parametrize("size", [pytest.param(1, id="single-cell"), pytest.param(9, id="spatial")])
def test_initial_identity(size: int) -> None:
    """Enabling the module initially preserves the pretrained representation."""
    module = MidLowFrequencyFusion(8)
    features = torch.randn(2, 8, size, size)
    torch.testing.assert_close(module(features), features, rtol=0, atol=0)


def test_padding_invariance() -> None:
    """Other image sizes in a batch must not change an image's valid output."""
    module = MidLowFrequencyFusion(8)
    torch.nn.init.normal_(module.project.weight, std=0.1)
    image = torch.randn(1, 8, 5, 7)
    padded = torch.randn(1, 8, 9, 11) * 100
    padded[:, :, :5, :7] = image
    mask = torch.ones(1, 9, 11, dtype=torch.bool)
    mask[:, :5, :7] = False
    result = module(padded, mask)
    torch.testing.assert_close(result[:, :, :5, :7], module(image))
    torch.testing.assert_close(result.masked_select(mask[:, None]), padded.masked_select(mask[:, None]))


def test_constant_has_no_mid_band() -> None:
    """Normalized smoothing must not manufacture edge structure at padding."""
    module = MidLowFrequencyFusion(4)
    valid = torch.ones(1, 1, 7, 9)
    valid[:, :, 4:, :] = 0
    constant = 3 * valid.expand(1, 4, 7, 9)
    mid = module._smooth(constant, valid, module.fine_kernel) - module._smooth(
        constant, valid, module.coarse_kernel
    )
    torch.testing.assert_close(mid[:, :, :4], torch.zeros_like(mid[:, :, :4]))


def test_learns_after_identity_initialization() -> None:
    """The zero output projection must not permanently block branch learning."""
    torch.manual_seed(0)
    module = MidLowFrequencyFusion(8)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    features = torch.randn(2, 8, 7, 9)
    target = torch.randn_like(features)
    for _ in range(2):
        optimizer.zero_grad()
        (module(features) - target).square().mean().backward()
        optimizer.step()
    for parameter in (module.project.weight, module.structure[0].weight, module.gate[-1].weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_fully_padded_image() -> None:
    """All-padding inputs remain finite and unchanged after learning."""
    module = MidLowFrequencyFusion(4)
    torch.nn.init.normal_(module.project.weight)
    features = torch.randn(1, 4, 3, 5)
    mask = torch.ones(1, 3, 5, dtype=torch.bool)
    torch.testing.assert_close(module(features, mask), features)


def test_export_and_state_restore() -> None:
    """Export tracing and checkpoint restoration retain the learned branch."""
    module = MidLowFrequencyFusion(4).eval()
    torch.nn.init.normal_(module.project.weight, std=0.1)
    features = torch.randn(1, 4, 7, 9)
    restored = MidLowFrequencyFusion(4).eval()
    restored.load_state_dict(module.state_dict())
    torch.testing.assert_close(restored(features), module(features))
    traced = torch.jit.trace(module, features)
    torch.testing.assert_close(traced(features), module(features))
