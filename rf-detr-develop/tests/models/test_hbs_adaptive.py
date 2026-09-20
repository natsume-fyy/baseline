# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for image-adaptive HBS strength."""

import torch
from torch import nn

from rfdetr.models.hbs import HBS


class AddOne(nn.Module):
    """Create a unit HBS residual so that the applied strength is observable."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Add one at every spatial position."""
        return features + 1


def _boxes(count: int, size: float) -> torch.Tensor:
    """Create non-overlapping-enough normalized boxes for gate tests."""
    if count == 0:
        return torch.empty(0, 4)
    positions = torch.linspace(0.05, 0.95, count)
    return torch.stack(
        (
            positions,
            positions.flip(0),
            torch.full_like(positions, size),
            torch.full_like(positions, size),
        ),
        dim=1,
    )


def test_adaptive_weight_is_higher_for_sparse_than_crowded_small_objects() -> None:
    """Many small objects must suppress HBS even when most pixels are background."""
    hbs = HBS(channels=1, kernel_sizes=[3], adaptive=True)
    feature = torch.zeros(2, 1, 20, 20)
    valid = torch.ones(2, 1, 20, 20)
    foreground = torch.zeros_like(valid)
    targets = [{"boxes": _boxes(1, 0.05)}, {"boxes": _boxes(30, 0.02)}]

    weights = hbs._adaptive_weights(feature, foreground, valid, targets)

    assert weights[0] > weights[1]
    assert weights[0] > 0.5
    assert weights[1] < 0.25


def test_adaptive_weight_increases_with_background_high_frequency() -> None:
    """At equal target density, complex background should receive more HBS."""
    hbs = HBS(channels=1, kernel_sizes=[3], adaptive=True)
    checker = (torch.arange(16).reshape(4, 4) % 2).float() * 2 - 1
    feature = torch.stack((torch.ones(1, 4, 4), checker.unsqueeze(0)))
    valid = torch.ones(2, 1, 4, 4)
    foreground = torch.zeros_like(valid)
    targets = [{"boxes": _boxes(3, 0.1)}, {"boxes": _boxes(3, 0.1)}]

    weights = hbs._adaptive_weights(feature, foreground, valid, targets)

    assert weights[1] > weights[0]


def test_adaptive_weight_scales_only_the_hbs_background_residual() -> None:
    """The per-image gate must preserve foreground and scale background changes."""
    hbs = HBS(channels=1, kernel_sizes=[3], adaptive=True)
    hbs.denoisers[0] = AddOne()
    feature = torch.zeros(1, 1, 4, 4)
    targets = [{"boxes": torch.tensor([[0.375, 0.375, 0.25, 0.25]])}]

    output = hbs([feature], targets)[0]

    assert output[0, 0, 1, 1] == 0
    assert 0 < output[0, 0, 0, 0] < 1


def test_non_adaptive_hbs_keeps_full_legacy_strength() -> None:
    """Adaptive gating must be opt-in for checkpoint and experiment compatibility."""
    hbs = HBS(channels=1, kernel_sizes=[3], adaptive=False)
    hbs.denoisers[0] = AddOne()
    feature = torch.zeros(1, 1, 3, 3)

    output = hbs([feature], [{"boxes": torch.empty(0, 4)}])[0]

    assert torch.equal(output, torch.ones_like(feature))
