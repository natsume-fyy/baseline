# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for contrast-gated HBS background smoothing."""

import torch
from torch import nn

from rfdetr.models.hbs import HBS


class AddOne(nn.Module):
    """Produce a known background residual for testing the HBS gate."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Add one to every feature position.

        Args:
            features: Input feature map.

        Returns:
            Feature map with a unit residual.
        """
        return features + 1


def test_gate_prefers_intermediate_background_contrast() -> None:
    """The initialized gate should be strongest near intermediate contrast."""
    hbs = HBS(channels=1, kernel_sizes=[3])
    checker = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    features = torch.stack((torch.ones(1, 2, 2), (1 + 0.5 * checker).unsqueeze(0), checker.unsqueeze(0)))
    mask = torch.ones(3, 1, 2, 2)

    gates = hbs._background_gate(features, mask).flatten()

    assert gates[1] > 0.6
    assert gates[0] < 0.1
    assert gates[2] < 0.1


def test_gate_scales_background_only_and_preserves_padding() -> None:
    """Foreground and padding remain identical even when the gate is active."""
    hbs = HBS(channels=1, kernel_sizes=[3])
    hbs.denoisers[0] = AddOne()
    features = [torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)]
    targets = [{"boxes": torch.tensor([[0.375, 0.375, 0.25, 0.25]])}]
    padding = [torch.zeros(1, 4, 4, dtype=torch.bool)]
    padding[0][:, 3, :] = True

    output = hbs(features, targets, padding)[0]
    valid_background = (~padding[0]).unsqueeze(1).float()
    valid_background[:, :, 1, 1] = 0
    gate = hbs._background_gate(features[0], valid_background).item()

    assert torch.equal(output[:, :, 1, 1], features[0][:, :, 1, 1])
    assert torch.equal(output[:, :, 3, :], features[0][:, :, 3, :])
    torch.testing.assert_close(output[0, 0, 0, 0] - features[0][0, 0, 0, 0], torch.tensor(gate))
    assert 0 < gate < 1


def test_gate_handles_empty_valid_background() -> None:
    """An entirely padded feature map must stay finite and unchanged."""
    hbs = HBS(channels=1, kernel_sizes=[3])
    feature = torch.ones(1, 1, 2, 2)

    output = hbs([feature], [{"boxes": torch.empty(0, 4)}], [torch.ones(1, 2, 2, dtype=torch.bool)])[0]

    assert torch.equal(output, feature)
