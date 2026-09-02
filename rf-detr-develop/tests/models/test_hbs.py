# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for gated residual HBS feature enhancement."""

import pytest
import torch
from torch import nn

from rfdetr.models.hbs import HBS


class _AddConstant(nn.Module):
    """Return a deterministic smoothed feature for formula verification."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Add the configured constant to the input feature."""
        return features + self.value


class _ConstantLogit(nn.Module):
    """Return a spatial gate logit independent of the input feature."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return one gate logit per spatial position."""
        return torch.full(
            (features.shape[0], 1, features.shape[2], features.shape[3]),
            self.value,
            dtype=features.dtype,
            device=features.device,
        )


class TestHBS:
    """Verify HBS implements the requested main-path residual fusion."""

    def test_gated_residual_formula(self) -> None:
        """Output must equal F + alpha * (1 - A) * (S(F) - F)."""
        hbs = HBS(channels=1, kernel_sizes=[3], reduction=1, alpha_init=0.5)
        hbs.denoisers[0] = _AddConstant(2.0)
        hbs.detail_gates[0] = _ConstantLogit(0.0)
        feature = torch.ones(1, 1, 2, 2)

        output = hbs([feature])[0]

        assert torch.allclose(output, torch.full_like(feature, 1.5))

    def test_padding_locations_remain_unchanged(self) -> None:
        """Padded feature positions must not receive the residual update."""
        hbs = HBS(channels=1, kernel_sizes=[3], reduction=1, alpha_init=1.0)
        hbs.denoisers[0] = _AddConstant(2.0)
        hbs.detail_gates[0] = _ConstantLogit(0.0)
        feature = torch.ones(1, 1, 2, 2)
        padding_mask = torch.tensor([[[False, True], [False, False]]])

        output = hbs([feature], [padding_mask])[0]

        assert output[0, 0, 0, 1].item() == pytest.approx(1.0)
        assert output[0, 0, 0, 0].item() == pytest.approx(2.0)

    def test_all_components_receive_gradients(self) -> None:
        """Main detection loss must train the smoother, gate, and residual scale."""
        hbs = HBS(channels=4, kernel_sizes=[3], reduction=2, alpha_init=0.1)
        feature = torch.randn(2, 4, 5, 5, requires_grad=True)

        hbs([feature])[0].square().mean().backward()

        assert hbs.alphas.grad is not None
        assert hbs.detail_gates[0].weight.grad is not None
        assert hbs.denoisers[0].conv_block[0].weight.grad is not None

    def test_rejects_feature_level_mismatch(self) -> None:
        """Each projected feature level must have its own HBS components."""
        hbs = HBS(channels=4, kernel_sizes=[3, 5], reduction=2)

        with pytest.raises(ValueError, match="Expected 2 feature levels"):
            hbs([torch.randn(1, 4, 5, 5)])
