"""Tests for the degradation-aware parallel feature branch."""

import torch

from rfdetr.models.degradation_aware import DegradationAwareFusion, DegradationAwareModule


def test_degradation_aware_module_preserves_shape_and_dtype() -> None:
    module = DegradationAwareModule(channels=4)
    feature = torch.randn(2, 4, 7, 9)

    output = module(feature)

    assert output.shape == feature.shape
    assert output.dtype == feature.dtype


def test_degradation_aware_fusion_starts_as_identity_and_learns_scale() -> None:
    fusion = DegradationAwareFusion(channels=4)
    feature = torch.randn(2, 4, 7, 9)

    output = fusion(feature)

    torch.testing.assert_close(output, feature, rtol=0, atol=0)
    output.sum().backward()
    assert fusion.fusion_scale.grad is not None
