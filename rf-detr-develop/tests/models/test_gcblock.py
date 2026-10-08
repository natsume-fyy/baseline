# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Golden Cudgel branch optimization and reparameterization checks."""

import copy

import pytest
import torch

from rfdetr.models.gcblock import GCBlock


def test_gcblock_shape_input_and_gradients() -> None:
    """All train-time paths learn without modifying main-path input storage."""
    module = GCBlock(8)
    features = torch.randn(2, 8, 7, 9, requires_grad=True)
    original = features.detach().clone()
    result = module(features)
    assert result.shape == features.shape
    torch.testing.assert_close(features.detach(), original, rtol=0, atol=0)
    result.square().mean().backward()
    for path in module.paths:
        assert path[0].weight.grad is not None
        assert torch.isfinite(path[0].weight.grad).all()
        assert path[0].weight.grad.abs().sum() > 0
    assert features.grad is not None
    assert features.grad.abs().sum() > 0


@pytest.mark.parametrize("act", [False, True])
def test_reparameterization_equivalence(act: bool) -> None:
    """Fused convolution matches all paths with nontrivial BN statistics."""
    torch.manual_seed(0)
    module = GCBlock(4, act=act)
    for _ in range(3):
        module(torch.randn(2, 4, 5, 7) + 2)
    module.eval()
    features = torch.randn(2, 4, 5, 7)
    expected = module(features)
    fused = copy.deepcopy(module)
    fused.switch_to_deploy()
    torch.testing.assert_close(fused(features), expected, atol=2e-5, rtol=2e-5)
    fused.switch_to_deploy()
    torch.testing.assert_close(fused(features), expected, atol=2e-5, rtol=2e-5)


def test_fusion_rejects_training_mode() -> None:
    """BN running-statistics fusion must not silently replace train-time behavior."""
    with pytest.raises(RuntimeError, match="eval"):
        GCBlock(4).switch_to_deploy()
