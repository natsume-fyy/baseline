# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for target-guided auxiliary structure enhancement."""

import torch

from rfdetr.models.target_mid_low_frequency import TargetMidLowFrequency


def test_initial_identity() -> None:
    """Zero projection preserves the pretrained representation at initialization."""
    module = TargetMidLowFrequency(8)
    features = torch.randn(2, 8, 7, 9)
    targets = [{"boxes": torch.tensor([[0.5, 0.5, 1.0, 1.0]])} for _ in range(2)]
    torch.testing.assert_close(module(features, targets), features, rtol=0, atol=0)


def test_target_region_and_input_preservation() -> None:
    """A learned branch changes target cells only, without mutating main features."""
    module = TargetMidLowFrequency(4)
    torch.nn.init.ones_(module.project.bias)
    features = torch.randn(1, 4, 4, 4)
    original = features.clone()
    targets = [{"boxes": torch.tensor([[0.25, 0.25, 0.5, 0.5]])}]
    expected = features.clone()
    expected[:, :, :2, :2] += 1
    torch.testing.assert_close(module(features, targets), expected)
    torch.testing.assert_close(features, original, rtol=0, atol=0)


def test_padding_invariance() -> None:
    """Batch padding must not alter valid auxiliary features or be enhanced."""
    module = TargetMidLowFrequency(4)
    torch.nn.init.normal_(module.project.weight, std=0.1)
    image = torch.randn(1, 4, 5, 7)
    targets = [{"boxes": torch.tensor([[0.5, 0.5, 1.0, 1.0]])}]
    padded = torch.randn(1, 4, 9, 11) * 100
    padded[:, :, :5, :7] = image
    mask = torch.ones(1, 9, 11, dtype=torch.bool)
    mask[:, :5, :7] = False
    result = module(padded, targets, mask)
    torch.testing.assert_close(result[:, :, :5, :7], module(image, targets))
    torch.testing.assert_close(result.masked_select(mask[:, None]), padded.masked_select(mask[:, None]))


def test_empty_target_identity() -> None:
    """No target means no enhancement even after the branch has learned."""
    module = TargetMidLowFrequency(4)
    torch.nn.init.normal_(module.project.weight)
    features = torch.randn(1, 4, 5, 7)
    targets = [{"boxes": torch.empty(0, 4)}]
    torch.testing.assert_close(module(features, targets), features, rtol=0, atol=0)


def test_sub_cell_target_preserved() -> None:
    """Tiny objects retain at least one foreground cell."""
    valid = torch.ones(1, 1, 4, 4)
    targets = [{"boxes": torch.tensor([[0.3, 0.3, 0.001, 0.001]])}]
    mask = TargetMidLowFrequency._foreground_mask(valid, targets)
    assert mask.sum() == 1
    assert mask[0, 0, 1, 1] == 1


def test_learning_and_backbone_gradient() -> None:
    """The branch learns past zero initialization and retains backbone gradients."""
    torch.manual_seed(0)
    module = TargetMidLowFrequency(4)
    features = torch.randn(1, 4, 5, 7, requires_grad=True)
    targets = [{"boxes": torch.tensor([[0.5, 0.5, 1.0, 1.0]])}]
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    for _ in range(2):
        optimizer.zero_grad()
        features.grad = None
        module(features, targets).square().mean().backward()
        optimizer.step()
    for parameter in (features, module.reduce.weight, module.project.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
