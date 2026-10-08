# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Checks for auxiliary CAA and the attention-free main projector."""

import torch

from rfdetr.models.backbone.projector import MultiScaleProjector
from rfdetr.models.caa import CAAAttention


def test_initial_identity() -> None:
    """CAA initially preserves the projected representation."""
    features = torch.randn(2, 8, 5, 7)
    torch.testing.assert_close(CAAAttention(8)(features), features, rtol=0, atol=0)


def test_padding_and_input_preservation() -> None:
    """Batch padding does not affect valid output or mutate the main features."""
    module = CAAAttention(4)
    torch.nn.init.normal_(module.conv2.weight, std=0.1)
    image = torch.randn(1, 4, 5, 7)
    padded = torch.randn(1, 4, 9, 11) * 100
    padded[:, :, :5, :7] = image
    original = padded.clone()
    mask = torch.ones(1, 9, 11, dtype=torch.bool)
    mask[:, :5, :7] = False
    result = module(padded, mask)
    torch.testing.assert_close(result[:, :, :5, :7], module(image))
    torch.testing.assert_close(result.masked_select(mask[:, None]), padded.masked_select(mask[:, None]))
    torch.testing.assert_close(padded, original, rtol=0, atol=0)


def test_branch_learns() -> None:
    """Zero initialization must not permanently block context-kernel gradients."""
    torch.manual_seed(0)
    module = CAAAttention(4)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    features = torch.randn(1, 4, 7, 9, requires_grad=True)
    for _ in range(2):
        optimizer.zero_grad()
        features.grad = None
        module(features).square().mean().backward()
        optimizer.step()
    for parameter in (features, module.conv1.weight, module.horizontal.weight, module.vertical.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_projector_output_is_unmodified() -> None:
    """All scales pass their normalized features directly to the main path."""
    projector = MultiScaleProjector([8, 8], 8, [2.0, 1.0, 0.5], num_blocks=1).eval()
    captured = []
    hooks = [stage.register_forward_hook(lambda m, a, out: captured.append(out.clone())) for stage in projector.stages]
    try:
        outputs = projector([torch.randn(1, 8, 4, 4) for _ in range(2)])
    finally:
        for hook in hooks:
            hook.remove()
    assert len(outputs) == len(captured) == 3
    for output, raw in zip(outputs, captured):
        torch.testing.assert_close(output, raw, rtol=0, atol=0)
