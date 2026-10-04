# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for P4 Efficient Channel Attention."""

import torch
from torch import nn

from rfdetr.models.backbone.projector import MultiScaleProjector
from rfdetr.models.eca import ECAAttention


def test_eca_starts_as_identity() -> None:
    """Adding ECA must not immediately shift pretrained feature magnitudes."""
    attention = ECAAttention(channels=8)
    features = torch.randn(2, 8, 5, 5)

    output = attention(features)

    torch.testing.assert_close(output, features)


def test_projector_applies_eca_after_c2f_and_layernorm() -> None:
    """Only P4 is reweighted, using the completed C2f/LayerNorm output."""
    projector = MultiScaleProjector([8, 8], 8, [2.0, 1.0, 0.5], num_blocks=1).eval()
    assert projector.p4_eca is not None
    nn.init.ones_(projector.p4_eca.channel_conv.weight)
    features = [torch.randn(1, 8, 4, 4), torch.randn(1, 8, 4, 4)]
    normalized, attention_inputs = [], []
    hooks = [
        projector.stages[1][1].register_forward_hook(
            lambda module, args, output: normalized.append(output.detach().clone())
        ),
        projector.p4_eca.register_forward_pre_hook(
            lambda module, args: attention_inputs.append(args[0].detach().clone())
        ),
    ]
    try:
        outputs = projector(features)
    finally:
        for hook in hooks:
            hook.remove()
    assert len(normalized) == len(attention_inputs) == 1
    assert attention_inputs[0].shape == (1, 8, 4, 4)
    torch.testing.assert_close(attention_inputs[0], normalized[0])
    torch.testing.assert_close(outputs[1], projector.p4_eca(normalized[0]))
    assert not torch.equal(outputs[1], normalized[0])
    assert [tuple(output.shape) for output in outputs] == [(1, 8, 8, 8), (1, 8, 4, 4), (1, 8, 2, 2)]
    outputs[1].square().mean().backward()
    assert projector.p4_eca.channel_conv.weight.grad is not None
    assert torch.isfinite(projector.p4_eca.channel_conv.weight.grad).all()
    with torch.no_grad():
        projector.p4_eca.channel_conv.weight.zero_()
    identity_outputs = projector(features)
    torch.testing.assert_close(identity_outputs[1], normalized[0])
    torch.testing.assert_close(outputs[0], identity_outputs[0])
    torch.testing.assert_close(outputs[2], identity_outputs[2])


def test_projector_eca_ignores_padding() -> None:
    """Padding is resized and excluded from the post-normalization descriptor."""
    projector = MultiScaleProjector([8, 8], 8, [1.0], num_blocks=1).eval()
    nn.init.ones_(projector.p4_eca.channel_conv.weight)
    features = [torch.randn(1, 8, 4, 4) for _ in range(2)]
    mask = torch.zeros(1, 8, 8, dtype=torch.bool)
    mask[:, 4:, :] = True
    normalized = []
    hook = projector.stages[0][1].register_forward_hook(
        lambda module, args, output: normalized.append(output.detach().clone())
    )
    try:
        outputs = projector(features, padding_mask=mask)
    finally:
        hook.remove()
    expected = projector.p4_eca(normalized[0], mask[:, ::2, ::2])
    torch.testing.assert_close(outputs[0], expected)


def test_projector_without_p4_has_no_eca() -> None:
    """Models without a P4 scale do not acquire an attention module."""
    projector = MultiScaleProjector([8], 8, [2.0, 0.5], num_blocks=1).eval()
    assert projector.p4_eca is None
    assert len(projector([torch.ones(1, 8, 4, 4)])) == 2
