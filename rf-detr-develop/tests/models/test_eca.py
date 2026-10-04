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


def test_projector_applies_eca_after_concat_before_c2f() -> None:
    """C2f receives the reweighted concatenation only on the P4 branch."""
    projector = MultiScaleProjector([8, 8], 8, [2.0, 1.0, 0.5], num_blocks=1).eval()
    assert projector.p4_eca is not None
    nn.init.ones_(projector.p4_eca.channel_conv.weight)
    features = [torch.ones(1, 8, 4, 4), torch.full((1, 8, 4, 4), 2.0)]
    before, after = [], []
    hooks = [
        projector.p4_eca.register_forward_pre_hook(lambda module, args: before.append(args[0].detach().clone())),
        projector.stages[1][0].register_forward_pre_hook(lambda module, args: after.append(args[0].detach().clone())),
    ]
    try:
        outputs = projector(features)
    finally:
        for hook in hooks:
            hook.remove()
    concatenated = torch.cat(features, dim=1)
    assert len(before) == len(after) == 1
    torch.testing.assert_close(before[0], concatenated)
    torch.testing.assert_close(after[0], projector.p4_eca(concatenated))
    assert not torch.equal(after[0], concatenated)
    assert [tuple(output.shape) for output in outputs] == [(1, 8, 8, 8), (1, 8, 4, 4), (1, 8, 2, 2)]
    outputs[1].sum().backward()
    assert projector.p4_eca.channel_conv.weight.grad is not None
    with torch.no_grad():
        projector.p4_eca.channel_conv.weight.zero_()
    identity_outputs = projector(features)
    torch.testing.assert_close(outputs[0], identity_outputs[0])
    torch.testing.assert_close(outputs[2], identity_outputs[2])


def test_projector_eca_ignores_padding() -> None:
    """Padding is resized and excluded from the pre-fusion channel descriptor."""
    projector = MultiScaleProjector([8, 8], 8, [1.0], num_blocks=1).eval()
    nn.init.ones_(projector.p4_eca.channel_conv.weight)
    features = [torch.ones(1, 8, 4, 4) for _ in range(2)]
    for feature in features:
        feature[:, :, 2:, :] = 100
    mask = torch.zeros(1, 8, 8, dtype=torch.bool)
    mask[:, 4:, :] = True
    captured = []
    hook = projector.stages[0][0].register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone())
    )
    try:
        projector(features, padding_mask=mask)
    finally:
        hook.remove()
    expected = projector.p4_eca(torch.cat(features, dim=1), mask[:, ::2, ::2])
    torch.testing.assert_close(captured[0], expected)


def test_projector_without_p4_has_no_eca() -> None:
    """Models without a P4 scale do not acquire an attention module."""
    projector = MultiScaleProjector([8], 8, [2.0, 0.5], num_blocks=1).eval()
    assert projector.p4_eca is None
    assert len(projector([torch.ones(1, 8, 4, 4)])) == 2
