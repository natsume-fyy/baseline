# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for P4 Efficient Channel Attention."""

import torch
from torch import nn

from rfdetr.models.eca import ECAAttention
from rfdetr.models.lwdetr import LWDETR
from rfdetr.utilities.tensors import NestedTensor


def test_eca_starts_as_identity() -> None:
    """Adding ECA must not immediately shift pretrained feature magnitudes."""
    attention = ECAAttention(channels=8)
    features = torch.randn(2, 8, 5, 5)

    output = attention(features)

    torch.testing.assert_close(output, features)


def test_lwdetr_applies_eca_only_to_p4() -> None:
    """The configured P4 level is reweighted without changing other levels or masks."""
    model = LWDETR.__new__(LWDETR)
    nn.Module.__init__(model)
    model.p4_feature_index = 1
    model.p4_eca = ECAAttention(channels=4)
    nn.init.ones_(model.p4_eca.channel_conv.weight)
    masks = [torch.zeros(1, size, size, dtype=torch.bool) for size in (8, 4, 2)]
    features = [
        NestedTensor(torch.ones(1, 4, size, size), mask)
        for size, mask in zip((8, 4, 2), masks)
    ]

    outputs = model._apply_p4_eca(features)

    assert outputs is not None
    assert outputs[0] is features[0]
    assert outputs[2] is features[2]
    assert outputs[1].mask is masks[1]
    assert not torch.equal(outputs[1].tensors, features[1].tensors)
