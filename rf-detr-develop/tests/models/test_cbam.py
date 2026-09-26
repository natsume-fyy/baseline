# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for Convolutional Block Attention Module (CBAM)."""

import torch
from torch import nn

from rfdetr.models.cbam import CBAMAttention, QueryCBAMAttention


def test_cbam_starts_as_identity() -> None:
    """CBAM insertion must initially preserve pretrained projector features."""
    attention = CBAMAttention(channels=8)
    features = torch.randn(2, 8, 5, 5)

    output = attention(features)

    torch.testing.assert_close(output, features)


def test_cbam_preserves_shape_and_reweights_channel_and_space() -> None:
    """Both CBAM branches should reweight features without changing their shape."""
    attention = CBAMAttention(channels=4, reduction=2)
    nn.init.ones_(attention.channel_mlp[-1].weight)
    nn.init.ones_(attention.spatial_conv.weight)
    features = torch.randn(2, 4, 6, 5)

    output = attention(features)

    assert output.shape == features.shape
    assert not torch.equal(output, features)


def test_cbam_ignores_padding_in_channel_pooling() -> None:
    """Padded pixels must not influence the channel descriptor."""
    attention = CBAMAttention(channels=4, reduction=2)
    nn.init.ones_(attention.channel_mlp[-1].weight)
    base = torch.randn(1, 4, 3, 3)
    padded = torch.nn.functional.pad(base, (0, 2, 0, 2), value=1000.0)
    mask = torch.ones(1, 5, 5, dtype=torch.bool)
    mask[:, :3, :3] = False

    base_channel_weights = attention._channel_weights(base, None)
    padded_channel_weights = attention._channel_weights(padded, mask)

    torch.testing.assert_close(padded_channel_weights, base_channel_weights)


def test_query_cbam_starts_as_identity() -> None:
    """Head CBAM insertion must initially preserve decoder query features."""
    attention = QueryCBAMAttention(channels=8)
    features = torch.randn(3, 2, 100, 8)

    output = attention(features)

    torch.testing.assert_close(output, features)


def test_query_cbam_reweights_channels_and_queries() -> None:
    """Query CBAM should jointly reweight both axes without changing shape."""
    attention = QueryCBAMAttention(channels=4, reduction=2)
    nn.init.ones_(attention.channel_mlp[-1].weight)
    nn.init.ones_(attention.query_conv.weight)
    features = torch.ones(3, 2, 10, 4)

    output = attention(features)

    assert output.shape == features.shape
    assert not torch.equal(output, features)
