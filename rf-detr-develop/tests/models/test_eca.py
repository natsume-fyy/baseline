# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for Efficient Channel Attention."""

import torch
from torch import nn

from rfdetr.models.eca import ECAAttention, QueryECAAttention


def test_eca_starts_as_identity() -> None:
    """Adding ECA must not immediately shift pretrained feature magnitudes."""
    attention = ECAAttention(channels=8)
    features = torch.randn(2, 8, 5, 5)

    output = attention(features)

    torch.testing.assert_close(output, features)


def test_query_eca_starts_as_identity() -> None:
    """Moving ECA to the head must preserve pretrained predictions initially."""
    attention = QueryECAAttention(channels=8)
    features = torch.randn(3, 2, 100, 8)

    output = attention(features)

    torch.testing.assert_close(output, features)


def test_query_eca_reweights_decoder_channels() -> None:
    """Query ECA should reweight channels without changing decoder tensor shape."""
    attention = QueryECAAttention(channels=4)
    nn.init.ones_(attention.channel_conv.weight)
    features = torch.ones(3, 2, 100, 4)

    output = attention(features)

    assert output.shape == features.shape
    assert not torch.equal(output, features)
