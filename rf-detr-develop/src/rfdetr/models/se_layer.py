# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""SELayer adapted from the user-provided code.zip/model.py.

Preserves its context-group attention multiplied by grouped channel attention.
Selects the largest compatible group count up to six when not explicitly set
(256 channels, reduction four -> four groups), without changing output channels.
"""

import math

import torch
from torch import nn


class SELayer(nn.Module):
    """Reweight C channels using context and channel gates, preserving B,C,H,W.

    Args:
        channel: Number of input and output channels.
        reduction: Channel-attention bottleneck reduction.
        num_context: Explicit group count, or choose a compatible count up to six.
    """

    def __init__(self, channel: int, reduction: int = 4, num_context: int | None = None) -> None:
        super().__init__()
        if channel < 1 or reduction < 1:
            raise ValueError("channel and reduction must be positive")
        hidden = max(1, channel // reduction)
        if num_context is None:
            common = math.gcd(channel, hidden)
            num_context = next(group for group in range(min(6, common), 0, -1) if common % group == 0)
        if num_context < 1 or channel % num_context or hidden % num_context:
            raise ValueError("num_context must divide both channel and bottleneck channel counts")
        self.channel = channel
        self.num_context = num_context
        self.context_channel = channel // num_context
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.context_attention = nn.Sequential(
            nn.Conv2d(channel, max(1, channel // 2), 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(max(1, channel // 2), num_context, 1, bias=False),
            nn.Sigmoid(),
        )
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channel, hidden, 1, groups=num_context, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, channel, 1, groups=num_context, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the source module's multiplicative gates, excluding padded cells."""
        if padding_mask is None:
            descriptor = self.avg_pool(features)
        else:
            valid = ~padding_mask[:, None]
            clean = features.masked_fill(~valid, 0)
            descriptor = clean.sum(dim=(-2, -1), keepdim=True) / valid.sum(dim=(-2, -1), keepdim=True).clamp_min(1)
        context = self.context_attention(descriptor).repeat_interleave(self.context_channel, dim=1)
        attention = context * self.channel_attention(descriptor)
        result = features * attention
        return result if padding_mask is None else torch.where(padding_mask[:, None], features, result)
