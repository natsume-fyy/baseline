# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Efficient Channel Attention for projected backbone features."""

import math

import torch
from torch import nn


class ECAAttention(nn.Module):
    """Apply lightweight local cross-channel attention.

    The channel interaction kernel follows the ECA-Net adaptation rule. The
    output scale is centered on one so insertion into a pretrained detector
    initially preserves its feature range.

    Args:
        channels: Number of feature channels.
        gamma: Kernel-size adaptation factor.
        bias: Kernel-size adaptation offset.
    """

    def __init__(self, channels: int, gamma: int = 2, bias: int = 1) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        kernel_size = int(abs((math.log2(channels) + bias) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        kernel_size = max(kernel_size, 1)
        self.channel_conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
        nn.init.zeros_(self.channel_conv.weight)

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Reweight a ``(B, C, H, W)`` feature map channel by channel.

        Args:
            features: Projected backbone feature map.
            padding_mask: Optional ``(B, H, W)`` mask where ``True`` marks padding.

        Returns:
            Feature map with the same shape and dtype as the input.
        """
        if padding_mask is None:
            descriptor = features.mean(dim=(-2, -1))
        else:
            valid = (~padding_mask).unsqueeze(1).to(dtype=features.dtype)
            valid_count = valid.sum(dim=(-2, -1)).clamp_min(1)
            descriptor = (features * valid).sum(dim=(-2, -1)) / valid_count
        descriptor = descriptor.unsqueeze(1)
        weights = self.channel_conv(descriptor).sigmoid().squeeze(1).unsqueeze(-1).unsqueeze(-1)
        return features * (2 * weights)
