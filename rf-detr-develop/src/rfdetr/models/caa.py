# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Context Anchor Attention adapted for a training-only detection branch.

Architecture reference: PKINet (CVPR 2024), Context Anchor Attention:
https://github.com/PKINet/PKINet/blob/main/mmrotate/models/backbones/pkinet.py
Uses local pooling, pointwise mixing and horizontal/vertical depthwise kernels.
This adaptation omits batch normalization, masks padding between operations,
and centers the sigmoid scale at one for identity initialization.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class CAAAttention(nn.Module):
    """Reweight projected features with channel-dependent spatial context.

    Args:
        channels: Projected feature channels.
        kernel_size: Odd length of the horizontal and vertical context kernels.
    """

    def __init__(self, channels: int, kernel_size: int = 11) -> None:
        super().__init__()
        if channels < 1 or kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("channels must be positive and kernel_size must be positive and odd")
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.activation = nn.SiLU()
        self.horizontal = nn.Conv2d(
            channels, channels, (1, kernel_size), padding=(0, kernel_size // 2), groups=channels
        )
        self.vertical = nn.Conv2d(
            channels, channels, (kernel_size, 1), padding=(kernel_size // 2, 0), groups=channels
        )
        self.conv2 = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return attended features without changing input storage or padded values."""
        valid = torch.ones_like(features[:, :1]) if padding_mask is None else (~padding_mask[:, None]).to(features.dtype)
        clean = features.masked_fill(valid == 0, 0)
        pooled = F.avg_pool2d(clean, 7, stride=1, padding=3)
        support = F.avg_pool2d(valid, 7, stride=1, padding=3).clamp_min(1e-6)
        context = self.activation(self.conv1(pooled / support)).masked_fill(valid == 0, 0)
        context = self.horizontal(context).masked_fill(valid == 0, 0)
        context = self.vertical(context).masked_fill(valid == 0, 0)
        weights = 2 * self.conv2(context).sigmoid()
        return torch.where(valid.bool(), features * weights, features)
