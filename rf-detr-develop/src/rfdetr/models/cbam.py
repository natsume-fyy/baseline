# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Channel and spatial attention for projected backbone features."""

import torch
from torch import nn


class CBAMAttention(nn.Module):
    """Apply sequential channel and spatial attention to a feature map.

    This is a Convolutional Block Attention Module (CBAM). Its channel branch
    combines global average- and max-pooled descriptors through a shared MLP;
    its spatial branch combines channel-wise average and maximum maps through
    a convolution. Both sigmoid gates are scaled by two and zero-initialized,
    making the module an identity when inserted into a pretrained detector.

    Args:
        channels: Number of input and output feature channels.
        reduction: Channel-MLP reduction ratio.
        spatial_kernel_size: Kernel size of the spatial-attention convolution.
    """

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel_size: int = 7) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        if reduction <= 0:
            raise ValueError(f"reduction must be positive, got {reduction}.")
        if spatial_kernel_size <= 0 or spatial_kernel_size % 2 == 0:
            raise ValueError("spatial_kernel_size must be a positive odd integer.")

        hidden_channels = max(channels // reduction, 1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.spatial_conv = nn.Conv2d(
            2,
            1,
            kernel_size=spatial_kernel_size,
            padding=spatial_kernel_size // 2,
            bias=False,
        )
        nn.init.zeros_(self.channel_mlp[-1].weight)
        nn.init.zeros_(self.spatial_conv.weight)

    def _channel_weights(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Calculate a broadcastable CBAM channel gate."""
        if padding_mask is None:
            avg_descriptor = features.mean(dim=(-2, -1), keepdim=True)
            max_descriptor = features.amax(dim=(-2, -1), keepdim=True)
        else:
            valid = (~padding_mask).unsqueeze(1)
            valid_float = valid.to(dtype=features.dtype)
            valid_count = valid_float.sum(dim=(-2, -1), keepdim=True).clamp_min(1)
            avg_descriptor = (features * valid_float).sum(dim=(-2, -1), keepdim=True) / valid_count

            masked_features = features.masked_fill(~valid, torch.finfo(features.dtype).min)
            max_descriptor = masked_features.amax(dim=(-2, -1), keepdim=True)
            has_valid_pixel = valid.any(dim=(-2, -1), keepdim=True)
            max_descriptor = torch.where(has_valid_pixel, max_descriptor, torch.zeros_like(max_descriptor))

        logits = self.channel_mlp(avg_descriptor) + self.channel_mlp(max_descriptor)
        return 2 * logits.sigmoid()

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Reweight a ``(B, C, H, W)`` projector feature map.

        Args:
            features: Projected backbone feature map.
            padding_mask: Optional ``(B, H, W)`` mask where ``True`` marks padding.

        Returns:
            Joint channel-spatially attended features with unchanged shape.
        """
        channel_refined = features * self._channel_weights(features, padding_mask)
        spatial_avg = channel_refined.mean(dim=1, keepdim=True)
        spatial_max = channel_refined.amax(dim=1, keepdim=True)
        spatial_descriptor = torch.cat((spatial_avg, spatial_max), dim=1)
        if padding_mask is not None:
            spatial_descriptor = spatial_descriptor.masked_fill(padding_mask.unsqueeze(1), 0)
        spatial_weights = 2 * self.spatial_conv(spatial_descriptor).sigmoid()
        return channel_refined * spatial_weights
