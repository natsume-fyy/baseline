# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Residual CBAM modules for projected multi-scale feature maps."""

import torch
from torch import nn


class CBAM2D(nn.Module):
    """Apply channel and spatial attention to a two-dimensional feature map.

    Args:
        channels: Number of feature channels.
        reduction: Reduction ratio used by the shared channel MLP.
        kernel_size: Spatial-attention convolution kernel size.
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        if reduction <= 0:
            raise ValueError(f"reduction must be positive, got {reduction}.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")

        reduced_channels = max(channels // reduction, 1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False),
        )
        self.spatial_conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def _channel_pool(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool spatial dimensions while excluding padded pixels when available.

        Args:
            features: Feature map with shape ``(B, C, H, W)``.
            padding_mask: Optional mask with shape ``(B, H, W)`` where
                ``True`` marks padding.

        Returns:
            Average-pooled and max-pooled channel descriptors.
        """
        if padding_mask is None:
            return features.mean(dim=(-2, -1), keepdim=True), features.amax(dim=(-2, -1), keepdim=True)

        if padding_mask.shape != (features.shape[0], features.shape[2], features.shape[3]):
            raise ValueError(
                f"Padding mask shape {tuple(padding_mask.shape)} does not match feature shape "
                f"{tuple(features.shape)}."
            )
        valid = (~padding_mask).unsqueeze(1)
        valid_values = valid.to(dtype=features.dtype)
        denominator = valid_values.sum(dim=(-2, -1), keepdim=True).clamp_min(1)
        average = (features * valid_values).sum(dim=(-2, -1), keepdim=True) / denominator
        maximum = features.masked_fill(~valid, torch.finfo(features.dtype).min).amax(
            dim=(-2, -1), keepdim=True
        )
        return average, maximum

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return CBAM-refined features.

        Args:
            features: Feature map with shape ``(B, C, H, W)``.
            padding_mask: Optional mask with shape ``(B, H, W)`` where
                ``True`` marks padding.

        Returns:
            Refined feature map with the same shape as ``features``.
        """
        if features.ndim != 4:
            raise ValueError(f"CBAM2D expects B,C,H,W features, got shape {tuple(features.shape)}.")

        average, maximum = self._channel_pool(features, padding_mask)
        channel_attention = self.sigmoid(self.channel_mlp(average) + self.channel_mlp(maximum))
        refined = features * channel_attention
        spatial_features = (
            refined
            if padding_mask is None
            else refined.masked_fill(padding_mask.unsqueeze(1), 0)
        )

        spatial_descriptors = torch.cat(
            (
                spatial_features.mean(dim=1, keepdim=True),
                spatial_features.amax(dim=1, keepdim=True),
            ),
            dim=1,
        )
        spatial_attention = self.sigmoid(self.spatial_conv(spatial_descriptors))
        return refined * spatial_attention


class ResidualCBAM2D(nn.Module):
    """Wrap CBAM in an identity-initialized residual branch.

    The learnable residual scale starts at zero, so adding this module to a
    pretrained model does not immediately change its predictions.

    Args:
        channels: Number of feature channels.
        reduction: Reduction ratio used by CBAM channel attention.
        kernel_size: Spatial-attention convolution kernel size.
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.attention = CBAM2D(channels, reduction=reduction, kernel_size=kernel_size)
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add a learnable CBAM residual to the input feature map.

        Args:
            features: Feature map with shape ``(B, C, H, W)``.
            padding_mask: Optional mask with shape ``(B, H, W)`` where
                ``True`` marks padding.

        Returns:
            Residually enhanced feature map.
        """
        residual_weight = torch.tanh(self.residual_scale)
        return features + residual_weight * self.attention(features, padding_mask)


class FeaturePyramidCBAM(nn.Module):
    """Apply independent residual CBAM blocks to selected pyramid levels.

    Args:
        channels: Shared channel count of projected feature levels.
        level_indexes: Pyramid indexes to enhance, normally P3 and P4.
    """

    def __init__(self, channels: int, level_indexes: list[int]) -> None:
        super().__init__()
        self.level_indexes = tuple(level_indexes)
        self.attention_blocks = nn.ModuleList([ResidualCBAM2D(channels) for _ in self.level_indexes])

    def forward(
        self,
        features: list[torch.Tensor],
        padding_masks: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        """Enhance selected levels and leave all other levels unchanged.

        Args:
            features: Projected pyramid features in high-to-low resolution order.
            padding_masks: Optional padding mask for each feature level.

        Returns:
            Feature list with CBAM applied only at configured indexes.
        """
        outputs = list(features)
        for level_index, attention in zip(self.level_indexes, self.attention_blocks):
            if level_index >= len(outputs):
                raise ValueError(f"Feature pyramid does not contain configured level index {level_index}.")
            padding_mask = None if padding_masks is None else padding_masks[level_index]
            outputs[level_index] = attention(outputs[level_index], padding_mask)
        return outputs
