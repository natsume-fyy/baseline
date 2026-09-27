# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Convolutional block attention for DETR query features."""

import torch
from torch import nn


class CBAM(nn.Module):
    """Apply channel and query attention to decoder hidden states.

    The original CBAM operates on image feature maps and calls its second stage
    spatial attention. At the detection head, RF-DETR features are query
    sequences rather than image grids, so the same stage operates along the
    query dimension. Any leading dimensions are supported; the final two
    dimensions must be ``(num_queries, channels)``.

    Args:
        channels: Size of the decoder feature dimension.
        reduction: Reduction ratio used by the shared channel MLP.
        kernel_size: Convolution kernel size used for query attention.
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if reduction <= 0:
            raise ValueError(f"reduction must be positive, got {reduction}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")

        reduced_channels = max(channels // reduction, 1)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
        )
        self.query_conv = nn.Conv1d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Refine query features while preserving their shape.

        Args:
            features: Tensor whose final dimensions are
                ``(num_queries, channels)``.

        Returns:
            Attention-weighted tensor with the same shape as ``features``.
        """
        if features.ndim < 3:
            raise ValueError(f"CBAM expects at least 3 dimensions, got shape {tuple(features.shape)}")

        average_channel = features.mean(dim=-2, keepdim=True)
        maximum_channel = features.amax(dim=-2, keepdim=True)
        channel_attention = self.sigmoid(
            self.channel_mlp(average_channel) + self.channel_mlp(maximum_channel)
        )
        features = features * channel_attention

        average_query = features.mean(dim=-1, keepdim=True)
        maximum_query = features.amax(dim=-1, keepdim=True)
        query_descriptors = torch.cat((average_query, maximum_query), dim=-1)
        num_queries = query_descriptors.shape[-2]
        flattened = query_descriptors.reshape(-1, num_queries, 2).transpose(1, 2)
        query_attention = self.sigmoid(self.query_conv(flattened))
        query_attention = query_attention.transpose(1, 2).reshape(*features.shape[:-1], 1)
        return features * query_attention
