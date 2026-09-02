# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Fog-adaptive background smoothing for projected multi-scale features."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class BackgroundSmoothingBlock(nn.Module):
    """Learn a residual, spatially smoothed version of one feature level."""

    def __init__(self, channels: int, reduction: int = 4, kernel_size: int = 3) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        if reduction <= 0 or channels // reduction <= 0:
            raise ValueError(
                f"reduction must produce at least one bottleneck channel, got channels={channels}, "
                f"reduction={reduction}."
            )
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")

        padding = (kernel_size - 1) // 2
        bottleneck_channels = channels // reduction
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, bottleneck_channels, kernel_size, stride=1, padding=padding, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, channels, kernel_size, stride=1, padding=padding, bias=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return ``F_HBS`` with the same shape as ``features``."""
        return features + self.conv_block(features)


class FogFrequencyGate(nn.Module):
    """Predict a per-image gate from feature statistics and frequency energy.

    The descriptor contains channel-wise mean/contrast plus two explicit scene
    statistics: normalized global contrast (a haze cue) and high-frequency
    residual energy. The small MLP learns how these cues should control HBS.
    """

    def __init__(self, channels: int, reduction: int = 4, initial_alpha: float = 0.25) -> None:
        super().__init__()
        if not 0.0 < initial_alpha < 1.0:
            raise ValueError(f"initial_alpha must be in (0, 1), got {initial_alpha}.")
        hidden_channels = max(channels // reduction, 4)
        self.predictor = nn.Sequential(
            nn.Linear(2 * channels + 2, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1),
        )

        # Begin conservatively, then learn the dependence on haze/frequency cues.
        final_layer = self.predictor[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.constant_(final_layer.bias, math.log(initial_alpha / (1.0 - initial_alpha)))

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return ``alpha`` shaped ``(B, 1, 1, 1)`` in [0, 1]."""
        # Accumulate global statistics in fp32 to avoid overflow under AMP on
        # high-resolution feature maps.
        stats_features = features.float() if features.dtype in {torch.float16, torch.bfloat16} else features
        if padding_mask is None:
            valid = torch.ones_like(stats_features[:, :1])
        else:
            valid = (~padding_mask).unsqueeze(1).to(dtype=stats_features.dtype)

        spatial_count = valid.sum(dim=(-2, -1)).clamp_min(1.0)
        channel_mean = (stats_features * valid).sum(dim=(-2, -1)) / spatial_count
        centered = (stats_features - channel_mean[:, :, None, None]) * valid
        channel_std = torch.sqrt(centered.square().sum(dim=(-2, -1)) / spatial_count + 1e-6)

        # Local low-pass residual: a differentiable measure of high-frequency energy.
        pooled_valid = F.avg_pool2d(valid, kernel_size=3, stride=1, padding=1)
        low_frequency = F.avg_pool2d(
            stats_features * valid, kernel_size=3, stride=1, padding=1
        ) / pooled_valid.clamp_min(1e-6)
        high_frequency = (stats_features - low_frequency).abs() * valid
        magnitude = (stats_features.abs() * valid).sum(dim=(1, 2, 3)).clamp_min(1e-6)
        frequency_ratio = high_frequency.sum(dim=(1, 2, 3)) / magnitude

        global_mean = channel_mean.mean(dim=1, keepdim=True)
        global_variance = (
            ((stats_features - global_mean[:, :, None, None]) * valid).square().sum(dim=(1, 2, 3))
            / (spatial_count.squeeze(1) * stats_features.shape[1]).clamp_min(1.0)
        )
        global_contrast = torch.sqrt(global_variance + 1e-6) / (
            magnitude / (spatial_count.squeeze(1) * stats_features.shape[1]).clamp_min(1.0) + 1e-6
        )

        descriptor = torch.cat(
            [channel_mean, channel_std, global_contrast[:, None], frequency_ratio[:, None]], dim=1
        )
        descriptor = descriptor.to(dtype=self.predictor[0].weight.dtype)
        return torch.sigmoid(self.predictor(descriptor)).view(-1, 1, 1, 1)


class HBS(nn.Module):
    """Apply fog-adaptive HBS to every projected feature level.

    For each image and scale this module implements
    ``F_out = F + alpha * (F_HBS - F)``. It requires no annotations, so the
    same learned behavior is active during training, evaluation, and export.
    """

    def __init__(
        self,
        channels: int,
        kernel_sizes: list[int],
        reduction: int = 4,
        initial_alpha: float = 0.25,
    ) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("kernel_sizes must contain at least one feature level.")
        self.denoisers = nn.ModuleList(
            [
                BackgroundSmoothingBlock(channels=channels, reduction=reduction, kernel_size=kernel_size)
                for kernel_size in kernel_sizes
            ]
        )
        self.gates = nn.ModuleList(
            [
                FogFrequencyGate(channels=channels, reduction=reduction, initial_alpha=initial_alpha)
                for _ in kernel_sizes
            ]
        )

    def forward(
        self,
        features: list[torch.Tensor],
        padding_masks: list[torch.Tensor | None] | None = None,
        *,
        return_alphas: bool = False,
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Adaptively smooth multi-scale projector outputs."""
        if len(features) != len(self.denoisers):
            raise ValueError(f"Expected {len(self.denoisers)} feature levels, received {len(features)}.")
        if padding_masks is None:
            padding_masks = [None] * len(features)
        if len(padding_masks) != len(features):
            raise ValueError(f"Expected {len(features)} padding masks, received {len(padding_masks)}.")

        outputs: list[torch.Tensor] = []
        alphas: list[torch.Tensor] = []
        for feature, padding_mask, denoiser, gate in zip(
            features, padding_masks, self.denoisers, self.gates
        ):
            valid = (
                torch.ones_like(feature[:, :1])
                if padding_mask is None
                else (~padding_mask).unsqueeze(1).to(dtype=feature.dtype)
            )
            hbs_feature = denoiser(feature * valid)
            alpha = gate(feature, padding_mask)
            fused = feature + alpha * (hbs_feature - feature)
            outputs.append(fused * valid + feature * (1.0 - valid))
            alphas.append(alpha)

        if return_alphas:
            return outputs, alphas
        return outputs
