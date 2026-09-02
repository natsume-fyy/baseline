# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Gated residual background smoothing for RF-DETR multi-scale features.

HBS operates directly on the Backbone/Projector outputs used by the main detection path. Each feature level has an
independent smoother, spatial detail gate, and learnable residual scale.
"""

from __future__ import annotations

import torch
from torch import nn


class BackgroundSmoothingBlock(nn.Module):
    """Residual convolutional denoiser used to smooth background features.

    Args:
        channels: Number of input and output feature channels.
        reduction: Bottleneck channel reduction factor.
        kernel_size: Odd spatial convolution kernel size.
    """

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
            nn.ReLU(),
            nn.Conv2d(bottleneck_channels, channels, kernel_size, stride=1, padding=padding, bias=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return the residual-smoothed feature map.

        Args:
            features: Feature tensor with shape ``(B, C, H, W)``.

        Returns:
            Tensor with the same shape as ``features``.
        """
        return features + self.conv_block(features)


class HBS(nn.Module):
    """Enhance multi-scale features with gated residual background smoothing.

    For feature level ``i``, the module computes
    ``F_i' = F_i + alpha_i * (1 - A_i) * (S_i(F_i) - F_i)``. ``A_i`` is a learned spatial gate in ``[0, 1]``:
    values near one preserve original detail, while values near zero select the smoothed representation. The learned
    scalar ``alpha_i`` controls the overall strength of the residual update for that level.

    Args:
        channels: Channel count shared by projected feature levels.
        kernel_sizes: One odd denoising kernel size per feature level.
        reduction: Bottleneck channel reduction factor.
        alpha_init: Initial value of every learnable residual scale.
    """

    def __init__(
        self,
        channels: int,
        kernel_sizes: list[int],
        reduction: int = 4,
        alpha_init: float = 0.1,
    ) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("kernel_sizes must contain at least one feature level.")
        if not 0 <= alpha_init <= 1:
            raise ValueError(f"alpha_init must be between 0 and 1, got {alpha_init}.")

        self.denoisers = nn.ModuleList(
            [
                BackgroundSmoothingBlock(channels=channels, reduction=reduction, kernel_size=kernel_size)
                for kernel_size in kernel_sizes
            ]
        )
        self.detail_gates = nn.ModuleList([nn.Conv2d(channels, 1, kernel_size=1) for _ in kernel_sizes])
        self.alphas = nn.Parameter(torch.full((len(kernel_sizes),), float(alpha_init)))

        for gate in self.detail_gates:
            nn.init.zeros_(gate.weight)
            if gate.bias is not None:
                nn.init.constant_(gate.bias, 2.0)

    def forward(
        self,
        features: list[torch.Tensor],
        padding_masks: list[torch.Tensor | None] | None = None,
    ) -> list[torch.Tensor]:
        """Return the gated residual enhancement of every feature level.

        Args:
            features: Projected feature maps, each shaped ``(B, C, H, W)``.
            padding_masks: Optional per-level boolean masks shaped ``(B, H, W)`` where ``True`` marks padding.

        Returns:
            Enhanced feature maps in the same order and shapes as ``features``.
        """
        if len(features) != len(self.denoisers):
            raise ValueError(f"Expected {len(self.denoisers)} feature levels, received {len(features)}.")
        if padding_masks is None:
            padding_masks = [None] * len(features)
        if len(padding_masks) != len(features):
            raise ValueError(f"Expected {len(features)} padding masks, received {len(padding_masks)}.")

        outputs: list[torch.Tensor] = []
        for level, (feature, denoiser, detail_gate, padding_mask) in enumerate(
            zip(features, self.denoisers, self.detail_gates, padding_masks)
        ):
            smoothed = denoiser(feature)
            preserve_gate = torch.sigmoid(detail_gate(feature))
            smoothing_gate = 1 - preserve_gate
            if padding_mask is not None:
                smoothing_gate = smoothing_gate * (~padding_mask).unsqueeze(1).to(dtype=feature.dtype)
            outputs.append(feature + self.alphas[level] * smoothing_gate * (smoothed - feature))
        return outputs
