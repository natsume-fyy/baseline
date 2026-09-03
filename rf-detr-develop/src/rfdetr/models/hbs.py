# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Fog-adaptive, foreground-aware smoothing for projected features."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _gaussian_kernel(kernel_size: int) -> torch.Tensor:
    """Create a normalized two-dimensional Gaussian kernel."""
    sigma = max(float(kernel_size) / 6.0, 0.5)
    coordinates = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-(coordinates.square()) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    return torch.outer(kernel_1d, kernel_1d)[None, None]


def rasterize_foreground_mask(
    boxes: torch.Tensor,
    height: int,
    width: int,
    padding_mask: torch.Tensor | None,
    *,
    dtype: torch.dtype,
    device: torch.device,
    expansion: float = 0.1,
) -> torch.Tensor:
    """Rasterize normalized cxcywh boxes and slightly protect their edges."""
    mask = torch.zeros((1, height, width), dtype=dtype, device=device)
    if boxes.numel() == 0:
        return mask

    if padding_mask is None:
        valid_height, valid_width = height, width
    else:
        valid_height = int((~padding_mask).any(dim=1).sum().item())
        valid_width = int((~padding_mask).any(dim=0).sum().item())

    boxes = boxes.detach().to(device=device, dtype=torch.float32)
    half_width = boxes[:, 2] * (0.5 + expansion)
    half_height = boxes[:, 3] * (0.5 + expansion)
    xyxy = torch.stack(
        [
            boxes[:, 0] - half_width,
            boxes[:, 1] - half_height,
            boxes[:, 0] + half_width,
            boxes[:, 1] + half_height,
        ],
        dim=1,
    ).clamp(0, 1)

    for box in xyxy:
        x1 = max(0, min(valid_width, int(torch.floor(box[0] * valid_width).item())))
        y1 = max(0, min(valid_height, int(torch.floor(box[1] * valid_height).item())))
        x2 = max(0, min(valid_width, int(torch.ceil(box[2] * valid_width).item())))
        y2 = max(0, min(valid_height, int(torch.ceil(box[3] * valid_height).item())))
        if x2 > x1 and y2 > y1:
            mask[:, y1:y2, x1:x2] = 1
    return mask


class BackgroundSmoothingBlock(nn.Module):
    """True low-pass smoothing using a learnable mixture of fixed Gaussian kernels."""

    def __init__(self, channels: int, reduction: int = 4, kernel_size: int = 5) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        if reduction <= 0:
            raise ValueError(f"reduction must be positive, got {reduction}.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")

        candidate_sizes = sorted({3, 5, 7, kernel_size})
        self.channels = channels
        self.kernel_sizes = candidate_sizes
        for index, size in enumerate(candidate_sizes):
            self.register_buffer(f"kernel_{index}", _gaussian_kernel(size), persistent=True)
        self.mixture_logits = nn.Parameter(torch.zeros(len(candidate_sizes)))

    def _blur(
        self,
        features: torch.Tensor,
        valid: torch.Tensor,
        kernel: torch.Tensor,
    ) -> torch.Tensor:
        channels = features.shape[1]
        kernel = kernel.to(dtype=features.dtype)
        padding = kernel.shape[-1] // 2
        numerator = F.conv2d(
            features * valid,
            kernel.expand(channels, 1, -1, -1),
            padding=padding,
            groups=channels,
        )
        denominator = F.conv2d(valid, kernel, padding=padding)
        return numerator / denominator.clamp_min(1e-6)

    def forward(
        self,
        features: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a convex mixture of Gaussian-smoothed features."""
        if valid is None:
            valid = torch.ones_like(features[:, :1])
        blurred = [
            self._blur(features, valid, getattr(self, f"kernel_{index}"))
            for index in range(len(self.kernel_sizes))
        ]
        weights = torch.softmax(self.mixture_logits, dim=0).to(dtype=features.dtype)
        return sum(weight * level for weight, level in zip(weights, blurred))


class FogFrequencyGate(nn.Module):
    """Predict a per-image smoothing gate from haze and frequency cues."""

    def __init__(self, channels: int, reduction: int = 4, initial_alpha: float = 0.25) -> None:
        super().__init__()
        if not 0.0 < initial_alpha < 1.0:
            raise ValueError(f"initial_alpha must be in (0, 1), got {initial_alpha}.")
        hidden_channels = max(channels // reduction, 4)
        self.predictor = nn.Sequential(
            nn.Linear(2 * channels + 1, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, 1),
        )
        final_layer = self.predictor[-1]
        nn.init.normal_(final_layer.weight, std=1e-3)
        nn.init.constant_(final_layer.bias, math.log(initial_alpha / (1.0 - initial_alpha)))
        # Positive coefficients encode the intended prior while remaining learnable:
        # more high-frequency energy raises alpha; heavier haze suppresses it.
        self.frequency_gain = nn.Parameter(torch.tensor(-2.0))
        self.haze_suppression = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        features: torch.Tensor,
        low_frequency: torch.Tensor,
        valid: torch.Tensor,
        image_haze: torch.Tensor,
    ) -> torch.Tensor:
        """Return scene-level alpha shaped (B, 1, 1, 1)."""
        stats = features.float() if features.dtype in {torch.float16, torch.bfloat16} else features
        low = low_frequency.float() if low_frequency.dtype != stats.dtype else low_frequency
        stats_valid = valid.to(dtype=stats.dtype)
        spatial_count = stats_valid.sum(dim=(-2, -1)).clamp_min(1.0)
        channel_mean = (stats * stats_valid).sum(dim=(-2, -1)) / spatial_count
        centered = (stats - channel_mean[:, :, None, None]) * stats_valid
        channel_std = torch.sqrt(centered.square().sum(dim=(-2, -1)) / spatial_count + 1e-6)

        magnitude = (stats.abs() * stats_valid).sum(dim=(1, 2, 3)).clamp_min(1e-6)
        frequency_ratio = ((stats - low).abs() * stats_valid).sum(dim=(1, 2, 3)) / magnitude
        global_contrast = channel_std.mean(dim=1) / (
            channel_mean.abs().mean(dim=1) + channel_std.mean(dim=1) + 1e-6
        )
        descriptor = torch.cat(
            [
                channel_mean,
                channel_std,
                global_contrast[:, None],
            ],
            dim=1,
        )
        descriptor = descriptor.to(dtype=self.predictor[0].weight.dtype)
        gate_logit = self.predictor(descriptor).squeeze(1)
        # Frequency and haze stay outside the unconstrained MLP so their
        # effects have guaranteed positive and negative signs respectively.
        frequency_term = frequency_ratio.to(dtype=gate_logit.dtype)
        haze_term = image_haze.to(dtype=gate_logit.dtype)
        frequency_gain = F.softplus(self.frequency_gain).to(dtype=gate_logit.dtype)
        haze_suppression = F.softplus(self.haze_suppression).to(dtype=gate_logit.dtype)
        gate_logit = gate_logit + frequency_gain * frequency_term
        gate_logit = gate_logit - haze_suppression * haze_term
        return torch.sigmoid(gate_logit).to(dtype=features.dtype).view(-1, 1, 1, 1)


class SpatialProtectionGate(nn.Module):
    """Predict spatial smoothing strength and foreground probability."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 8)
        self.shared = nn.Sequential(
            nn.Conv2d(channels + 1, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.spatial_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.objectness_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        nn.init.normal_(self.spatial_head.weight, std=1e-3)
        nn.init.constant_(self.spatial_head.bias, math.log(0.9 / 0.1))
        nn.init.normal_(self.objectness_head.weight, std=1e-3)
        nn.init.constant_(self.objectness_head.bias, math.log(0.1 / 0.9))

    def forward(
        self,
        features: torch.Tensor,
        low_frequency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        high_frequency = (features - low_frequency).abs().mean(dim=1, keepdim=True)
        hidden = self.shared(torch.cat([features, high_frequency], dim=1))
        return torch.sigmoid(self.spatial_head(hidden)), self.objectness_head(hidden)


class HBS(nn.Module):
    """Apply scene-adaptive, spatial and foreground-aware Gaussian smoothing."""

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
        self.smoothers = nn.ModuleList(
            [
                BackgroundSmoothingBlock(channels, reduction=reduction, kernel_size=kernel_size)
                for kernel_size in kernel_sizes
            ]
        )
        self.scene_gates = nn.ModuleList(
            [
                FogFrequencyGate(channels, reduction=reduction, initial_alpha=initial_alpha)
                for _ in kernel_sizes
            ]
        )
        self.spatial_gates = nn.ModuleList(
            [SpatialProtectionGate(channels, reduction=reduction) for _ in kernel_sizes]
        )

    @staticmethod
    def estimate_image_haze(
        images: torch.Tensor | None,
        image_padding_mask: torch.Tensor | None,
        batch_size: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Estimate haze from normalized-image contrast and channel separation."""
        if images is None:
            return torch.zeros(batch_size, device=device)
        stats = images.float()
        if image_padding_mask is None:
            valid = torch.ones_like(stats[:, :1])
        else:
            valid = (~image_padding_mask).unsqueeze(1).to(dtype=stats.dtype)
        count = (valid.sum(dim=(-2, -1)) * stats.shape[1]).clamp_min(1.0)
        mean = (stats * valid).sum(dim=(1, 2, 3)) / count
        contrast = torch.sqrt(
            ((stats - mean[:, None, None, None]) * valid).square().sum(dim=(1, 2, 3)) / count + 1e-6
        )
        channel_spread = (
            stats.max(dim=1, keepdim=True).values - stats.min(dim=1, keepdim=True).values
        ) * valid
        channel_spread = channel_spread.sum(dim=(1, 2, 3)) / valid.sum(dim=(1, 2, 3)).clamp_min(1.0)
        scale = stats.abs().mul(valid).sum(dim=(1, 2, 3)) / count + 1e-6
        clarity = contrast / (contrast + scale) + channel_spread / (channel_spread + scale)
        return torch.exp(-clarity).clamp(0, 1)

    def forward(
        self,
        features: list[torch.Tensor],
        padding_masks: list[torch.Tensor | None] | None = None,
        *,
        images: torch.Tensor | None = None,
        image_padding_mask: torch.Tensor | None = None,
        return_alphas: bool = False,
        return_aux: bool = False,
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], Any]:
        """Smooth features using alpha_scene * alpha_spatial * (1-Pfg)."""
        if len(features) != len(self.smoothers):
            raise ValueError(f"Expected {len(self.smoothers)} feature levels, received {len(features)}.")
        if padding_masks is None:
            padding_masks = [None] * len(features)
        if len(padding_masks) != len(features):
            raise ValueError(f"Expected {len(features)} padding masks, received {len(padding_masks)}.")
        if not features:
            return ([], {}) if return_aux else []

        image_haze = self.estimate_image_haze(
            images,
            image_padding_mask,
            features[0].shape[0],
            device=features[0].device,
        )
        outputs: list[torch.Tensor] = []
        alphas: list[torch.Tensor] = []
        objectness_logits: list[torch.Tensor] = []
        smoothing_ratios: list[torch.Tensor] = []
        for feature, padding_mask, smoother, scene_gate, spatial_gate in zip(
            features, padding_masks, self.smoothers, self.scene_gates, self.spatial_gates
        ):
            valid = (
                torch.ones_like(feature[:, :1])
                if padding_mask is None
                else (~padding_mask).unsqueeze(1).to(dtype=feature.dtype)
            )
            low_frequency = smoother(feature, valid)
            scene_alpha = scene_gate(feature, low_frequency, valid, image_haze)
            spatial_alpha, objectness_logit = spatial_gate(feature, low_frequency)
            # Keep objectness supervised only by its dense loss. Otherwise the
            # detector could bypass smoothing by predicting foreground everywhere.
            foreground_probability = objectness_logit.sigmoid().detach()
            alpha = scene_alpha * spatial_alpha * (1.0 - foreground_probability) * valid
            fused = feature + alpha * (low_frequency - feature)
            outputs.append(fused * valid + feature * (1.0 - valid))
            alphas.append(alpha)
            objectness_logits.append(objectness_logit)
            residual_norm = ((low_frequency - feature) * valid).flatten(1).norm(dim=1)
            feature_norm = (feature * valid).flatten(1).norm(dim=1).clamp_min(1e-6)
            smoothing_ratios.append(residual_norm / feature_norm)

        if return_aux:
            return outputs, {
                "alphas": alphas,
                "objectness_logits": objectness_logits,
                "padding_masks": padding_masks,
                "image_haze": image_haze,
                "smoothing_ratios": smoothing_ratios,
            }
        if return_alphas:
            return outputs, alphas
        return outputs
