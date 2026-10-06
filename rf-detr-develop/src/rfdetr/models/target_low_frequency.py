# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Predicted-foreground low-frequency residual enhancement for projected features."""

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class TargetLowFrequencyEnhancement(nn.Module):
    """Add a bounded, foreground-gated low-pass residual without subtracting high-pass features.

    Args:
        channels: Projected feature channel count.
        reduction: Bottleneck ratio for the spatial foreground predictor.
        init_gain: Initial residual gain, strictly between zero and max_gain.
        max_gain: Upper bound on learned residual gain.
    """

    def __init__(self, channels: int, reduction: int = 4, init_gain: float = 0.02, max_gain: float = 0.5) -> None:
        super().__init__()
        if channels < 1 or reduction < 1:
            raise ValueError("channels and reduction must be positive")
        if not 0 < init_gain < max_gain or not math.isfinite(max_gain):
            raise ValueError("Require finite 0 < init_gain < max_gain")
        self.max_gain = max_gain
        self.gain_logit = nn.Parameter(torch.tensor(math.log(init_gain / (max_gain - init_gain))))
        self.gate = nn.Sequential(
            nn.Conv2d(channels, max(1, channels // reduction), 1),
            nn.ReLU(),
            nn.Conv2d(max(1, channels // reduction), 1, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, math.log(0.1 / 0.9))
        # Separable binomial/Gaussian-like smoothing; no image-domain frequency cutoff is assumed.
        kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 16
        self.register_buffer("kernel", kernel[None, None].repeat(channels, 1, 1, 1))

    def forward(
        self, features: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return enhanced features and foreground logits; never consume ground truth.

        Smoothing is normalized over valid neighbors so padded values cannot leak into
        valid features. Padded output values are preserved exactly. The same predicted
        gate is used for training, inference and export.
        """
        valid = (
            torch.ones_like(features[:, :1]) if padding_mask is None else (~padding_mask[:, None]).to(features.dtype)
        )
        clean = features.masked_fill(valid == 0, 0)
        kernel = self.kernel.to(dtype=features.dtype)
        numerator = F.conv2d(clean, kernel, padding=1, groups=self.kernel.shape[0])
        denominator = F.conv2d(valid, kernel[:1], padding=1).clamp_min(1e-6)
        low = numerator / denominator
        logits = self.gate(clean)
        gain = (self.max_gain * self.gain_logit.sigmoid()).to(features.dtype)
        return features + gain * logits.sigmoid() * low * valid, logits


def foreground_gate_loss(
    levels: list[tuple[torch.Tensor, torch.Tensor | None]], targets: list[dict[str, torch.Tensor]]
) -> torch.Tensor:
    """Compute balanced foreground/background BCE, equally averaged over images and levels.

    Normalized cxcywh target boxes are mapped to each image's unpadded extent.
    A feature cell is positive if its area overlaps any nondegenerate target box,
    retaining supervision for sub-cell objects. Box masks are coarse supervision,
    not pixel-accurate object masks. Empty targets supervise background only.
    """
    losses = []
    for logits, padding_mask in levels:
        batch, _, height, width = logits.shape
        if len(targets) != batch:
            raise ValueError("Target and foreground-logit batch sizes differ")
        valid = torch.ones_like(logits, dtype=torch.bool) if padding_mask is None else ~padding_mask[:, None]
        labels = torch.zeros_like(logits, dtype=torch.float32)
        for index, target in enumerate(targets):
            boxes = target["boxes"].detach().to(device=logits.device, dtype=torch.float32)
            boxes = boxes[(boxes[:, 2:] > 0).all(dim=1)]
            valid_h = valid[index, 0].any(dim=1).sum().clamp_min(1)
            valid_w = valid[index, 0].any(dim=0).sum().clamp_min(1)
            x = torch.arange(width, device=logits.device, dtype=torch.float32)
            y = torch.arange(height, device=logits.device, dtype=torch.float32)
            lower = (boxes[:, :2] - boxes[:, 2:] / 2).clamp(0, 1)
            upper = (boxes[:, :2] + boxes[:, 2:] / 2).clamp(0, 1)
            overlap_x = (x[None] < upper[:, 0, None] * valid_w) & (x[None] + 1 > lower[:, 0, None] * valid_w)
            overlap_y = (y[None] < upper[:, 1, None] * valid_h) & (y[None] + 1 > lower[:, 1, None] * valid_h)
            labels[index, 0] = (overlap_y[:, :, None] & overlap_x[:, None, :]).any(dim=0)
        element_loss = F.binary_cross_entropy_with_logits(logits.float(), labels, reduction="none")
        positive = labels * valid
        negative = (1 - labels) * valid
        pos_count = positive.sum(dim=(1, 2, 3))
        neg_count = negative.sum(dim=(1, 2, 3))
        pos_loss = (element_loss * positive).sum(dim=(1, 2, 3)) / pos_count.clamp_min(1)
        neg_loss = (element_loss * negative).sum(dim=(1, 2, 3)) / neg_count.clamp_min(1)
        active_groups = (pos_count > 0).float() + (neg_count > 0).float()
        losses.append(((pos_loss + neg_loss) / active_groups.clamp_min(1)).mean())
    return torch.stack(losses).mean()
