# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Predicted-background high-pass residual suppression after feature projection.

F_out = F - beta * (1 - sigmoid(foreground_logits)) * (F - LowPass(F)).
The forward pass never uses ground-truth boxes. Box masks supervise the gate only.
"""

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class BackgroundHighFrequencySuppression(nn.Module):
    """Conservatively attenuate the high-pass residual in predicted background.

    Args:
        channels: Projected feature channel count.
        reduction: Foreground gate bottleneck reduction ratio.
        init_strength: Initial suppression strength, between zero and max_strength.
        max_strength: Maximum learned suppression strength, at most one.
    """

    def __init__(
        self, channels: int, reduction: int = 4, init_strength: float = 0.02, max_strength: float = 0.5
    ) -> None:
        super().__init__()
        if channels < 1 or reduction < 1:
            raise ValueError("channels and reduction must be positive")
        if not 0 < init_strength < max_strength <= 1:
            raise ValueError("Require 0 < init_strength < max_strength <= 1")
        self.max_strength = max_strength
        self.strength_logit = nn.Parameter(torch.tensor(math.log(init_strength / (max_strength - init_strength))))
        hidden = max(1, channels // reduction)
        self.gate = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, 1, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        # Gaussian-like binomial low-pass in feature coordinates, not a sharp FFT cutoff.
        kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 16
        self.register_buffer("kernel", kernel[None, None].repeat(channels, 1, 1, 1))

    def forward(
        self, features: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return suppressed features and foreground logits, preserving padded locations.

        Normalize smoothing over valid neighbors to prevent padding leakage. A spatially
        varying gate can change the output spectrum; this is not an ideal band-stop filter.
        Predicted foreground may be imperfect, so true objects are not guaranteed unchanged.
        """
        if padding_mask is None:
            valid = torch.ones_like(features[:, :1])
        else:
            valid = (~padding_mask[:, None]).to(features.dtype)
        clean = features.masked_fill(valid == 0, 0)
        kernel = self.kernel.to(features.dtype)
        smooth = F.conv2d(clean, kernel, padding=1, groups=self.kernel.shape[0])
        weight = F.conv2d(valid, kernel[:1], padding=1).clamp_min(1e-6)
        low = smooth / weight
        high = clean - low
        logits = self.gate(clean)
        strength = (self.max_strength * self.strength_logit.sigmoid()).to(features.dtype)
        return features - strength * (1 - logits.sigmoid()) * high * valid, logits


def foreground_gate_loss(
    levels: list[tuple[torch.Tensor, torch.Tensor | None]], targets: list[dict[str, torch.Tensor]]
) -> torch.Tensor:
    """Supervise the foreground gate with balanced BCE on coarse box occupancy.

    Targets use normalized cxcywh boxes relative to the unpadded image. Any cell overlapping
    a box is foreground, retaining sub-cell objects. Padding is excluded; empty images
    supervise background only. Average foreground/background separately, then images/levels.
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
