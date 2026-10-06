# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Target low-pass enhancement with weaker predicted-background high-pass suppression.

F_out = F + alpha * M * L - alpha * bg_ratio * (1 - M) * H,
where L = LowPass(F), H = F - L, and M is a predicted foreground gate.
Both residuals use the original features; ground truth is used only in the gate loss.
"""

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class TargetGuidedFrequency(nn.Module):
    """Adjust projected features using shared foreground and background gates.

    Args:
        channels: Number of projected feature channels.
        reduction: Foreground predictor bottleneck reduction ratio.
        init_gain: Initial positive low-pass enhancement gain.
        max_gain: Upper bound on the learned enhancement gain.
        bg_ratio: Background suppression coefficient relative to enhancement,
            between zero and 0.5; zero disables background suppression.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        init_gain: float = 0.02,
        max_gain: float = 0.5,
        bg_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        if channels < 1 or reduction < 1:
            raise ValueError("channels and reduction must be positive")
        if not 0 < init_gain < max_gain <= 1:
            raise ValueError("Require 0 < init_gain < max_gain <= 1")
        if not 0 <= bg_ratio <= 0.5:
            raise ValueError("bg_ratio must be between zero and 0.5")
        self.max_gain = max_gain
        self.bg_ratio = bg_ratio
        self.gain_logit = nn.Parameter(torch.tensor(math.log(init_gain / (max_gain - init_gain))))
        hidden = max(1, channels // reduction)
        self.gate = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, 1, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        # Feature-domain Gaussian-like binomial filter, not an image-domain FFT cutoff.
        kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 16
        self.register_buffer("kernel", kernel[None, None].repeat(channels, 1, 1, 1))

    def forward(
        self, features: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return adjusted features and foreground logits without consuming ground truth.

        Filtering is normalized over valid neighbors; padded outputs are unchanged.
        A spatial gate can alter the output spectrum, so this is not an ideal frequency
        filter. The relative coefficient bound does not bound relative residual norms.
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
        foreground = logits.sigmoid()
        gain = (self.max_gain * self.gain_logit.sigmoid()).to(features.dtype)
        residual = gain * foreground * low - (gain * self.bg_ratio) * (1 - foreground) * high
        return features + residual * valid, logits


def foreground_gate_loss(
    levels: list[tuple[torch.Tensor, torch.Tensor | None]], targets: list[dict[str, torch.Tensor]]
) -> torch.Tensor:
    """Supervise coarse box occupancy with balanced foreground/background BCE.

    Normalized cxcywh boxes are mapped to the unpadded image extent at each level.
    Cells overlapping a box are foreground, including sub-cell objects. Exclude padding
    and average the foreground/background groups, then images and levels. Empty images
    supervise background only. This provides box-level, not segmentation supervision.
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
