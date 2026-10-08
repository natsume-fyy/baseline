# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Training-only, box-guided mid/low-frequency residual enhancement."""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class TargetMidLowFrequency(nn.Module):
    """Learn a structure residual inside coarse ground-truth foreground regions.

    The detector runs this branch only during training. Its original feature
    path is untouched. Low = G5(F), mid = G3(F) - G5(F), and auxiliary features
    are F + box_mask * project(GELU(reduce(concat(low, mid)))). Fixed binomial
    filters approximate Gaussian scales, not ideal Fourier frequency bands.

    Args:
        channels: Projected feature channel count.
        reduction: Bottleneck channel reduction ratio.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        if channels < 1 or reduction < 1:
            raise ValueError("channels and reduction must be positive")
        hidden = max(1, channels // reduction)
        for name, coefficients in (("fine_kernel", [1, 2, 1]), ("coarse_kernel", [1, 4, 6, 4, 1])):
            vector = torch.tensor(coefficients, dtype=torch.float32)
            kernel = vector[:, None] * vector[None, :]
            self.register_buffer(name, (kernel / kernel.sum())[None, None].repeat(channels, 1, 1, 1))
        self.reduce = nn.Conv2d(2 * channels, hidden, 1)
        self.activation = nn.GELU()
        self.project = nn.Conv2d(hidden, channels, 1)
        # Begin with the same representation as the main branch.
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def _smooth(self, clean: torch.Tensor, valid: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Normalize smoothing over valid neighbors, including image boundaries."""
        kernel = kernel.to(dtype=clean.dtype)
        padding = kernel.shape[-1] // 2
        numerator = F.conv2d(clean, kernel, padding=padding, groups=kernel.shape[0])
        denominator = F.conv2d(valid, kernel[:1], padding=padding).clamp_min(1e-6)
        return numerator / denominator

    @staticmethod
    def _foreground_mask(valid: torch.Tensor, targets: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        """Rasterize normalized cxcywh boxes onto unpadded extents.

        Cells with any box overlap are foreground, retaining sub-cell targets.
        Box occupancy is coarse guidance, not pixel-accurate segmentation.
        """
        batch, _, height, width = valid.shape
        if len(targets) != batch:
            raise ValueError("Target batch size must match the feature batch size")
        foreground = torch.zeros_like(valid)
        x = torch.arange(width, device=valid.device, dtype=torch.float32)
        y = torch.arange(height, device=valid.device, dtype=torch.float32)
        for index, target in enumerate(targets):
            boxes = target["boxes"].detach().to(device=valid.device, dtype=torch.float32)
            boxes = boxes[torch.isfinite(boxes).all(dim=1) & (boxes[:, 2:] > 0).all(dim=1)]
            valid_h = valid[index, 0].bool().any(dim=1).sum()
            valid_w = valid[index, 0].bool().any(dim=0).sum()
            lower = (boxes[:, :2] - boxes[:, 2:] / 2).clamp(0, 1)
            upper = (boxes[:, :2] + boxes[:, 2:] / 2).clamp(0, 1)
            nonempty = (upper > lower).all(dim=1)
            lower, upper = lower[nonempty], upper[nonempty]
            overlap_x = (x[None] < upper[:, 0, None] * valid_w) & (x[None] + 1 > lower[:, 0, None] * valid_w)
            overlap_y = (y[None] < upper[:, 1, None] * valid_h) & (y[None] + 1 > lower[:, 1, None] * valid_h)
            foreground[index, 0] = (overlap_y[:, :, None] & overlap_x[:, None, :]).any(dim=0)
        return foreground * valid

    def forward(
        self,
        features: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return auxiliary features, preserving background, padding and input storage."""
        valid = torch.ones_like(features[:, :1]) if padding_mask is None else (~padding_mask[:, None]).to(features.dtype)
        clean = features.masked_fill(valid == 0, 0)
        low = self._smooth(clean, valid, self.coarse_kernel)
        mid = self._smooth(clean, valid, self.fine_kernel) - low
        residual = self.project(self.activation(self.reduce(torch.cat((low, mid), dim=1))))
        foreground = self._foreground_mask(valid, targets)
        return features + residual.masked_fill(foreground == 0, 0)
