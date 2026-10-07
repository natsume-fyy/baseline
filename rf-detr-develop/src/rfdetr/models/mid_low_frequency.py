# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Learned mid/low-frequency residual fusion of projected detection features."""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class MidLowFrequencyFusion(nn.Module):
    """Fuse a structure branch with an unchanged feature shortcut.

    Binomial 3x3/5x5 smoothers approximate two Gaussian scales. Their difference
    is a smooth band-pass, not an ideal Fourier cutoff. The spatial gate learns
    from detection loss; it is not a supervised foreground segmentation mask.
    A zero-initialized output projection starts the module as an identity.

    Args:
        channels: Projected feature channel count.
        reduction: Bottleneck reduction ratio.
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
        self.structure = nn.Sequential(nn.Conv2d(2 * channels, hidden, 1), nn.GELU())
        self.gate = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 1, 1))
        self.project = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def _smooth(self, clean: torch.Tensor, valid: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Smooth over valid neighbors without leaking batch padding into features."""
        kernel = kernel.to(dtype=clean.dtype)
        padding = kernel.shape[-1] // 2
        numerator = F.conv2d(clean, kernel, padding=padding, groups=kernel.shape[0])
        denominator = F.conv2d(valid, kernel[:1], padding=padding).clamp_min(1e-6)
        return numerator / denominator

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return fused features with the original shape and unchanged padded cells."""
        valid = torch.ones_like(features[:, :1]) if padding_mask is None else (~padding_mask[:, None]).to(features.dtype)
        clean = features.masked_fill(valid == 0, 0)
        fine = self._smooth(clean, valid, self.fine_kernel)
        low = self._smooth(clean, valid, self.coarse_kernel)
        mid = fine - low
        structure = self.structure(torch.cat((low, mid), dim=1))
        residual = self.project(structure)
        gate = self.gate(clean).sigmoid()
        return features + (gate * residual).masked_fill(valid == 0, 0)
