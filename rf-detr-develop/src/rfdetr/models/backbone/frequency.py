# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Frequency-domain feature refinement modules."""

from __future__ import annotations

import math

import torch
from torch import nn

__all__ = ["UnderwaterFrequencyAwareFeatureReweighting"]


class UnderwaterFrequencyAwareFeatureReweighting(nn.Module):
    """Reweight centered low- and high-frequency backbone features.

    This implements the UFFR module described by Li et al. (2026). A 2D FFT is
    applied to each feature channel, the shifted spectrum is split by a centered
    circular mask, and the two bands are multiplied by shared scalar weights.
    FFT operations always run in fp32 for numerical stability under AMP.

    Args:
        channels: Expected feature channel count.
        freq_radius: Radius of the low-frequency disk on a grid normalized to
            ``[-1, 1]`` along both spatial axes.
        alpha_low: Initial multiplier for the centered low-frequency band.
        alpha_high: Initial multiplier for the complementary high-frequency band.
        learnable: Whether the two band multipliers are trainable parameters.
    """

    def __init__(
        self,
        channels: int,
        freq_radius: float = 0.6,
        alpha_low: float = 0.95,
        alpha_high: float = 1.55,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be positive, got {channels}.")
        if not 0.0 < freq_radius <= math.sqrt(2.0):
            raise ValueError(f"freq_radius must be in (0, sqrt(2)], got {freq_radius}.")
        if not math.isfinite(alpha_low) or not math.isfinite(alpha_high):
            raise ValueError("alpha_low and alpha_high must be finite.")

        self.channels = channels
        self.freq_radius = float(freq_radius)
        low = torch.tensor(float(alpha_low), dtype=torch.float32)
        high = torch.tensor(float(alpha_high), dtype=torch.float32)
        if learnable:
            self.alpha_low = nn.Parameter(low)
            self.alpha_high = nn.Parameter(high)
        else:
            self.register_buffer("alpha_low", low)
            self.register_buffer("alpha_high", high)

    def _radial_mask(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Build the centered low-frequency disk from the paper's normalized grid.

        Args:
            height: Feature height.
            width: Feature width.
            device: Device on which to construct the mask.

        Returns:
            A boolean tensor with shape ``(height, width)``.
        """
        vertical = torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32)
        horizontal = torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(vertical, horizontal, indexing="ij")
        radius = torch.sqrt(grid_y.square() + grid_x.square())
        return radius < self.freq_radius

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply shared radial two-band reweighting to a BCHW feature tensor.

        Args:
            features: Backbone feature map with shape ``(B, C, H, W)``.

        Returns:
            Reconstructed real-valued features with the input dtype restored.

        Raises:
            ValueError: If ``features`` is not BCHW or has the wrong channel count.
        """
        if features.ndim != 4:
            raise ValueError(f"UFFR expects a four-dimensional BCHW tensor, got shape {tuple(features.shape)}.")
        if features.shape[1] != self.channels:
            raise ValueError(
                f"UFFR was configured for {self.channels} channels, "
                f"but received {features.shape[1]} channels."
            )

        input_dtype = features.dtype
        with torch.autocast(device_type=features.device.type, enabled=False):
            spectrum = torch.fft.fft2(features.float(), dim=(-2, -1), norm="ortho")
            shifted = torch.fft.fftshift(spectrum, dim=(-2, -1))
            low_mask = self._radial_mask(features.shape[-2], features.shape[-1], features.device)
            weight = torch.where(low_mask, self.alpha_low.float(), self.alpha_high.float())
            weighted = shifted * weight[None, None]
            reconstructed = torch.fft.ifft2(
                torch.fft.ifftshift(weighted, dim=(-2, -1)),
                dim=(-2, -1),
                norm="ortho",
            ).real
        return reconstructed.to(dtype=input_dtype)
