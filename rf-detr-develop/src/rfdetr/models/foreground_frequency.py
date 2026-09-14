# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Foreground-guided, image-adaptive frequency refinement for projected features."""

import torch
from torch import nn
from torch.nn import functional as F


class ForegroundFrequencyRefiner(nn.Module):
    """Separate spatial frequencies and retain them according to foreground probability.

    The low-pass scale and four retention rates are predicted per image and feature
    level. A spatial foreground head mixes the foreground and background rates.
    Foreground high-frequency retention is constrained to be no lower than its
    background counterpart. The residual gain starts small for stable fine-tuning.

    Args:
        channels: Projected feature width.
        levels: Number of projected feature levels.
        reduction: Channel reduction for the per-image gating network.
    """

    def __init__(self, channels: int, levels: int, reduction: int = 4) -> None:
        super().__init__()
        if channels <= 0 or levels <= 0 or reduction <= 0:
            raise ValueError("channels, levels, and reduction must be positive")
        hidden = max(1, channels // reduction)
        self.frequency_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, hidden, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden, 5, 1),
                )
                for _ in range(levels)
            ]
        )
        self.foreground_heads = nn.ModuleList(
            [nn.Conv2d(channels, 1, 3, padding=1) for _ in range(levels)]
        )
        self.residual_gain = nn.Parameter(torch.full((levels,), 0.1))

    def forward(
        self,
        features: list[torch.Tensor],
        padding_masks: list[torch.Tensor | None] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Refine features and return foreground logits for optional supervision.

        Args:
            features: Feature maps with shape ``(B, C, H, W)``.
            padding_masks: Boolean maps with ``True`` at padded positions.

        Returns:
            Refined features and per-level foreground logits.
        """
        if len(features) != len(self.frequency_gates):
            raise ValueError(f"Expected {len(self.frequency_gates)} feature levels, got {len(features)}")
        if padding_masks is None:
            padding_masks = [None] * len(features)
        if len(padding_masks) != len(features):
            raise ValueError("padding_masks must match the feature levels")

        refined, foreground_logits = [], []
        for level, (feature, padding_mask) in enumerate(zip(features, padding_masks)):
            # Pooling is computed in fp32 so AMP does not lose small high-frequency residuals.
            source = feature.float()
            if padding_mask is not None:
                valid = (~padding_mask).unsqueeze(1).to(source.dtype)
                pooled = (source * valid).sum(dim=(-2, -1), keepdim=True) / valid.sum(
                    dim=(-2, -1), keepdim=True
                ).clamp_min(1)
            else:
                valid = None
                pooled = source.mean(dim=(-2, -1), keepdim=True)
            gates = self.frequency_gates[level](pooled.to(feature.dtype)).float()
            if valid is None:
                blur3 = F.avg_pool2d(source, 3, stride=1, padding=1, count_include_pad=False)
                blur5 = F.avg_pool2d(source, 5, stride=1, padding=2, count_include_pad=False)
            else:
                blur3 = F.avg_pool2d(source * valid, 3, stride=1, padding=1, count_include_pad=False)
                blur5 = F.avg_pool2d(source * valid, 5, stride=1, padding=2, count_include_pad=False)
                count3 = F.avg_pool2d(valid, 3, stride=1, padding=1, count_include_pad=False)
                count5 = F.avg_pool2d(valid, 5, stride=1, padding=2, count_include_pad=False)
                blur3 = blur3 / count3.clamp_min(1e-6)
                blur5 = blur5 / count5.clamp_min(1e-6)
            low_mix = torch.sigmoid(gates[:, 0:1])
            low = low_mix * blur3 + (1 - low_mix) * blur5
            high = source - low

            fg_logits = self.foreground_heads[level](feature).float()
            foreground = torch.sigmoid(fg_logits)
            bg_high = torch.sigmoid(gates[:, 1:2])
            fg_high = bg_high + (1 - bg_high) * torch.sigmoid(gates[:, 2:3])
            bg_low = 0.5 + 0.5 * torch.sigmoid(gates[:, 3:4])
            fg_low = 0.5 + 0.5 * torch.sigmoid(gates[:, 4:5])
            high_keep = foreground * fg_high + (1 - foreground) * bg_high
            low_keep = foreground * fg_low + (1 - foreground) * bg_low
            result = source + self.residual_gain[level] * (low_keep * low + high_keep * high - source)
            if valid is not None:
                result = result * valid + source * (1 - valid)
            refined.append(result.to(feature.dtype))
            foreground_logits.append(fg_logits)
        return refined, foreground_logits
