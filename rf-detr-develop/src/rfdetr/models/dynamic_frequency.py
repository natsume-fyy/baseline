# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Lightweight image-adaptive high/low-frequency refinement."""

import torch
from torch import nn
from torch.nn import functional as F


class DynamicFrequencyRefiner(nn.Module):
    """Separate and reweight low/high spatial frequencies per image and level.

    A masked global descriptor predicts the blend of two low-pass scales and
    independent low/high gains. The gains start at one, so an enabled module
    initially preserves pretrained backbone features exactly.

    Args:
        channels: Projected feature width.
        levels: Number of projected feature levels.
    """

    def __init__(self, channels: int, levels: int) -> None:
        super().__init__()
        if channels <= 0 or levels <= 0:
            raise ValueError("channels and levels must be positive")
        self.gates = nn.ModuleList([nn.Conv2d(channels, 3, 1) for _ in range(levels)])
        for gate in self.gates:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)

    def forward(
        self,
        features: list[torch.Tensor],
        padding_masks: list[torch.Tensor | None] | None = None,
    ) -> list[torch.Tensor]:
        """Return dynamically refined features in the original shapes and dtype.

        Args:
            features: Feature maps shaped ``(B, C, H, W)``.
            padding_masks: Boolean maps with ``True`` at padded positions.

        Returns:
            Refined feature maps, leaving padded positions unchanged.
        """
        if len(features) != len(self.gates):
            raise ValueError(f"Expected {len(self.gates)} feature levels, got {len(features)}")
        if padding_masks is None:
            padding_masks = [None] * len(features)
        if len(padding_masks) != len(features):
            raise ValueError("padding_masks must match the feature levels")

        refined = []
        for feature, padding_mask, gate in zip(features, padding_masks, self.gates):
            source = feature.float()
            if padding_mask is None:
                valid = None
                descriptor = source.mean(dim=(-2, -1), keepdim=True)
            else:
                valid = (~padding_mask).unsqueeze(1).to(source.dtype)
                descriptor = (source * valid).sum(dim=(-2, -1), keepdim=True) / valid.sum(
                    dim=(-2, -1), keepdim=True
                ).clamp_min(1)

            coefficients = gate(descriptor.to(feature.dtype)).float()
            blur3 = F.avg_pool2d(
                source if valid is None else source * valid,
                3, stride=1, padding=1, count_include_pad=False,
            )
            blur5 = F.avg_pool2d(
                source if valid is None else source * valid,
                5, stride=1, padding=2, count_include_pad=False,
            )
            if valid is not None:
                count3 = F.avg_pool2d(valid, 3, stride=1, padding=1, count_include_pad=False)
                count5 = F.avg_pool2d(valid, 5, stride=1, padding=2, count_include_pad=False)
                blur3 = blur3 / count3.clamp_min(1e-6)
                blur5 = blur5 / count5.clamp_min(1e-6)

            low_mix = torch.sigmoid(coefficients[:, 0:1])
            low = low_mix * blur3 + (1 - low_mix) * blur5
            high = source - low
            low_gain = 1 + 0.5 * torch.tanh(coefficients[:, 1:2])
            high_gain = 1 + 0.5 * torch.tanh(coefficients[:, 2:3])
            output = low_gain * low + high_gain * high
            if valid is not None:
                output = output * valid + source * (1 - valid)
            refined.append(output.to(feature.dtype))
        return refined
