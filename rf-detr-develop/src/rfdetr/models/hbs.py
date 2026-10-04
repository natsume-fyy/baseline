# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Training-only HBS background smoothing modules.

The implementation adapts the HBS branch from SET's FCOS integration to RF-DETR's normalized ``cxcywh`` targets and
multi-scale projected backbone features. HBS never runs in evaluation/export mode, so it adds no inference latency.
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
    """Apply scale-adaptive smoothing only outside ground-truth boxes.

    RF-DETR targets store boxes as normalized ``(cx, cy, width, height)`` coordinates. A foreground mask is rasterized
    independently at each feature level, which is equivalent to SET's full-resolution mask followed by nearest-neighbor
    resizing while avoiding a large intermediate image-sized mask.

    Args:
        channels: Channel count shared by projected feature levels.
        kernel_sizes: One odd denoising kernel size per feature level.
        reduction: Bottleneck channel reduction factor.
    """

    def __init__(self, channels: int, kernel_sizes: list[int], reduction: int = 4) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("kernel_sizes must contain at least one feature level.")
        self.denoisers = nn.ModuleList(
            [
                BackgroundSmoothingBlock(channels=channels, reduction=reduction, kernel_size=kernel_size)
                for kernel_size in kernel_sizes
            ]
        )

    @staticmethod
    def _foreground_mask(
        boxes: torch.Tensor,
        height: int,
        width: int,
        *,
        valid_height: int | None = None,
        valid_width: int | None = None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Rasterize normalized ``cxcywh`` boxes into one feature-level mask.

        Args:
            boxes: Normalized boxes with shape ``(N, 4)``.
            height: Feature-map height.
            width: Feature-map width.
            valid_height: Unpadded feature height used to scale normalized box coordinates.
            valid_width: Unpadded feature width used to scale normalized box coordinates.
            dtype: Output mask dtype.
            device: Output mask device.

        Returns:
            Foreground mask with shape ``(1, H, W)``.
        """
        mask = torch.zeros((1, height, width), dtype=dtype, device=device)
        if boxes.numel() == 0:
            return mask

        box_height = height if valid_height is None else valid_height
        box_width = width if valid_width is None else valid_width

        boxes = boxes.detach().to(device=device, dtype=torch.float32)
        xyxy = torch.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        xyxy = xyxy.clamp(0, 1)

        for box in xyxy:
            x1 = max(0, min(box_width, int(torch.floor(box[0] * box_width).item())))
            y1 = max(0, min(box_height, int(torch.floor(box[1] * box_height).item())))
            x2 = max(0, min(box_width, int(torch.ceil(box[2] * box_width).item())))
            y2 = max(0, min(box_height, int(torch.ceil(box[3] * box_height).item())))
            if x2 > x1 and y2 > y1:
                mask[:, y1:y2, x1:x2] = 1
        return mask

    def forward(
        self,
        features: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        padding_masks: list[torch.Tensor | None] | None = None,
    ) -> list[torch.Tensor]:
        """Build HBS features while preserving every foreground location.

        Args:
            features: Projected feature maps, each shaped ``(B, C, H, W)``.
            targets: Per-image target dictionaries containing normalized ``boxes``.
            padding_masks: Optional per-level boolean masks shaped ``(B, H, W)`` where ``True`` marks padding.

        Returns:
            Smoothed feature maps in the same order and shapes as ``features``.
        """
        if len(features) != len(self.denoisers):
            raise ValueError(
                f"Expected {len(self.denoisers)} feature levels, received {len(features)}."
            )
        if not features:
            return []
        if len(targets) != features[0].shape[0]:
            raise ValueError(
                f"Target batch size {len(targets)} does not match feature batch size {features[0].shape[0]}."
            )
        if padding_masks is None:
            padding_masks = [None] * len(features)
        if len(padding_masks) != len(features):
            raise ValueError(
                f"Expected {len(features)} padding masks, received {len(padding_masks)}."
            )

        outputs: list[torch.Tensor] = []
        for feature, denoiser, padding_mask in zip(features, self.denoisers, padding_masks):
            batch_size, _, height, width = feature.shape
            foreground_mask = torch.stack(
                [
                    self._foreground_mask(
                        targets[batch_index]["boxes"],
                        height,
                        width,
                        valid_height=(
                            int((~padding_mask[batch_index]).any(dim=1).sum().item())
                            if padding_mask is not None
                            else None
                        ),
                        valid_width=(
                            int((~padding_mask[batch_index]).any(dim=0).sum().item())
                            if padding_mask is not None
                            else None
                        ),
                        dtype=feature.dtype,
                        device=feature.device,
                    )
                    for batch_index in range(batch_size)
                ],
                dim=0,
            )
            if padding_mask is None:
                valid_mask = torch.ones_like(foreground_mask)
            else:
                valid_mask = (~padding_mask).unsqueeze(1).to(dtype=feature.dtype)
                foreground_mask = foreground_mask * valid_mask
            background_mask = valid_mask - foreground_mask
            smoothed_background = denoiser(feature * background_mask)
            outputs.append(smoothed_background * background_mask + feature * (1 - background_mask))
        return outputs
