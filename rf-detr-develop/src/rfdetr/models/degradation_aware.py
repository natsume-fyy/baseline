# ------------------------------------------------------------------------
# Adapted from ILAWR (https://github.com/xin1u/ILAWR).
# ------------------------------------------------------------------------
"""Degradation-aware feature refinement and residual fusion."""

from collections.abc import Sequence

import torch
from torch import nn


class ResidualDenseDilatedBlock(nn.Module):
    """Dense dilated convolutions used by the degradation-aware branch."""

    def __init__(self, channels: int, dilations: Sequence[int] = (1, 3, 5)) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels * (index + 1), channels, 3, padding=dilation, dilation=dilation),
                nn.GELU(),
            )
            for index, dilation in enumerate(dilations)
        )
        self.fusion = nn.Conv2d(channels * (len(dilations) + 1), channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for block in self.blocks:
            features.append(block(torch.cat(features, dim=1)))
        return x + self.fusion(torch.cat(features, dim=1))


class DegradationAwareModule(nn.Module):
    """Extract degradation-aware features from the Fourier amplitude spectrum.

    This is the ILAWR degradation-aware module adapted to RF-DETR feature
    channels. FFT calculations are performed in float32 because CUDA FFT does
    not support arbitrary spatial sizes for float16 inputs.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.avg_head = nn.Conv2d(1, 1, 3, padding=1)
        self.fft_head = nn.Conv2d(channels, channels, 3, padding=1)
        self.rdbs = ResidualDenseDilatedBlock(channels)
        self.out_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        input_dtype = feature.dtype
        fft_feature = feature.float()
        channel_average = fft_feature.mean(dim=1, keepdim=True)
        average_amplitude = torch.fft.fft2(channel_average).abs()
        average_amplitude = self.act(self.avg_head(average_amplitude))
        feature_amplitude = torch.fft.fft2(fft_feature).abs()
        feature_amplitude = self.act(self.fft_head(feature_amplitude))
        specific_amplitude = feature_amplitude - average_amplitude
        weights = self.act(self.rdbs(specific_amplitude))
        return self.out_conv(weights * fft_feature).to(input_dtype)

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False

    def unfreeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = True


class DegradationAwareFusion(nn.Module):
    """Fuse an untouched identity path with a degradation-aware parallel path."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.degradation_branch = DegradationAwareModule(channels)
        # Start from the exact pretrained RF-DETR behavior. A scalar gate also
        # lets training determine how strongly the new branch should contribute.
        self.fusion_scale = nn.Parameter(torch.zeros(()))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return feature + self.fusion_scale * self.degradation_branch(feature)
