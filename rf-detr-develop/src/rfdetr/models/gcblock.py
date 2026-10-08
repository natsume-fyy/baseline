# Adapted from gyyang23/GCNet, mmsegmentation/mmseg/models/backbones/gcnet.py.
# https://github.com/gyyang23/GCNet
# MIT License
# Copyright (c) 2026 Guoyu Yang
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Pure-PyTorch Golden Cudgel block for equal-channel, stride-one features.

Preserves the upstream three two-convolution paths and BN identity path.
Only the shape-preserving configuration needed by Projector features is exposed.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class GCBlock(nn.Module):
    """Golden Cudgel convolution block with optional inference reparameterization.

    Args:
        channels: Input and output channel count.
        act: Apply the upstream terminal ReLU. Disabled after Projector LayerNorm.
    """

    def __init__(self, channels: int, act: bool = False) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels = channels
        self.activation = nn.ReLU() if act else nn.Identity()
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel, padding=kernel // 2, bias=False),
                nn.BatchNorm2d(channels),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
            )
            for kernel in (3, 3, 1)
        ])
        self.identity_bn = nn.BatchNorm2d(channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply either the train-time paths or the fused evaluation convolution."""
        if hasattr(self, "reparam"):
            return self.activation(self.reparam(features))
        return self.activation(self.identity_bn(features) + sum(path(features) for path in self.paths))

    @staticmethod
    def _fuse(conv: nn.Conv2d, norm: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
        """Fold evaluation-mode BatchNorm into a bias-free convolution."""
        scale = norm.weight / (norm.running_var + norm.eps).sqrt()
        return conv.weight * scale[:, None, None, None], norm.bias - norm.running_mean * scale

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        """Fuse into a single 3x3 convolution; call only on an evaluation copy."""
        if self.training:
            raise RuntimeError("Call eval() before fusing GCBlock")
        if hasattr(self, "reparam"):
            return
        norm = self.identity_bn
        scale = norm.weight / (norm.running_var + norm.eps).sqrt()
        kernel = torch.zeros(self.channels, self.channels, 3, 3, device=scale.device, dtype=scale.dtype)
        indices = torch.arange(self.channels, device=scale.device)
        kernel[indices, indices, 1, 1] = scale
        bias = norm.bias - norm.running_mean * scale
        for path in self.paths:
            first, first_bias = self._fuse(path[0], path[1])
            second, second_bias = self._fuse(path[2], path[3])
            mixing = second[:, :, 0, 0]
            composed = torch.einsum("oi,ichw->ochw", mixing, first)
            if composed.shape[-1] == 1:
                composed = F.pad(composed, (1, 1, 1, 1))
            kernel += composed
            bias += second_bias + mixing @ first_bias
        self.reparam = nn.Conv2d(self.channels, self.channels, 3, padding=1).to(kernel)
        self.reparam.weight.copy_(kernel)
        self.reparam.bias.copy_(bias)
        self.reparam.eval()
        del self.paths
        del self.identity_bn
