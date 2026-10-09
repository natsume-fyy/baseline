# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from ViTDet (https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""Projector."""

from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class LayerNorm(nn.Module):
    """A LayerNorm variant, popularized by Transformers, that performs point-wise mean and variance normalization over
    the channel dimension for inputs that have shape (batch_size, channels, height, width).

    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        """
        LayerNorm forward
        TODO: this is a hack to avoid overflow when using fp16
        """
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


def get_norm(norm: Optional[Union[str, Callable[[int], nn.Module]]], out_channels: int) -> Optional[nn.Module]:
    """
    Args:
        norm: Either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns the normalization layer as a nn.Module.

    Returns:
        The normalization layer.
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "LN": lambda channels: LayerNorm(channels),
        }[norm]
    return norm(out_channels)


def get_activation(name, inplace=False):
    """Get activation."""
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name in ["LeakyReLU", "leakyrelu", "lrelu"]:
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name is None:
        module = nn.Identity()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class ConvX(nn.Module):
    """Conv-bn module."""

    def __init__(
        self,
        in_planes,
        out_planes,
        kernel=3,
        stride=1,
        groups=1,
        dilation=1,
        act="relu",
        layer_norm=False,
        rms_norm=False,
    ):
        super(ConvX, self).__init__()
        if not isinstance(kernel, tuple):
            kernel = (kernel, kernel)
        padding = (kernel[0] // 2, kernel[1] // 2)
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        if rms_norm:
            self.bn = nn.RMSNorm(out_planes)
        else:
            self.bn = get_norm("LN", out_planes) if layer_norm else nn.BatchNorm2d(out_planes)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        """forward."""
        out = self.act(self.bn(self.conv(x.contiguous())))
        return out


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, act="silu", layer_norm=False, rms_norm=False):
        """ch_in, ch_out, shortcut, groups, kernels, expand."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvX(c1, c_, k[0], 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.cv2 = ConvX(c_, c2, k[1], 1, groups=g, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, act="silu", layer_norm=False, rms_norm=False):
        """ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = ConvX(c1, 2 * self.c, 1, 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.cv2 = ConvX(
            (2 + n) * self.c, c2, 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm
        )  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
            for _ in range(n)
        )

    def forward(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class FFM(nn.Module):
    """Frequency modulation adapted from UAV-DETR's uav_modules/block.py.

    Source: https://github.com/ValiantDiligent/UAV-DETR/blob/main/ultralytics/nn/uav_modules/block.py
    The unused upstream convolution is omitted. FFT runs in FP32 under AMP.

    Args:
        dim: Number of input and output channels.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv1 = nn.Conv2d(dim, dim, 1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Modulate a spectrum and combine its inverse with the input."""
        x1, x2 = self.dwconv1(x), self.dwconv2(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.fft2(x2.float(), norm="backward")
            out = torch.fft.ifft2(x1.float() * spectrum, norm="backward").abs()
        return (out * self.alpha + x.float() * self.beta).to(x.dtype)


class ImprovedFFTKernel(nn.Module):
    """UAV-DETR spatial/frequency kernel with shape-preserving convolutions.

    Args:
        dim: Number of input and output channels, at least four.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.in_conv = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU())
        self.out_conv = nn.Conv2d(dim, dim, 1)
        self.dw_33 = nn.Conv2d(dim, dim, 31, padding=15, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, 1, groups=dim)
        self.act = nn.SiLU()
        self.conv1x1 = nn.Conv2d(dim, dim, 1)
        self.conv3x3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv5x5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.fac_conv = nn.Conv2d(dim, dim, 1)
        self.fac_pool = nn.AdaptiveAvgPool2d(1)
        self.ffm = FFM(dim)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1), nn.ReLU(), nn.Conv2d(dim // 4, dim, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Combine frequency modulation, channel attention and spatial kernels."""
        out = self.in_conv(x)
        x_att = self.fac_conv(self.fac_pool(out))
        with torch.autocast(device_type=x.device.type, enabled=False):
            spectrum = torch.fft.fft2(out.float(), norm="backward")
            x_fca = torch.fft.ifft2(x_att.float() * spectrum, norm="backward").abs()
        x_fca = x_fca.to(out.dtype)
        x_sca = self.conv1x1(x_fca) + self.conv3x3(x_fca) + self.conv5x5(x_fca)
        x_sca = self.ffm(x_sca * self.channel_attention(x_att))
        return self.out_conv(self.act(x + self.dw_33(out) + self.dw_11(out) + x_sca))


class MFFF(nn.Module):
    """UAV-DETR partial-channel frequency enhancement before projector fusion.

    Args:
        dim: Number of concatenated feature channels.
        e: Fraction of channels processed by the frequency kernel.
    """

    def __init__(self, dim: int, e: float = 0.25) -> None:
        super().__init__()
        self.frequency_channels = int(dim * e)
        if not 0 < e < 1 or not 4 <= self.frequency_channels < dim:
            raise ValueError("MFFF requires at least four frequency channels and a nonempty bypass branch.")
        self.cv1 = ConvX(dim, dim, kernel=1, act="silu")
        self.cv2 = ConvX(dim, dim, kernel=1, act="silu")
        self.m = ImprovedFFTKernel(self.frequency_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhance a channel subset while preserving the feature map shape."""
        frequency, bypass = self.cv1(x).split((self.frequency_channels, x.shape[1] - self.frequency_channels), dim=1)
        return self.cv2(torch.cat((self.m(frequency), bypass), dim=1))


class MultiScaleProjector(nn.Module):
    """This module implements MultiScaleProjector in :paper:`lwdetr`.

    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        scale_factors: Sequence[float],
        num_blocks: int = 3,
        layer_norm: bool = False,
        rms_norm: bool = False,
        survival_prob: float = 1.0,
        force_drop_last_n_features: int = 0,
        mfff_enabled: bool = False,
    ) -> None:
        """
        Args:
            in_channels: Channels in each input feature map level.
            out_channels: Number of channels in the output feature maps.
            scale_factors: List of scaling factors to upsample or downsample
                the input features for creating pyramid features.
            mfff_enabled: Enhance concatenated features before C2f at each scale.
        """
        super(MultiScaleProjector, self).__init__()

        self.scale_factors = scale_factors
        self.survival_prob = survival_prob
        self.force_drop_last_n_features = force_drop_last_n_features

        stages_sampling = []
        stages = []
        frequency_modules = []
        # use_bias = norm == ""
        self.use_extra_pool = False
        for scale in scale_factors:
            stages_sampling.append([])
            for in_dim in in_channels:
                layers = []

                # if in_dim > 512:
                #     layers.append(ConvX(in_dim, in_dim // 2, kernel=1))
                #     in_dim = in_dim // 2

                if scale == 4.0:
                    layers.extend(
                        [
                            nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
                            get_norm("LN", in_dim // 2),
                            nn.GELU(),
                            nn.ConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2),
                        ]
                    )
                    # in_dim // 4
                elif scale == 2.0:
                    # a hack to reduce the FLOPs and Params when the dimension of output feature is too large
                    # if in_dim > 512:
                    #     layers = [
                    #         ConvX(in_dim, in_dim // 2, kernel=1),
                    #         nn.ConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2),
                    #     ]
                    #     out_dim = in_dim // 4
                    # else:
                    layers.extend(
                        [
                            nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
                        ]
                    )
                    # in_dim // 2
                elif scale == 1.0:
                    pass
                elif scale == 0.5:
                    layers.extend(
                        [
                            ConvX(in_dim, in_dim, 3, 2, layer_norm=layer_norm),
                        ]
                    )
                elif scale == 0.25:
                    self.use_extra_pool = True
                    continue
                else:
                    raise NotImplementedError("Unsupported scale_factor:{}".format(scale))
                layers = nn.Sequential(*layers)
                stages_sampling[-1].append(layers)
            stages_sampling[-1] = nn.ModuleList(stages_sampling[-1])

            in_dim = int(sum(in_channel // max(1, scale) for in_channel in in_channels))
            frequency_modules.append(MFFF(in_dim) if mfff_enabled else nn.Identity())
            layers = [
                C2f(in_dim, out_channels, num_blocks, layer_norm=layer_norm),
                get_norm("LN", out_channels),
            ]
            layers = nn.Sequential(*layers)
            stages.append(layers)

        self.stages_sampling = nn.ModuleList(stages_sampling)
        self.stages = nn.ModuleList(stages)
        # Keep existing stages.* keys unchanged when loading pretrained checkpoints.
        self.mfff = nn.ModuleList(frequency_modules)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        num_features = len(x)
        if self.survival_prob < 1.0 and self.training:
            final_drop_prob = 1 - self.survival_prob
            drop_p = np.random.uniform()
            for i in range(1, num_features):
                critical_drop_prob = i * (final_drop_prob / (num_features - 1))
                if drop_p < critical_drop_prob:
                    x[i][:] = 0
        elif self.force_drop_last_n_features > 0:
            for i in range(self.force_drop_last_n_features):
                # don't do it inplace to ensure the compiler can optimize out the backbone layers
                x[-(i + 1)] = torch.zeros_like(x[-(i + 1)])

        results = []
        # x list of len(out_features_indexes)
        for i, stage in enumerate(self.stages):
            feat_fuse = []
            for j, stage_sampling in enumerate(self.stages_sampling[i]):
                feat_fuse.append(stage_sampling(x[j]))
            if len(feat_fuse) > 1:
                feat_fuse = torch.cat(feat_fuse, dim=1)
            else:
                feat_fuse = feat_fuse[0]
            results.append(stage(self.mfff[i](feat_fuse)))
        if self.use_extra_pool:
            results.append(F.max_pool2d(results[-1], kernel_size=1, stride=2, padding=0))
        return results


class SimpleProjector(nn.Module):
    def __init__(self, in_dim, out_dim, factor_kernel=False):
        super(SimpleProjector, self).__init__()
        if not factor_kernel:
            self.convx1 = ConvX(in_dim, in_dim * 2, layer_norm=True, act="silu")
            self.convx2 = ConvX(in_dim * 2, out_dim, layer_norm=True, act="silu")
        else:
            self.convx1 = ConvX(in_dim, out_dim, kernel=(3, 1), layer_norm=True, act="silu")
            self.convx2 = ConvX(out_dim, out_dim, kernel=(1, 3), layer_norm=True, act="silu")
        self.ln = get_norm("LN", out_dim)

    def forward(self, x):
        """forward."""
        out = self.ln(self.convx2(self.convx1(x[0])))
        return [out]
