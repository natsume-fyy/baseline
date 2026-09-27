# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for query-sequence CBAM used before the detection head."""

import pytest
import torch

from rfdetr.models.heads.cbam import CBAM


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((2, 100, 256), id="batch-query-channel"),
        pytest.param((6, 2, 100, 256), id="layer-batch-query-channel"),
    ],
)
def test_cbam_preserves_decoder_feature_shape(shape: tuple[int, ...]) -> None:
    """CBAM must support final and auxiliary decoder outputs without changing shape."""
    module = CBAM(channels=shape[-1])
    features = torch.randn(shape)

    output = module(features)

    assert output.shape == features.shape
    assert torch.isfinite(output).all()


def test_cbam_backpropagates_through_both_attention_stages() -> None:
    """Both channel and query attention parameters must receive gradients."""
    module = CBAM(channels=32, reduction=8, kernel_size=7)
    features = torch.randn(2, 20, 32, requires_grad=True)

    module(features).sum().backward()

    assert features.grad is not None
    assert module.channel_mlp[0].weight.grad is not None
    assert module.query_conv.weight.grad is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"channels": 0}, id="non-positive-channels"),
        pytest.param({"channels": 32, "reduction": 0}, id="non-positive-reduction"),
        pytest.param({"channels": 32, "kernel_size": 4}, id="even-kernel"),
    ],
)
def test_cbam_rejects_invalid_configuration(kwargs: dict[str, int]) -> None:
    """Invalid dimensions should fail during model construction."""
    with pytest.raises(ValueError):
        CBAM(**kwargs)
