# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for residual CBAM on projected feature-pyramid levels."""

import pytest
import torch

from rfdetr.models.backbone.attention import CBAM2D, FeaturePyramidCBAM, ResidualCBAM2D


def test_residual_cbam_is_identity_at_initialization() -> None:
    """A newly added block must preserve pretrained backbone features exactly."""
    module = ResidualCBAM2D(channels=32)
    features = torch.randn(2, 32, 12, 16)

    output = module(features)

    assert torch.equal(output, features)
    assert module.residual_scale.item() == 0


def test_feature_pyramid_cbam_only_changes_selected_levels() -> None:
    """Attention should target high-resolution levels without altering lower-resolution levels."""
    module = FeaturePyramidCBAM(channels=16, level_indexes=[0, 1])
    for block in module.attention_blocks:
        block.residual_scale.data.fill_(1)
    features = [
        torch.randn(2, 16, 16, 16),
        torch.randn(2, 16, 8, 8),
        torch.randn(2, 16, 4, 4),
    ]

    outputs = module(features)

    assert all(output.shape == feature.shape for output, feature in zip(outputs, features))
    assert not torch.equal(outputs[0], features[0])
    assert not torch.equal(outputs[1], features[1])
    assert torch.equal(outputs[2], features[2])


def test_cbam_channel_pool_ignores_padding() -> None:
    """Changing padded pixels must not affect the channel-attention statistics."""
    module = CBAM2D(channels=8)
    features = torch.randn(1, 8, 4, 4)
    padding_mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    padding_mask[:, :, -1] = True
    changed_padding = features.clone()
    changed_padding[:, :, :, -1] = 1000

    average, maximum = module._channel_pool(features, padding_mask)
    changed_average, changed_maximum = module._channel_pool(changed_padding, padding_mask)

    assert torch.equal(average, changed_average)
    assert torch.equal(maximum, changed_maximum)


def test_cbam_valid_output_is_independent_of_padded_values() -> None:
    """Padded activations must not leak into valid spatial-attention responses."""
    module = CBAM2D(channels=8)
    features = torch.randn(1, 8, 4, 4)
    padding_mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    padding_mask[:, :, -1] = True
    changed_padding = features.clone()
    changed_padding[:, :, :, -1] = 1000

    output = module(features, padding_mask)
    changed_output = module(changed_padding, padding_mask)

    valid = (~padding_mask).unsqueeze(1).expand_as(output)
    assert torch.equal(output[valid], changed_output[valid])


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((2, 16, 32), id="missing-spatial-dimension"),
        pytest.param((2, 4, 8, 8, 8), id="extra-dimension"),
    ],
)
def test_cbam_rejects_non_image_features(shape: tuple[int, ...]) -> None:
    """CBAM2D must not silently reinterpret decoder query sequences as images."""
    module = CBAM2D(channels=shape[1])

    with pytest.raises(ValueError, match="B,C,H,W"):
        module(torch.randn(shape))
