# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""HBS foreground preservation and training-only integration tests."""

from types import MethodType

import torch
from torch import nn

from rfdetr.models.hbs import HBS
from rfdetr.models.lwdetr import LWDETR
from rfdetr.utilities.tensors import NestedTensor


def test_hbs_preserves_foreground_and_padding() -> None:
    """Only valid background changes, and smoothing parameters receive gradients."""
    hbs = HBS(channels=4, kernel_sizes=[3])
    for parameter in hbs.parameters():
        nn.init.constant_(parameter, 0.1)
    feature = torch.ones(1, 4, 4, 6, requires_grad=True)
    padding = torch.zeros(1, 4, 6, dtype=torch.bool)
    padding[:, :, 4:] = True
    targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]])}]

    output = hbs([feature], targets, [padding])[0]

    torch.testing.assert_close(output[:, :, 1:3, 1:3], feature[:, :, 1:3, 1:3])
    torch.testing.assert_close(output[:, :, :, 4:], feature[:, :, :, 4:])
    assert (output[:, :, 0, :4] > feature[:, :, 0, :4]).all()
    output.sum().backward()
    assert feature.grad is not None
    assert all(parameter.grad is not None for parameter in hbs.parameters())


class _FeatureBackbone(nn.Module):
    """Pass through already projected features for a lightweight branch test."""

    def forward(self, samples: NestedTensor) -> tuple:
        return [samples], [torch.zeros_like(samples.tensors)], None


def test_hbs_branch_runs_only_during_training_with_targets() -> None:
    """Training reuses the head twice; evaluation and target-free forward bypass HBS."""
    model = LWDETR.__new__(LWDETR)
    nn.Module.__init__(model)
    model.backbone = _FeatureBackbone()
    model.hbs = HBS(channels=4, kernel_sizes=[3])
    calls = []

    def head(self, samples, features, poss, cross_attn_features):
        calls.append(features[0].tensors)
        return {"pred_logits": features[0].tensors.mean(dim=(-2, -1))}

    model._forward_from_backbone_features = MethodType(head, model)
    samples = NestedTensor(torch.ones(1, 4, 4, 4), torch.zeros(1, 4, 4, dtype=torch.bool))
    targets = [{"boxes": torch.empty(0, 4)}]
    model.train()
    assert "hbs_outputs" in model(samples, targets)
    assert len(calls) == 2
    model.eval()
    assert "hbs_outputs" not in model(samples, targets)
    assert len(calls) == 3
    model.train()
    assert "hbs_outputs" not in model(samples)
    assert len(calls) == 4
