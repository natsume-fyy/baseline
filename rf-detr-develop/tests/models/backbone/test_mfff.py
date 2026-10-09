# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Regression tests for frequency enhancement after the projector."""

import pytest
import torch

from rfdetr.models.backbone.backbone import Backbone
from rfdetr.models.backbone.projector import MFFF
from rfdetr.utilities.tensors import NestedTensor


@pytest.mark.parametrize(
    "shape", [pytest.param((2, 32, 7, 9), id="rectangular"), pytest.param((1, 32, 8, 8), id="square")]
)
def test_mfff_shape_and_gradients(shape: tuple[int, ...]) -> None:
    """Preserve spatial shape and backpropagate through both split branches."""
    module = MFFF(shape[1])
    x = torch.randn(shape, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert torch.isfinite(y).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


class _FakeEncoder(torch.nn.Module):
    """Supply cheap spatial features without downloading DINOv2 weights."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__()
        self._out_feature_channels = [32, 32]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return two distinct feature maps with encoder-like channels."""
        feature = x.mean(dim=1, keepdim=True).repeat(1, 32, 1, 1)
        return [feature, feature * 2]


@pytest.mark.parametrize("dual", [pytest.param(False, id="single"), pytest.param(True, id="dual")])
def test_backbone_mfff_order_and_pretrained_keys(monkeypatch: pytest.MonkeyPatch, dual: bool) -> None:
    """Apply MFFF to projector outputs in normal and export forwards."""
    monkeypatch.setattr("rfdetr.models.backbone.backbone.DinoV2", _FakeEncoder)
    kwargs = dict(name="dinov2_small", out_channels=16, projector_scale=["P4"], dual_projector=dual)
    baseline = Backbone(**kwargs)
    enhanced = Backbone(**kwargs, mfff_enabled=True).eval()
    incompatible = enhanced.load_state_dict(baseline.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(key.startswith(("mfff.", "cross_attn_mfff.")) for key in incompatible.missing_keys)
    inputs = torch.randn(2, 3, 7, 9)
    observed = {}
    handles = [
        enhanced.projector.register_forward_hook(lambda _, args, output: observed.update(projected=output[0])),
        enhanced.mfff[0].register_forward_pre_hook(lambda _, args: observed.update(mfff_input=args[0])),
        enhanced.mfff[0].register_forward_hook(lambda _, args, output: observed.update(enhanced=output)),
    ]
    if dual:
        handles.extend([
            enhanced.cross_attn_projector.register_forward_hook(
                lambda _, args, output: observed.update(cross_projected=output[0])
            ),
            enhanced.cross_attn_mfff[0].register_forward_pre_hook(
                lambda _, args: observed.update(cross_input=args[0])
            ),
            enhanced.cross_attn_mfff[0].register_forward_hook(
                lambda _, args, output: observed.update(cross_enhanced=output)
            ),
        ])
    try:
        outputs, cross_outputs = enhanced(NestedTensor(inputs, torch.zeros(2, 7, 9, dtype=torch.bool)))
        assert observed["projected"] is observed["mfff_input"]
        assert outputs[0].tensors is observed["enhanced"]
        assert outputs[0].tensors.shape == (2, 16, 7, 9)
        if dual:
            assert observed["cross_projected"] is observed["cross_input"]
            assert cross_outputs[0].tensors is observed["cross_enhanced"]
        exported, _, exported_cross = enhanced.forward_export(inputs)
        assert observed["projected"] is observed["mfff_input"]
        assert exported[0] is observed["enhanced"]
        torch.testing.assert_close(exported[0], outputs[0].tensors)
        if dual:
            assert observed["cross_projected"] is observed["cross_input"]
            assert exported_cross[0] is observed["cross_enhanced"]
            torch.testing.assert_close(exported_cross[0], cross_outputs[0].tensors)
        else:
            assert cross_outputs is None and exported_cross is None
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("device", [pytest.param("cpu", id="cpu"), pytest.param("cuda", id="cuda")])
def test_mfff_autocast_non_power_of_two(device: str) -> None:
    """FFT must work with autocast on rectangular, non-power-of-two features."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    dtype = torch.float16 if device == "cuda" else torch.bfloat16
    module = MFFF(32).to(device)
    x = torch.randn(2, 32, 7, 9, device=device, requires_grad=True)
    with torch.autocast(device_type=device, dtype=dtype):
        output = module(x)
        loss = output.float().square().mean()
    loss.backward()
    assert output.dtype == dtype
    assert torch.isfinite(output).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_projector_multiscale_mfff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each pyramid level preserves the shape expected by the detector."""
    monkeypatch.setattr("rfdetr.models.backbone.backbone.DinoV2", _FakeEncoder)
    module = Backbone(name="dinov2_small", out_channels=16, projector_scale=["P3", "P4", "P5"], mfff_enabled=True)
    outputs, _ = module(NestedTensor(torch.randn(2, 3, 8, 10), torch.zeros(2, 8, 10, dtype=torch.bool)))
    assert [tuple(output.tensors.shape) for output in outputs] == [(2, 16, 16, 20), (2, 16, 8, 10), (2, 16, 4, 5)]


def test_mfff_config_namespace_roundtrip() -> None:
    """Persist the architecture switch and forward it to the model builder."""
    from rfdetr._namespace import _namespace_from_configs
    from rfdetr.config import RFDETRSmallConfig, TrainConfig

    config = RFDETRSmallConfig(mfff_enabled=True, hbs_enabled=False, pretrain_weights=None)
    restored = RFDETRSmallConfig.model_validate(config.model_dump())
    args = _namespace_from_configs(restored, TrainConfig(dataset_dir="unused"))
    assert args.mfff_enabled is True
    assert args.hbs_enabled is False
