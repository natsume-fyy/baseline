# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Regression tests for frequency enhancement in the projector."""

import pytest
import torch

from rfdetr.models.backbone.projector import MFFF, MultiScaleProjector


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


def test_projector_mfff_order_and_pretrained_keys() -> None:
    """Enhance concatenated features before C2f without renaming pretrained layers."""
    kwargs = dict(in_channels=[16, 16], out_channels=16, scale_factors=[1.0], num_blocks=1)
    baseline = MultiScaleProjector(**kwargs)
    enhanced = MultiScaleProjector(**kwargs, mfff_enabled=True)
    incompatible = enhanced.load_state_dict(baseline.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(key.startswith("mfff.") for key in incompatible.missing_keys)
    inputs = [torch.randn(2, 16, 7, 9) for _ in range(2)]
    observed = {}
    handles = [
        enhanced.mfff[0].register_forward_pre_hook(lambda _, args: observed.update(concat=args[0])),
        enhanced.mfff[0].register_forward_hook(lambda _, args, output: observed.update(enhanced=output)),
        enhanced.stages[0][0].register_forward_pre_hook(lambda _, args: observed.update(c2f=args[0])),
    ]
    try:
        outputs = enhanced(inputs)
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(observed["concat"], torch.cat(inputs, dim=1))
    assert observed["enhanced"] is observed["c2f"]
    assert outputs[0].shape == (2, 16, 7, 9)


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


def test_projector_multiscale_mfff() -> None:
    """Each pyramid level preserves the shape expected by the detector."""
    module = MultiScaleProjector([32, 32], 16, [2.0, 1.0, 0.5], num_blocks=1, mfff_enabled=True)
    outputs = module([torch.randn(2, 32, 8, 10) for _ in range(2)])
    assert [tuple(output.shape) for output in outputs] == [(2, 16, 16, 20), (2, 16, 8, 10), (2, 16, 4, 5)]


def test_mfff_config_namespace_roundtrip() -> None:
    """Persist the architecture switch and forward it to the model builder."""
    from rfdetr._namespace import _namespace_from_configs
    from rfdetr.config import RFDETRSmallConfig, TrainConfig

    config = RFDETRSmallConfig(mfff_enabled=True, hbs_enabled=False, pretrain_weights=None)
    restored = RFDETRSmallConfig.model_validate(config.model_dump())
    args = _namespace_from_configs(restored, TrainConfig(dataset_dir="unused"))
    assert args.mfff_enabled is True
    assert args.hbs_enabled is False
