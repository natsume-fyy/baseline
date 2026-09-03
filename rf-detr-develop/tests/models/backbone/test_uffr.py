# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for underwater frequency-aware feature reweighting."""

import pytest
import torch
from pydantic import ValidationError

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import RFDETRSmallConfig, TrainConfig
from rfdetr.models.backbone.frequency import UnderwaterFrequencyAwareFeatureReweighting


class TestUnderwaterFrequencyAwareFeatureReweighting:
    """Validate the paper-defined two-band Fourier reweighting."""

    def test_unit_weights_are_identity(self) -> None:
        """Unit low/high weights should reconstruct the input."""
        module = UnderwaterFrequencyAwareFeatureReweighting(
            channels=4,
            freq_radius=0.6,
            alpha_low=1.0,
            alpha_high=1.0,
            learnable=False,
        )
        inputs = torch.randn(2, 4, 9, 11)

        outputs = module(inputs)

        torch.testing.assert_close(outputs, inputs, rtol=1e-5, atol=1e-5)

    def test_constant_feature_uses_low_frequency_weight(self) -> None:
        """A DC-only feature should be scaled by alpha_low."""
        module = UnderwaterFrequencyAwareFeatureReweighting(
            channels=2,
            freq_radius=0.6,
            alpha_low=0.75,
            alpha_high=2.0,
            learnable=False,
        )
        inputs = torch.ones(1, 2, 8, 8)

        outputs = module(inputs)

        torch.testing.assert_close(outputs, inputs * 0.75)

    def test_learnable_weights_receive_gradients(self) -> None:
        """Both paper coefficients should participate in optimization."""
        module = UnderwaterFrequencyAwareFeatureReweighting(channels=3)
        inputs = torch.randn(2, 3, 8, 8, requires_grad=True)

        module(inputs).square().mean().backward()

        assert module.alpha_low.grad is not None
        assert module.alpha_high.grad is not None
        assert torch.isfinite(module.alpha_low.grad)
        assert torch.isfinite(module.alpha_high.grad)

    def test_fft_runs_in_fp32_and_restores_input_dtype(self) -> None:
        """Half precision input should be processed safely and returned unchanged in dtype."""
        module = UnderwaterFrequencyAwareFeatureReweighting(channels=2)
        inputs = torch.randn(1, 2, 8, 8, dtype=torch.float16)

        outputs = module(inputs)

        assert outputs.dtype == torch.float16
        assert torch.isfinite(outputs).all()

    @pytest.mark.parametrize(
        "shape, message",
        [
            pytest.param((2, 8, 8), "four-dimensional", id="rank"),
            pytest.param((1, 2, 8, 8), "3 channels", id="channels"),
        ],
    )
    def test_rejects_invalid_input_shape(self, shape: tuple[int, ...], message: str) -> None:
        """UFFR should fail clearly when its BCHW contract is violated."""
        module = UnderwaterFrequencyAwareFeatureReweighting(channels=3)

        with pytest.raises(ValueError, match=message):
            module(torch.randn(shape))


class TestUFFRConfiguration:
    """Validate public model configuration and builder propagation."""

    def test_defaults_preserve_existing_models(self) -> None:
        """UFFR must remain opt-in for checkpoint and behavior compatibility."""
        config = RFDETRSmallConfig(pretrain_weights=None)

        assert config.uffr is False
        assert config.uffr_feature_indexes == [1, 2, 3]

    def test_paper_settings_reach_builder_namespace(self) -> None:
        """Configured UFFR settings should survive the config bridge."""
        config = RFDETRSmallConfig(uffr=True, pretrain_weights=None)
        namespace = _namespace_from_configs(config, TrainConfig(dataset_dir="."))

        assert namespace.uffr is True
        assert namespace.uffr_freq_radius == pytest.approx(0.6)
        assert namespace.uffr_alpha_low == pytest.approx(0.95)
        assert namespace.uffr_alpha_high == pytest.approx(1.55)
        assert namespace.uffr_learnable is True
        assert namespace.uffr_feature_indexes == [1, 2, 3]

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"uffr_freq_radius": 0.0}, id="zero_radius"),
            pytest.param({"uffr_freq_radius": 1.5}, id="radius_outside_grid"),
            pytest.param({"uffr_feature_indexes": [1, 1]}, id="duplicate_feature_index"),
            pytest.param({"uffr_feature_indexes": [-1]}, id="negative_feature_index"),
        ],
    )
    def test_rejects_invalid_configuration(self, kwargs: dict[str, object]) -> None:
        """Invalid spectral geometry or feature selection should be rejected early."""
        with pytest.raises(ValidationError):
            RFDETRSmallConfig(pretrain_weights=None, **kwargs)
