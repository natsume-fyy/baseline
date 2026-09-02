"""Tests for fog-adaptive multi-scale HBS."""

import pytest
import torch

from rfdetr.models.hbs import FogFrequencyGate, HBS


def test_hbs_uses_residual_alpha_fusion() -> None:
    module = HBS(channels=8, kernel_sizes=[3], reduction=4, initial_alpha=0.2)
    feature = torch.randn(2, 8, 10, 12)

    outputs, alphas = module([feature], return_alphas=True)
    hbs_feature = module.denoisers[0](feature)

    assert alphas[0].shape == (2, 1, 1, 1)
    assert torch.allclose(alphas[0], torch.full_like(alphas[0], 0.2), atol=1e-6)
    assert torch.allclose(outputs[0], feature + alphas[0] * (hbs_feature - feature))


def test_hbs_preserves_padding_and_backpropagates_to_gate() -> None:
    module = HBS(channels=8, kernel_sizes=[3], reduction=4)
    feature = torch.randn(2, 8, 8, 8, requires_grad=True)
    padding_mask = torch.zeros(2, 8, 8, dtype=torch.bool)
    padding_mask[:, 6:, :] = True

    outputs, alphas = module([feature], [padding_mask], return_alphas=True)
    outputs[0].sum().backward()

    assert torch.equal(outputs[0][:, :, 6:, :], feature.detach()[:, :, 6:, :])
    assert module.gates[0].predictor[-1].bias.grad is not None
    assert torch.all((alphas[0] >= 0) & (alphas[0] <= 1))


def test_hbs_validates_feature_level_count() -> None:
    module = HBS(channels=8, kernel_sizes=[3, 5])
    with pytest.raises(ValueError, match="Expected 2 feature levels"):
        module([torch.randn(1, 8, 8, 8)])


def test_gate_can_raise_alpha_for_high_frequency_features() -> None:
    gate = FogFrequencyGate(channels=1, reduction=1, initial_alpha=0.25)
    with torch.no_grad():
        gate.predictor[0].weight.zero_()
        gate.predictor[0].bias.zero_()
        gate.predictor[0].weight[0, -1] = 1.0
        gate.predictor[-1].weight.zero_()
        gate.predictor[-1].weight[0, 0] = 1.0

    smooth = torch.ones(1, 1, 8, 8)
    checkerboard = ((torch.arange(8)[:, None] + torch.arange(8)[None, :]) % 2).float()
    noisy = checkerboard[None, None]

    assert gate(noisy).item() > gate(smooth).item()
