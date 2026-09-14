"""Tests for foreground-guided frequency refinement."""

import pytest
import torch

from rfdetr.models.foreground_frequency import ForegroundFrequencyRefiner


@pytest.mark.parametrize("shape", [(2, 8, 9, 11), (1, 8, 1, 1)])
def test_refiner_preserves_shape_and_backpropagates(shape: tuple[int, ...]) -> None:
    """Both normal and tiny feature maps must remain differentiable."""
    refiner = ForegroundFrequencyRefiner(channels=8, levels=1)
    feature = torch.randn(shape, requires_grad=True)

    refined, logits = refiner([feature])
    (refined[0].square().mean() + logits[0].square().mean()).backward()

    assert refined[0].shape == feature.shape
    assert logits[0].shape == (shape[0], 1, shape[2], shape[3])
    assert feature.grad is not None
    assert refiner.frequency_gates[0][-1].weight.grad is not None
    assert refiner.foreground_heads[0].weight.grad is not None


def test_refiner_preserves_padded_positions() -> None:
    """Refinement must leave padded feature positions unchanged."""
    refiner = ForegroundFrequencyRefiner(channels=8, levels=1)
    feature = torch.randn(1, 8, 7, 9)
    padding = torch.zeros(1, 7, 9, dtype=torch.bool)
    padding[:, -2:, :] = True

    refined, _ = refiner([feature], [padding])

    torch.testing.assert_close(refined[0][:, :, -2:, :], feature[:, :, -2:, :])
