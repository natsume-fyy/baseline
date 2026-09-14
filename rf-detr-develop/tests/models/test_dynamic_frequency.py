"""Tests for dynamic high/low-frequency refinement."""

import pytest
import torch

from rfdetr.models.dynamic_frequency import DynamicFrequencyRefiner


@pytest.mark.parametrize("shape", [(2, 8, 9, 11), (1, 8, 1, 1)])
def test_refiner_starts_as_identity_and_backpropagates(shape: tuple[int, ...]) -> None:
    """The new path must preserve pretrained features at initialization."""
    refiner = DynamicFrequencyRefiner(channels=8, levels=1)
    feature = torch.randn(shape, requires_grad=True)

    refined = refiner([feature])
    torch.testing.assert_close(refined[0], feature, rtol=1e-6, atol=1e-6)
    refined[0].square().mean().backward()

    assert refined[0].shape == feature.shape
    assert feature.grad is not None
    assert refiner.gates[0].weight.grad is not None


def test_refiner_preserves_padded_positions() -> None:
    """Padding must not be altered or included in the image descriptor."""
    refiner = DynamicFrequencyRefiner(channels=8, levels=1)
    feature = torch.randn(1, 8, 7, 9)
    padding = torch.zeros(1, 7, 9, dtype=torch.bool)
    padding[:, -2:, :] = True
    with torch.no_grad():
        refiner.gates[0].bias[1] = 0.5

    refined = refiner([feature], [padding])

    torch.testing.assert_close(refined[0][:, :, -2:, :], feature[:, :, -2:, :])
