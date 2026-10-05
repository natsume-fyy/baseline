# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Offline projector contracts; no model downloads or training dependencies needed."""

import importlib.util
import unittest
from pathlib import Path

import torch
from torch import nn

# Load the self-contained module without executing rfdetr's top-level imports.
_source = Path(__file__).resolve().parents[3] / "src/rfdetr/models/backbone/projector.py"
_spec = importlib.util.spec_from_file_location("projector_under_test", _source)
assert _spec is not None and _spec.loader is not None
projector = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(projector)


class TestProjectorCIB(unittest.TestCase):
    """Exercise fusion, gradients, normalization, and checkpoint compatibility."""

    def test_small_forward(self) -> None:
        """Preserve the Small model's four 384-channel input maps and P4 output."""
        model = projector.MultiScaleProjector([384] * 4, 256, [1.0], layer_norm=True).eval()
        with torch.no_grad():
            output = model([torch.randn(1, 384, 32, 32) for _ in range(4)])
        self.assertEqual([tuple(value.shape) for value in output], [(1, 256, 32, 32)])
        self.assertIsInstance(model.stages[0][0], projector.C2fCIB)

    def test_backward(self) -> None:
        """Propagate finite nonzero gradients into every source map and CIB parameter."""
        model = projector.MultiScaleProjector([16] * 4, 32, [1.0], layer_norm=True)
        inputs = [torch.randn(2, 16, 8, 8, requires_grad=True) for _ in range(4)]
        output = model(inputs)[0]
        (output * torch.randn_like(output)).mean().backward()
        for tensor in [*inputs, *model.parameters()]:
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all().item())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)

    def test_pyramid_shapes(self) -> None:
        """Preserve spatial scaling for both one-map and four-map projectors."""
        for channels in ([16], [16] * 4):
            with self.subTest(channels=channels):
                model = projector.MultiScaleProjector(channels, 32, [2.0, 1.0, 0.5], layer_norm=True).eval()
                with torch.no_grad():
                    output = model([torch.randn(1, c, 8, 8) for c in channels])
                self.assertEqual([tuple(t.shape) for t in output], [(1, 32, 16, 16), (1, 32, 8, 8), (1, 32, 4, 4)])

    def test_normalization(self) -> None:
        """Apply the selected normalization to every convolution inside the fusion block."""
        for use_ln in (False, True):
            with self.subTest(layer_norm=use_ln):
                model = projector.C2fCIB(64, 32, n=3, layer_norm=use_ln)
                expected = projector.LayerNorm if use_ln else nn.BatchNorm2d
                for layer in model.modules():
                    if isinstance(layer, projector.ConvX):
                        self.assertIsInstance(layer.bn, expected)

    def test_original_checkpoint_partial_load(self) -> None:
        """Reuse outer fusion weights without mistaking old bottlenecks for CIB weights."""
        old = projector.C2f(64, 32, n=3, layer_norm=True)
        new = projector.C2fCIB(64, 32, n=3, layer_norm=True)
        result = new.load_state_dict(old.state_dict(), strict=False)
        self.assertTrue(result.missing_keys)
        self.assertTrue(result.unexpected_keys)
        self.assertTrue(all(key.startswith("m.") for key in result.missing_keys + result.unexpected_keys))
        for key, value in old.state_dict().items():
            if not key.startswith("m."):
                torch.testing.assert_close(new.state_dict()[key], value, rtol=0, atol=0)

    def test_new_checkpoint_roundtrip(self) -> None:
        """Reload C2fCIB weights strictly with identical predictions."""
        original = projector.C2fCIB(64, 32, n=3, layer_norm=True).eval()
        restored = projector.C2fCIB(64, 32, n=3, layer_norm=True).eval()
        restored.load_state_dict(original.state_dict(), strict=True)
        x = torch.randn(1, 64, 8, 8)
        with torch.no_grad():
            torch.testing.assert_close(original(x), restored(x), rtol=0, atol=0)

    def test_residual(self) -> None:
        """Only add the input for compatible channels with shortcut enabled."""
        for c2, shortcut in ((16, True), (16, False), (32, True)):
            with self.subTest(c2=c2, shortcut=shortcut):
                block = projector.CIB(16, c2, shortcut=shortcut, layer_norm=True).eval()
                x = torch.randn(1, 16, 8, 8)
                with torch.no_grad():
                    expected = block.cv1(x)
                    if shortcut and c2 == 16:
                        expected = expected + x
                    torch.testing.assert_close(block(x), expected)


if __name__ == "__main__":
    unittest.main()
