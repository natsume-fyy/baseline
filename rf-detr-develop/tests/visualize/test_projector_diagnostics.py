# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Standalone CPU checks: python -m unittest discover -s tests/visualize -p test_projector_diagnostics.py."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from visualize_projector import activation_map, capture_features, paired_metrics, save_comparison


class ProjectorDiagnosticsTest(unittest.TestCase):
    """Check shared scales, signed energy, aligned metrics and real PyTorch hooks."""

    def test_signed_energy(self) -> None:
        """Opposite signed channels must not cancel."""
        np.testing.assert_allclose(activation_map(np.array([[[-2.0]], [[2.0]]])), [[2.0]])

    def test_aligned_metrics(self) -> None:
        """Scaling changes magnitude while preserving cosine similarity."""
        feature = np.ones((3, 4, 4), dtype=np.float32)
        result = paired_metrics(feature, feature * 2)
        self.assertAlmostEqual(result["cosine_mean"], 1.0)
        self.assertAlmostEqual(result["relative_l2"], 1.0)
        self.assertIsNone(paired_metrics(feature * 0, feature)["cosine_mean"])

    def test_save_shared_scale(self) -> None:
        """Both weather rows must use the same limits and persist native maps."""
        feature = np.ones((3, 4, 4), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            rows = save_comparison([Image.new("RGB", (16, 12))] * 2,
                                   [{"pre_0": feature}, {"pre_0": feature * 2}], output, True, True)
            self.assertEqual(rows[0]["color_vmax"], rows[1]["color_vmax"])
            with np.load(output / "activation_maps.npz") as data:
                np.testing.assert_allclose(data["difference_pre_0"], 1.0)
            with Image.open(output / "comparison.png") as image:
                image.verify()
            self.assertTrue((output / "clear_features.npz").is_file())

    @unittest.skipUnless(hasattr(torch, "nn"), "Working PyTorch installation unavailable")
    def test_capture_and_cleanup(self) -> None:
        """Hooks capture CHW values and are removed even if prediction fails."""
        projector = torch.nn.Identity()
        backbone = SimpleNamespace(projector=projector)
        feature = torch.ones(1, 3, 4, 4)

        def predict(image: Image.Image, **kwargs: object) -> None:
            """Exercise the hooked module without downloading a model."""
            projector([feature])

        model = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(backbone=[backbone])), predict=predict)
        captured = capture_features(model, Image.new("RGB", (16, 16)))
        feature.zero_()
        np.testing.assert_allclose(captured["pre_0"], 1.0)
        self.assertEqual(captured["post_0"].shape, (3, 4, 4))
        self.assertFalse(projector._forward_hooks)

        def fail(image: Image.Image, **kwargs: object) -> None:
            """Simulate failed inference."""
            raise RuntimeError("failure")

        model.predict = fail
        with self.assertRaises(RuntimeError):
            capture_features(model, Image.new("RGB", (16, 16)))
        self.assertFalse(projector._forward_pre_hooks)
        self.assertFalse(projector._forward_hooks)
