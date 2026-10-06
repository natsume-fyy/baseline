# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Numerical and CLI checks; synthetic images are not HazyDet evidence."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from visualize_hazydet_frequency import background_box, exclusion_integral, spectrum


class FrequencyTests(unittest.TestCase):
    """Check frequency meaning, power scaling, and background isolation."""

    def test_constant_has_no_ac_energy(self) -> None:
        """A constant patch must not acquire texture through windowing."""
        result = spectrum(np.full((32, 48), 0.7), 32, (0.08, 0.2))
        self.assertLess(result["energy"], 1e-20)

    def test_known_sine_frequency(self) -> None:
        """A horizontal sine must peak at its known cycles/pixel frequency."""
        patch = np.tile(np.sin(2 * np.pi * np.arange(128) / 8), (64, 1))
        result = spectrum(patch, 64, (0.08, 0.2))
        self.assertAlmostEqual(result["frequency"][result["radial"].argmax()], 0.125, delta=0.015)
        self.assertGreater(result["bands"][1], 0.99)

    def test_amplitude_squared_scaling(self) -> None:
        """Normalized fractions stay fixed while absolute power scales quadratically."""
        patch = np.random.default_rng(1).random((48, 64))
        first = spectrum(patch, 32, (0.08, 0.2))
        second = spectrum(patch * 3 + 5, 32, (0.08, 0.2))
        self.assertAlmostEqual(second["energy"] / first["energy"], 9)
        np.testing.assert_allclose(first["radial"], second["radial"], atol=1e-12)
        self.assertAlmostEqual(first["bands"].sum(), 1)

    def test_background_excludes_all_boxes(self) -> None:
        """Sampling must avoid both target and ignored/crowd box regions."""
        integral = exclusion_integral(80, 100, [(0, 0, 50, 80)], 3)
        box = background_box(integral, 20, 20, np.random.default_rng(4), 500)
        self.assertIsNotNone(box)
        self.assertGreaterEqual(box[0], 53)

    def test_no_background(self) -> None:
        """A fully annotated image has no valid background patch."""
        integral = exclusion_integral(40, 40, [(0, 0, 40, 40)], 0)
        self.assertIsNone(background_box(integral, 16, 16, np.random.default_rng(2), 30))

    def test_cli(self) -> None:
        """A tiny COCO fixture produces plots, reproducible samples, and provenance."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = root / "valid"
            split.mkdir()
            rgb = np.random.default_rng(3).integers(0, 256, (96, 128, 3), dtype=np.uint8)
            Image.fromarray(rgb).save(split / "synthetic.png")
            coco = {
                "images": [{"id": 1, "file_name": "synthetic.png", "width": 128, "height": 96}],
                "categories": [{"id": 1, "name": "synthetic"}],
                "annotations": [{"id": 7, "image_id": 1, "category_id": 1, "bbox": [5, 5, 24, 32]}],
            }
            (split / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
            command = [sys.executable, str(ROOT / "visualize_hazydet_frequency.py"),
                       "--dataset-dir", str(root), "--output-dir", str(root / "out"), "--examples", "1"]
            subprocess.run(command, check=True, capture_output=True, text=True)
            out = root / "out"
            report = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(report["counts"]["pairs"], 1)
            self.assertTrue((out / "summary.png").is_file())
            self.assertTrue((out / "example_001.png").is_file())
            original = (out / "pairs.csv").read_bytes()
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(original, (out / "pairs.csv").read_bytes())


if __name__ == "__main__":
    unittest.main()
