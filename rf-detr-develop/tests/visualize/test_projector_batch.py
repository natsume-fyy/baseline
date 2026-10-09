# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""CPU-only tests for random sampling and diagnostic report generation."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from projector_batch import run_batch, select_images, summarize


class BatchDiagnosticsTest(unittest.TestCase):
    """Test reproducibility, weather separation, and the saved analysis pipeline."""

    def test_random_200(self) -> None:
        """Select exactly 200 distinct files with stable seeds, including nested files."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            for index in range(220):
                (root / "nested" / f"{index:04d}.JPG").touch()
            first = select_images({"unknown": root}, 200, 42)
            self.assertEqual(len({item["path"] for item in first}), 200)
            self.assertEqual(first, select_images({"unknown": root}, 200, 42))
            self.assertNotEqual(first, select_images({"unknown": root}, 200, 43))
            with self.assertRaises(ValueError):
                select_images({"unknown": root}, 221, 42)

    def test_balanced_groups(self) -> None:
        """A total count is divided across the two known weather groups."""
        with tempfile.TemporaryDirectory() as directory:
            sources = {group: Path(directory) / group for group in ("clear", "haze")}
            for source in sources.values():
                source.mkdir()
                for index in range(4):
                    (source / f"{index}.png").touch()
            samples = select_images(sources, 6, 1)
            self.assertEqual(sum(item["group"] == "clear" for item in samples), 3)
            with self.assertRaises(ValueError):
                select_images({"clear": sources["clear"], "haze": sources["clear"]}, 2, 1)

    def test_summary_constant(self) -> None:
        """Undefined correlations stay missing rather than emitting NaN."""
        rows = [{"level": "pre_0", "group": "unknown", "rms_mean": 2.0,
                 "spatial_cv": 0.0, "image_contrast": 0.2}] * 3
        self.assertIsNone(summarize(rows)[0]["contrast_energy_pearson"])

    def test_batch_pipeline_with_synthetic_features(self) -> None:
        """Exercise extraction-to-report plumbing without pretending to run RF-DETR."""
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "inputs"
            images.mkdir()
            for index in range(2):
                Image.new("RGB", (24, 16), (index * 80, 30, 40)).save(images / f"{index}.png")
            checkpoint = root / "fake.pth"
            checkpoint.touch()
            args = argparse.Namespace(image_dir=images, clear_dir=None, haze_dir=None, num_samples=2,
                                      seed=42, checkpoint=checkpoint, output_dir=root / "outputs", device="cpu", save_raw=True)
            module = SimpleNamespace(eval=lambda: None, backbone=[SimpleNamespace(projector_scale=["P4"])])
            model = SimpleNamespace(model=SimpleNamespace(model=module, resolution=32),
                                    model_config=SimpleNamespace(out_feature_indexes=[2]))
            features = {"pre_0": np.ones((3, 4, 4), dtype=np.float32),
                        "post_0": np.ones((2, 2, 2), dtype=np.float32) * 2}
            with patch.object(torch, "inference_mode", create=True), \
                 patch.dict("sys.modules", {"rfdetr": SimpleNamespace(from_checkpoint=lambda *a, **k: model)}), \
                 patch("visualize_projector.capture_features", return_value=features):
                run_batch(args)
            manifest = json.loads((args.output_dir / "samples.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["count"], 2)
            self.assertEqual(len(list((args.output_dir / "images").glob("*.png"))), 2)
            report = (args.output_dir / "analysis.md").read_text(encoding="utf-8")
            self.assertIn("没有可靠天气标签", report)
            self.assertTrue((args.output_dir / "summary.png").is_file())
            self.assertTrue((args.output_dir / "raw" / "0000.npz").is_file())
