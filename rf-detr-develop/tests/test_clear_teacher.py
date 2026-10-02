# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Teacher entry-point checks requiring only Pillow and the standard library."""

import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

SPEC = importlib.util.spec_from_file_location(
    "train_clear_teacher", Path(__file__).parents[1] / "train_clear_teacher.py"
)
teacher_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(teacher_script)


class TestClearTeacher(unittest.TestCase):
    """Validate clear-only input selection, annotation preservation, and training flags."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.clear = self.root / "clear"
        self.destination = self.root / "prepared"
        self.clear.mkdir()
        for split in ("train", "valid"):
            folder = self.source / split
            folder.mkdir(parents=True)
            Image.new("RGB", (32, 24)).save(folder / f"{split}.png")
            data = {
                "images": [{"id": 1, "file_name": f"{split}.png", "width": 32, "height": 24}],
                "categories": [{"id": 1, "name": "car"}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 3, 4, 5]}],
            }
            (folder / "_annotations.coco.json").write_text(json.dumps(data), encoding="utf-8")
        Image.new("RGB", (32, 24)).save(self.clear / "train.png")

    def test_clear_train_existing_valid_and_idempotence(self) -> None:
        """Only the training image source changes; annotations and source indexes are intact."""
        original = (self.source / "train/_annotations.coco.json").read_text(encoding="utf-8")
        for _ in range(2):
            teacher_script.prepare_clear_dataset(self.source, self.destination, self.clear)
        train = json.loads((self.destination / "train/_annotations.coco.json").read_text(encoding="utf-8"))
        valid = json.loads((self.destination / "valid/_annotations.coco.json").read_text(encoding="utf-8"))
        self.assertEqual(Path(train["images"][0]["file_name"]), self.clear / "train.png")
        self.assertEqual(Path(valid["images"][0]["file_name"]), self.source / "valid/valid.png")
        self.assertEqual(train["annotations"], json.loads(original)["annotations"])
        self.assertEqual((self.source / "train/_annotations.coco.json").read_text(encoding="utf-8"), original)

    def test_manifest_and_clear_validation(self) -> None:
        """Explicit renaming and a separate clear validation root are supported."""
        (self.clear / "train.png").rename(self.clear / "original.png")
        Image.new("RGB", (32, 24)).save(self.clear / "valid.png")
        manifest = self.root / "pairs.json"
        manifest.write_text(json.dumps({"train.png": {"clear": "original.png", "hazy": "fog.png"}}))
        teacher_script.prepare_clear_dataset(self.source, self.destination, self.clear, self.clear, manifest)
        data = json.loads((self.destination / "valid/_annotations.coco.json").read_text(encoding="utf-8"))
        self.assertEqual(Path(data["images"][0]["file_name"]), self.clear / "valid.png")

    def test_mismatched_geometry_does_not_write_indexes(self) -> None:
        """A resized COCO export cannot silently supervise different image coordinates."""
        Image.new("RGB", (64, 48)).save(self.clear / "train.png")
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            teacher_script.prepare_clear_dataset(self.source, self.destination, self.clear)
        self.assertFalse(self.destination.exists())

    def test_missing_file_and_source_directory_protection(self) -> None:
        """Incorrect paths fail before training or overwriting source annotations."""
        with self.assertRaisesRegex(ValueError, "separate"):
            teacher_script.prepare_clear_dataset(self.source, self.source, self.clear)
        with self.assertRaises(FileNotFoundError):
            teacher_script.prepare_clear_dataset(self.source, self.destination, self.root / "missing")

    def test_training_disables_hbs_and_distillation(self) -> None:
        """Check the actual training call without downloads or GPU execution."""
        fake_package = types.ModuleType("rfdetr")
        factory = MagicMock()
        fake_package.RFDETRSmall = factory
        args = teacher_script.build_parser().parse_args([])
        with patch.dict("sys.modules", {"rfdetr": fake_package}):
            teacher_script.train_teacher(args, self.destination)
        factory.assert_called_once_with(hbs_enabled=False)
        options = factory.return_value.train.call_args.kwargs
        self.assertEqual(options["fg_distill_coef"], 0)
        self.assertEqual(options["hbs_loss_coef"], 0)
        self.assertEqual(options["dataset_dir"], str(self.destination))
        self.assertNotIn("fg_teacher_weights", options)


if __name__ == "__main__":
    unittest.main()
