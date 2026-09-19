# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for model-independent, reusable HazyDet image selection."""

import json
from pathlib import Path

import numpy as np
import pytest

from visualize_hazydet import ImageResult, load_or_create_fixed_selection, select_representatives


def _records(directory: Path) -> list[ImageResult]:
    """Create nine synthetic records with distinct files and image descriptors."""
    records = []
    for index in range(9):
        path = directory / f"image_{index}.jpg"
        path.write_bytes(f"image {index}".encode())
        records.append(
            ImageResult(
                image_id=index,
                path=path,
                width=100,
                height=100,
                gt_boxes=np.empty((0, 4)),
                gt_classes=np.empty(0, dtype=int),
                object_count=index,
                mean_area_ratio=0.01,
                small_ratio=index / 8,
                brightness=0.5,
                contrast=0.1,
                saturation=0.2,
                haze_score=index / 8,
                f1=index / 8,
                fp=index,
            )
        )
    return records


def _many_records(directory: Path, count: int) -> list[ImageResult]:
    records = []
    for index in range(count):
        path = directory / f"many_{index}.jpg"
        path.write_bytes(f"image {index}".encode())
        records.append(
            ImageResult(
                image_id=index,
                path=path,
                width=100,
                height=100,
                gt_boxes=np.empty((0, 4)),
                gt_classes=np.empty(0, dtype=int),
                object_count=index,
                mean_area_ratio=0.01,
                small_ratio=index / max(count - 1, 1),
                brightness=0.5,
                contrast=0.1,
                saturation=0.2,
                haze_score=index / max(count - 1, 1),
            )
        )
    return records


def test_selection_does_not_depend_on_model_predictions(tmp_path: Path) -> None:
    """Changing model metrics cannot change the selected image IDs or order."""
    records = _records(tmp_path)
    before = [record.image_id for _, record in select_representatives(records)]
    for record in records:
        record.f1 = 1 - record.f1
        record.fp = 100 - record.fp
        record.fn = record.image_id * 2
    after = [record.image_id for _, record in select_representatives(list(reversed(records)))]
    assert before == after


def test_manifest_reuses_images_and_detects_changed_content(tmp_path: Path) -> None:
    """A later experiment reuses the manifest and rejects modified images."""
    records = _records(tmp_path)
    manifest = tmp_path / "fixed_samples_valid.json"
    first = load_or_create_fixed_selection(records, manifest, "valid")
    later = load_or_create_fixed_selection(list(reversed(records)), manifest, "valid")
    assert [record.image_id for _, record in first] == [record.image_id for _, record in later]

    first[0][1].path.write_bytes(b"different image")
    with pytest.raises(ValueError, match="missing or changed"):
        load_or_create_fixed_selection(records, manifest, "valid")


def test_multiple_samples_per_group_are_distinct_and_persisted(tmp_path: Path) -> None:
    records = _many_records(tmp_path, 27)
    manifest = tmp_path / "fixed_samples_valid_3.json"
    selected = load_or_create_fixed_selection(records, manifest, "valid", samples_per_group=3)

    assert len(selected) == 27
    assert len({record.image_id for _, record in selected}) == 27
    assert json.loads(manifest.read_text(encoding="utf-8"))["samples_per_group"] == 3
