# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for paired HBS subgroup analysis."""

from pathlib import Path

import numpy as np
import pytest

from compare_hbs_hazydet import make_groups, summarize_groups
from visualize_hazydet import ImageResult


def _records(count: int) -> list[ImageResult]:
    """Make records with enough variation to exercise fixed subgroup definitions."""
    return [
        ImageResult(
            image_id=index,
            path=Path(f"image_{index}.jpg"),
            width=100,
            height=100,
            gt_boxes=np.empty((0, 4)),
            gt_classes=np.empty(0, dtype=int),
            object_count=index + 1,
            mean_area_ratio=(index + 1) / 1000,
            small_ratio=float(index % 2),
            brightness=0.5,
            contrast=index / count,
            saturation=0.2,
            haze_score=index / count,
        )
        for index in range(count)
    ]


def test_groups_require_at_least_ten_images_and_ignore_predictions() -> None:
    """Group membership must be fixed before either model is evaluated."""
    records = _records(30)
    before = make_groups(records)
    for record in records:
        record.f1 = 1 - record.haze_score
        record.fp = record.image_id
    assert make_groups(records) == before
    assert all(len(indices) >= 10 for indices in before.values())


def test_groups_reject_undersized_validation_set() -> None:
    """Do not report image-type results for fewer than ten images."""
    with pytest.raises(ValueError, match="at least 10"):
        make_groups(_records(9))


def test_summary_uses_paired_image_differences() -> None:
    """A benefit concentrated in one subgroup should remain visible."""
    baseline = _records(20)
    hbs = _records(20)
    for index, (base, changed) in enumerate(zip(baseline, hbs)):
        base.tp, base.fn, base.f1 = 1, 1, 0.5
        changed.tp, changed.fn = (2, 0) if index < 10 else (1, 1)
        changed.f1 = 1.0 if index < 10 else 0.5
    rows = summarize_groups(baseline, hbs, {"first": list(range(10)), "second": list(range(10, 20))})
    assert rows[0]["mean_paired_image_f1_delta"] == 0.5
    assert rows[0]["hbs_better_images"] == 10
    assert rows[1]["mean_paired_image_f1_delta"] == 0
