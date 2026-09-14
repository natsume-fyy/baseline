# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Compare baseline and HBS checkpoints on the same COCO validation images."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from visualize_hazydet import ImageResult, load_validation_records, match_detections


def make_groups(records: list[ImageResult], min_images: int = 10) -> dict[str, list[int]]:
    """Define image types from input images and labels, without using predictions.

    Args:
        records: Validation images in a stable order.
        min_images: Minimum size for every reported group.

    Returns:
        Group names mapped to image positions. A group below the minimum is omitted.
    """
    if min_images < 10:
        raise ValueError("min_images must be at least 10")
    groups: dict[str, list[int]] = {}
    if len(records) < min_images:
        raise ValueError(f"Validation set has {len(records)} images; at least {min_images} are required")
    groups["all"] = list(range(len(records)))

    def add_tertiles(name: str, value: Callable[[ImageResult], float]) -> None:
        """Add low, middle, and high groups with numeric tie boundaries."""
        values = np.asarray([value(record) for record in records], dtype=float)
        lower, upper = np.quantile(values, [1 / 3, 2 / 3])
        partitions = {
            "low": np.flatnonzero(values <= lower).tolist(),
            "middle": np.flatnonzero((values > lower) & (values <= upper)).tolist(),
            "high": np.flatnonzero(values > upper).tolist(),
        }
        for level, indices in partitions.items():
            if len(indices) >= min_images:
                groups[f"{name}_{level}"] = indices

    add_tertiles("haze_proxy", lambda record: record.haze_score)
    add_tertiles("contrast", lambda record: record.contrast)
    add_tertiles("object_count", lambda record: float(record.object_count))
    add_tertiles("mean_object_area", lambda record: record.mean_area_ratio)

    for name, opposite, predicate in [
        ("has_small_object", "without_small_object", lambda record: record.small_ratio > 0),
        (
            "mostly_small_objects",
            "not_mostly_small_objects",
            lambda record: record.small_ratio >= 0.5 and record.object_count > 0,
        ),
    ]:
        indices = [index for index, record in enumerate(records) if predicate(record)]
        selected = set(indices)
        if len(indices) >= min_images:
            groups[name] = indices
        complement = [index for index in range(len(records)) if index not in selected]
        if len(complement) >= min_images:
            groups[opposite] = complement
    return groups


def evaluate_checkpoint(
    records: list[ImageResult], checkpoint: Path, device: str, confidence: float, iou: float
) -> list[ImageResult]:
    """Evaluate one checkpoint on all records with class-aware IoU matching."""
    from rfdetr import from_checkpoint

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = from_checkpoint(checkpoint, device=device)
    model.optimize_for_inference()
    results = copy.deepcopy(records)
    for number, record in enumerate(results, start=1):
        detections = model.predict(str(record.path), threshold=confidence, include_source_image=False)
        record.pred_boxes = np.asarray(detections.xyxy, dtype=float).reshape(-1, 4)
        record.pred_classes = np.asarray(detections.class_id, dtype=int)
        record.pred_scores = np.asarray(detections.confidence, dtype=float)
        match_detections(record, iou)
        if number % 50 == 0 or number == len(results):
            print(f"{checkpoint.name}: {number}/{len(results)}")
    return results


def _micro_f1(records: list[ImageResult], indices: list[int]) -> float:
    """Calculate pooled F1 from summed true, false, and missed detections."""
    tp = sum(records[index].tp for index in indices)
    fp = sum(records[index].fp for index in indices)
    fn = sum(records[index].fn for index in indices)
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def summarize_groups(
    baseline: list[ImageResult], hbs: list[ImageResult], groups: dict[str, list[int]], seed: int = 0
) -> list[dict[str, str | int | float]]:
    """Calculate paired group differences and image-bootstrap confidence intervals."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, str | int | float]] = []
    for name, indices in groups.items():
        changes = np.asarray([hbs[index].f1 - baseline[index].f1 for index in indices], dtype=float)
        samples = rng.choice(changes, size=(2000, len(changes)), replace=True).mean(axis=1)
        low, high = np.quantile(samples, [0.025, 0.975])
        base_micro = _micro_f1(baseline, indices)
        hbs_micro = _micro_f1(hbs, indices)
        rows.append(
            {
                "group": name,
                "n_images": len(indices),
                "baseline_micro_f1": round(base_micro, 6),
                "hbs_micro_f1": round(hbs_micro, 6),
                "delta_micro_f1": round(hbs_micro - base_micro, 6),
                "mean_paired_image_f1_delta": round(float(changes.mean()), 6),
                "paired_delta_ci95_low": round(float(low), 6),
                "paired_delta_ci95_high": round(float(high), 6),
                "hbs_better_images": int((changes > 0).sum()),
                "hbs_worse_images": int((changes < 0).sum()),
                "equal_images": int((changes == 0).sum()),
            }
        )
    return rows


def write_results(
    output_dir: Path,
    records: list[ImageResult],
    baseline: list[ImageResult],
    hbs: list[ImageResult],
    groups: dict[str, list[int]],
    summary: list[dict[str, str | int | float]],
    settings: dict[str, str | float | int],
) -> None:
    """Save the group summary, paired image table, and experiment settings."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "group_summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    membership = {index: [] for index in range(len(records))}
    for name, indices in groups.items():
        for index in indices:
            membership[index].append(name)
    with (output_dir / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id", "file_name", "groups", "haze_proxy", "contrast", "object_count",
                "mean_object_area", "small_ratio", "baseline_tp", "baseline_fp", "baseline_fn",
                "baseline_f1", "hbs_tp", "hbs_fp", "hbs_fn", "hbs_f1", "delta_f1",
            ],
        )
        writer.writeheader()
        for index, (record, base, changed) in enumerate(zip(records, baseline, hbs)):
            writer.writerow(
                {
                    "image_id": record.image_id,
                    "file_name": record.path.name,
                    "groups": ";".join(membership[index]),
                    "haze_proxy": record.haze_score,
                    "contrast": record.contrast,
                    "object_count": record.object_count,
                    "mean_object_area": record.mean_area_ratio,
                    "small_ratio": record.small_ratio,
                    "baseline_tp": base.tp,
                    "baseline_fp": base.fp,
                    "baseline_fn": base.fn,
                    "baseline_f1": base.f1,
                    "hbs_tp": changed.tp,
                    "hbs_fp": changed.fp,
                    "hbs_fn": changed.fn,
                    "hbs_f1": changed.f1,
                    "delta_f1": changed.f1 - base.f1,
                }
            )
    (output_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Run a paired, fixed-split comparison of two trained checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--hbs-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--min-images", type=int, default=10)
    args = parser.parse_args()
    if args.baseline_checkpoint.resolve() == args.hbs_checkpoint.resolve():
        parser.error("Baseline and HBS checkpoints must be different files")
    records, _ = load_validation_records(args.dataset_dir, args.split)
    groups = make_groups(records, args.min_images)
    baseline = evaluate_checkpoint(records, args.baseline_checkpoint, args.device, args.confidence, args.iou)
    hbs = evaluate_checkpoint(records, args.hbs_checkpoint, args.device, args.confidence, args.iou)
    summary = summarize_groups(baseline, hbs, groups)
    settings = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "split": args.split,
        "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
        "hbs_checkpoint": str(args.hbs_checkpoint.resolve()),
        "confidence": args.confidence,
        "iou": args.iou,
        "min_images": args.min_images,
        "haze_proxy_note": "Relative brightness/contrast/saturation proxy; not measured physical haze.",
    }
    write_results(args.output_dir, records, baseline, hbs, groups, summary, settings)
    print(f"Saved paired comparison to: {args.output_dir}")


if __name__ == "__main__":
    main()
