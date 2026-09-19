# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Select and visualize representative HazyDet validation predictions."""
# visualize_hazydet.py
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
)


@dataclass
class ImageResult:
    """Image metadata, predictions, and per-image detection statistics."""

    image_id: int
    path: Path
    width: int
    height: int
    gt_boxes: np.ndarray
    gt_classes: np.ndarray
    object_count: int
    mean_area_ratio: float
    small_ratio: float
    brightness: float
    contrast: float
    saturation: float
    haze_score: float = 0.0
    pred_boxes: np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=float))
    pred_classes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    pred_scores: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    matched_gt: set[int] = field(default_factory=set)
    matched_pred: set[int] = field(default_factory=set)
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    mean_iou: float = 0.0


def _xywh_to_xyxy(boxes: list[list[float]]) -> np.ndarray:
    """Convert COCO xywh boxes to an ``N x 4`` xyxy array."""
    if not boxes:
        return np.empty((0, 4), dtype=float)
    result = np.asarray(boxes, dtype=float).copy()
    result[:, 2] += result[:, 0]
    result[:, 3] += result[:, 1]
    return result


def _image_statistics(path: Path) -> tuple[float, float, float]:
    """Return brightness, contrast, and saturation from a small RGB preview."""
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    channel_max = rgb.max(axis=2)
    saturation = np.divide(
        channel_max - rgb.min(axis=2),
        channel_max,
        out=np.zeros_like(channel_max),
        where=channel_max > 1e-6,
    )
    return float(gray.mean()), float(gray.std()), float(saturation.mean())


def _rank01(values: np.ndarray) -> np.ndarray:
    """Map values to deterministic percentile ranks in the closed interval [0, 1]."""
    if len(values) <= 1:
        return np.full(len(values), 0.5, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks / (len(values) - 1)


def load_validation_records(dataset_dir: Path, split: str = "valid") -> tuple[list[ImageResult], dict[int, str]]:
    """Load COCO validation annotations and calculate image/dataset descriptors.

    Args:
        dataset_dir: Dataset root containing the selected split directory.
        split: COCO split name, normally ``valid`` or ``test``.

    Returns:
        Image records and a contiguous model-label-to-class-name mapping.
    """
    split_dir = dataset_dir / split
    annotation_path = split_dir / "_annotations.coco.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {annotation_path}")

    with annotation_path.open("r", encoding="utf-8") as file:
        coco = json.load(file)

    categories = sorted(coco["categories"], key=lambda item: int(item["id"]))
    category_to_label = {int(category["id"]): index for index, category in enumerate(categories)}
    label_to_name = {index: str(category["name"]) for index, category in enumerate(categories)}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        if not annotation.get("iscrowd", 0):
            annotations_by_image[int(annotation["image_id"])].append(annotation)

    records: list[ImageResult] = []
    for image_info in sorted(coco["images"], key=lambda item: (str(item["file_name"]), int(item["id"]))):
        image_id = int(image_info["id"])
        image_path = split_dir / image_info["file_name"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image listed in COCO annotations was not found: {image_path}")
        width, height = int(image_info["width"]), int(image_info["height"])
        annotations = annotations_by_image[image_id]
        gt_boxes = _xywh_to_xyxy([annotation["bbox"] for annotation in annotations])
        gt_classes = np.asarray(
            [category_to_label[int(annotation["category_id"])] for annotation in annotations], dtype=int
        )
        area_ratios = np.asarray(
            [float(annotation.get("area", annotation["bbox"][2] * annotation["bbox"][3])) / (width * height)
             for annotation in annotations],
            dtype=float,
        )
        brightness, contrast, saturation = _image_statistics(image_path)
        records.append(
            ImageResult(
                image_id=image_id,
                path=image_path,
                width=width,
                height=height,
                gt_boxes=gt_boxes,
                gt_classes=gt_classes,
                object_count=len(annotations),
                mean_area_ratio=float(area_ratios.mean()) if len(area_ratios) else 0.0,
                small_ratio=float((area_ratios < 0.01).mean()) if len(area_ratios) else 0.0,
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
            )
        )

    # This image-only haze score is a within-dataset proxy: bright, low-contrast,
    # low-saturation images receive higher scores. Prefer real haze labels when available.
    brightness_rank = _rank01(np.asarray([record.brightness for record in records]))
    contrast_rank = _rank01(np.asarray([record.contrast for record in records]))
    saturation_rank = _rank01(np.asarray([record.saturation for record in records]))
    for index, record in enumerate(records):
        record.haze_score = float(
            0.20 * brightness_rank[index] + 0.45 * (1 - contrast_rank[index]) + 0.35 * (1 - saturation_rank[index])
        )
    return records, label_to_name


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Calculate intersection over union for two xyxy boxes."""
    top_left = np.maximum(box_a[:2], box_b[:2])
    bottom_right = np.minimum(box_a[2:], box_b[2:])
    intersection = float(np.prod(np.maximum(bottom_right - top_left, 0)))
    area_a = float(np.prod(np.maximum(box_a[2:] - box_a[:2], 0)))
    area_b = float(np.prod(np.maximum(box_b[2:] - box_b[:2], 0)))
    return intersection / max(area_a + area_b - intersection, 1e-12)


def match_detections(record: ImageResult, iou_threshold: float) -> None:
    """Greedily match same-class predictions and GT, then update per-image metrics."""
    candidates: list[tuple[float, int, int]] = []
    for gt_index, (gt_box, gt_class) in enumerate(zip(record.gt_boxes, record.gt_classes)):
        for pred_index, (pred_box, pred_class) in enumerate(zip(record.pred_boxes, record.pred_classes)):
            if int(gt_class) == int(pred_class):
                iou = _iou(gt_box, pred_box)
                if iou >= iou_threshold:
                    candidates.append((iou, gt_index, pred_index))

    matched_ious: list[float] = []
    for iou, gt_index, pred_index in sorted(candidates, reverse=True):
        if gt_index not in record.matched_gt and pred_index not in record.matched_pred:
            record.matched_gt.add(gt_index)
            record.matched_pred.add(pred_index)
            matched_ious.append(iou)

    record.tp = len(record.matched_gt)
    record.fp = len(record.pred_boxes) - record.tp
    record.fn = len(record.gt_boxes) - record.tp
    record.precision = record.tp / max(record.tp + record.fp, 1)
    record.recall = record.tp / max(record.tp + record.fn, 1)
    denominator = record.precision + record.recall
    record.f1 = 2 * record.precision * record.recall / denominator if denominator else 0.0
    record.mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0


def run_inference(records: list[ImageResult], checkpoint: Path, threshold: float, iou_threshold: float) -> None:
    """Run RF-DETR inference and calculate metrics for every validation image."""
    from rfdetr import from_checkpoint

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = from_checkpoint(checkpoint, device="cuda")
    model.optimize_for_inference()

    for index, record in enumerate(records, start=1):
        detections = model.predict(str(record.path), threshold=threshold, include_source_image=False)
        record.pred_boxes = np.asarray(detections.xyxy, dtype=float).reshape(-1, 4)
        record.pred_classes = np.asarray(detections.class_id, dtype=int)
        record.pred_scores = np.asarray(detections.confidence, dtype=float)
        match_detections(record, iou_threshold)
        if index % 50 == 0 or index == len(records):
            print(f"Inference: {index}/{len(records)}")


def _choose_closest(
    records: list[ImageResult], attribute: str, target: float, used: set[int]
) -> int:
    """Return an unused record closest to a dataset-only target."""
    available = [index for index in range(len(records)) if index not in used]
    if not available:
        raise ValueError("Not enough distinct images to fill all representative slots.")
    return min(
        available,
        key=lambda index: (
            abs(float(getattr(records[index], attribute)) - target),
            record_sort_key(records[index]),
        ),
    )


def record_sort_key(record: ImageResult) -> tuple[str, int]:
    """Provide a stable tie-breaker independent of COCO JSON ordering."""
    return (record.path.name, record.image_id)


def select_representatives(
    records: list[ImageResult], samples_per_group: int = 1
) -> list[tuple[str, ImageResult]]:
    """Select distinct images for nine representative groups using image and GT properties."""
    if samples_per_group < 1:
        raise ValueError("samples_per_group must be at least 1.")
    required = 9 * samples_per_group
    if len(records) < required:
        raise ValueError(
            f"At least {required} validation images are required for "
            f"{samples_per_group} sample(s) in each of the nine representative groups."
        )

    selected: list[tuple[str, int]] = []
    used: set[int] = set()

    def add_closest(title: str, attribute: str, quantile: float) -> None:
        values = np.asarray([getattr(record, attribute) for record in records], dtype=float)
        target = float(np.quantile(values, quantile))
        for sample_number in range(1, samples_per_group + 1):
            index = _choose_closest(records, attribute, target, used)
            panel_title = title if samples_per_group == 1 else f"{title} ({sample_number}/{samples_per_group})"
            selected.append((panel_title, index))
            used.add(index)

    add_closest("Light haze", "haze_score", 0.10)
    add_closest("Moderate haze", "haze_score", 0.50)
    add_closest("Heavy haze", "haze_score", 0.90)
    add_closest("Sparse scene", "object_count", 0.10)
    add_closest("Typical density", "object_count", 0.50)
    add_closest("Dense scene", "object_count", 0.90)
    add_closest("Few small objects", "small_ratio", 0.10)
    add_closest("Mixed object sizes", "small_ratio", 0.50)
    add_closest("Many small objects", "small_ratio", 0.90)
    return [(title, records[index]) for title, index in selected]


def _sha256(path: Path) -> str:
    """Hash one selected image to detect content changes between experiments."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_or_create_fixed_selection(
    records: list[ImageResult], sample_file: Path, split: str, samples_per_group: int = 1
) -> list[tuple[str, ImageResult]]:
    """Persist panel IDs once, then reuse and validate the exact same image files."""
    if samples_per_group < 1:
        raise ValueError("samples_per_group must be at least 1.")
    expected_panels = 9 * samples_per_group
    if sample_file.exists():
        with sample_file.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        version = manifest.get("version")
        if version not in (1, 2) or manifest.get("split") != split:
            raise ValueError(f"Fixed-sample manifest has a different version or split: {sample_file}")
        panels = manifest.get("panels")
        manifest_samples = 1 if version == 1 else manifest.get("samples_per_group")
        if manifest_samples != samples_per_group or not isinstance(panels, list) or len(panels) != expected_panels:
            raise ValueError(
                f"Fixed-sample manifest does not match samples_per_group={samples_per_group} "
                f"({expected_panels} panels expected). Use a different --sample-file or remove the old manifest: "
                f"{sample_file}"
            )
        by_id = {record.image_id: record for record in records}
        selected: list[tuple[str, ImageResult]] = []
        used: set[int] = set()
        for panel in panels:
            image_id = int(panel["image_id"])
            record = by_id.get(image_id)
            if record is None or record.path.name != panel["file_name"] or _sha256(record.path) != panel["sha256"]:
                raise ValueError(f"A fixed image is missing or changed (image_id={image_id}): {sample_file}")
            if image_id in used:
                raise ValueError(f"Fixed-sample manifest contains a duplicate image_id={image_id}: {sample_file}")
            selected.append((str(panel["reason"]), record))
            used.add(image_id)
        print(f"Reusing fixed images from: {sample_file}")
        return selected

    selected = select_representatives(records, samples_per_group)
    manifest = {
        "version": 2,
        "split": split,
        "samples_per_group": samples_per_group,
        "panels": [
            {
                "reason": reason,
                "image_id": record.image_id,
                "file_name": record.path.name,
                "sha256": _sha256(record.path),
            }
            for reason, record in selected
        ],
    }
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    with sample_file.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Fixed image list created: {sample_file}")
    return selected


def _draw_box(axis: Any, box: np.ndarray, color: str, linestyle: str, label: str) -> None:
    """Draw one bounding box and a compact label."""
    x1, y1, x2, y2 = box
    axis.add_patch(
        Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.4, linestyle=linestyle)
    )
    axis.text(
        x1,
        max(y1 - 2, 0),
        label,
        color="white",
        fontsize=6,
        bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
    )


def save_visualizations(
    records: list[ImageResult],
    selected: list[tuple[str, ImageResult]],
    label_to_name: dict[int, str],
    output_dir: Path,
) -> None:
    """Save the representative grid, distribution plot, and selection table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = 3
    rows = math.ceil(len(selected) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows), constrained_layout=True)
    flat_axes = np.atleast_1d(axes).flat
    for axis, (title, record) in zip(flat_axes, selected):
        with Image.open(record.path) as image:
            axis.imshow(image.convert("RGB"))

        for gt_index, (box, class_id) in enumerate(zip(record.gt_boxes, record.gt_classes)):
            matched = gt_index in record.matched_gt
            color = "#2E8B57" if matched else "#F39C12"
            prefix = "GT" if matched else "FN"
            _draw_box(axis, box, color, "--", f"{prefix}: {label_to_name.get(int(class_id), str(class_id))}")

        for pred_index, (box, class_id, score) in enumerate(
            zip(record.pred_boxes, record.pred_classes, record.pred_scores)
        ):
            matched = pred_index in record.matched_pred
            color = "#2878B5" if matched else "#C82423"
            prefix = "TP" if matched else "FP"
            name = label_to_name.get(int(class_id), str(class_id))
            _draw_box(axis, box, color, "-", f"{prefix}: {name} {score:.2f}")

        axis.set_title(
            f"{title} | haze={record.haze_score:.2f}\n"
            f"TP={record.tp}  FP={record.fp}  FN={record.fn}  F1={record.f1:.2f}",
            fontsize=8,
        )
        axis.axis("off")

    for axis in flat_axes:
        axis.axis("off")

    fig.suptitle("Representative HazyDet validation examples", fontsize=12)
    fig.savefig(output_dir / "representative_grid.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "representative_grid.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    axis.scatter(
        [record.haze_score for record in records],
        [record.f1 for record in records],
        s=12,
        c="#B8B8B8",
        alpha=0.55,
        label="All validation images",
    )
    for number, (_, record) in enumerate(selected, start=1):
        axis.scatter(record.haze_score, record.f1, s=48, c="#C82423", edgecolors="white", linewidths=0.6)
        axis.annotate(str(number), (record.haze_score, record.f1), xytext=(4, 4), textcoords="offset points")
    axis.set(xlabel="Relative haze score", ylabel="Per-image F1", xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))
    axis.legend(loc="lower left")
    fig.savefig(output_dir / "selection_distribution.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "selection_distribution.pdf", bbox_inches="tight")
    plt.close(fig)

    with (output_dir / "representative_samples.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["panel", "reason", "image_id", "image", "haze_score", "objects", "small_ratio", "tp", "fp", "fn", "f1", "mean_iou"]
        )
        for panel, (reason, record) in enumerate(selected, start=1):
            writer.writerow(
                [
                    panel,
                    reason,
                    record.image_id,
                    record.path.name,
                    f"{record.haze_score:.6f}",
                    record.object_count,
                    f"{record.small_ratio:.6f}",
                    record.tp,
                    record.fp,
                    record.fn,
                    f"{record.f1:.6f}",
                    f"{record.mean_iou:.6f}",
                ]
            )


def generate_representative_visualization(
    dataset_dir: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    split: str = "valid",
    confidence_threshold: float = 0.30,
    iou_threshold: float = 0.50,
    sample_file: str | Path | None = None,
    samples_per_group: int = 1,
) -> None:
    """Run the complete representative-sample visualization pipeline.

    Args:
        dataset_dir: COCO-format HazyDet dataset root.
        checkpoint: Trained RF-DETR ``.pth`` checkpoint.
        output_dir: Directory in which figures and CSV are written.
        split: Dataset split to analyze.
        confidence_threshold: Prediction confidence threshold.
        iou_threshold: Same-class IoU threshold used to define a true positive.
        sample_file: Shared JSON manifest. Defaults to one file in the dataset root.
        samples_per_group: Number of images selected for each of the nine representative groups.
    """
    dataset_dir = Path(dataset_dir)
    records, label_to_name = load_validation_records(dataset_dir, split)
    fixed_file = Path(sample_file) if sample_file is not None else dataset_dir / f"fixed_samples_{split}.json"
    selected = load_or_create_fixed_selection(records, fixed_file, split, samples_per_group)
    run_inference(records, Path(checkpoint), confidence_threshold, iou_threshold)
    save_visualizations(records, selected, label_to_name, Path(output_dir))
    print(f"Representative visualizations saved to: {output_dir}")


def main() -> None:
    """Parse command-line arguments and generate representative visualizations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--sample-file", type=Path, help="Shared fixed-image JSON manifest")
    parser.add_argument(
        "--samples-per-group",
        type=int,
        default=1,
        help="Number of images selected for each representative group (default: 1)",
    )
    args = parser.parse_args()
    generate_representative_visualization(
        dataset_dir=args.dataset_dir,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        sample_file=args.sample_file,
        samples_per_group=args.samples_per_group,
    )


if __name__ == "__main__":
    main()
