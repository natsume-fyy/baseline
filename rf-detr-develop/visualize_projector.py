# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Compare clear/hazy images at the actual Projector boundary without training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def capture_features(model: Any, image: Image.Image) -> dict[str, np.ndarray]:
    """Capture all Projector input/output tensors using normal predict preprocessing.

    Args:
        model: Unoptimized RF-DETR wrapper in evaluation mode.
        image: RGB source image.

    Returns:
        Independent float32 CHW arrays, named by stage and level index.
    """
    backbone = model.model.model.backbone[0]
    features: dict[str, np.ndarray] = {}

    def snapshot(stage: str, tensors: Any) -> None:
        """Copy tensors before any subsequent operation can modify their storage."""
        for index, tensor in enumerate(tensors):
            if tensor.ndim != 4 or tensor.shape[0] != 1:
                raise ValueError(f"Expected batch-one BCHW at {stage}, got {tensor.shape}")
            features[f"{stage}_{index}"] = tensor[0].detach().float().cpu().numpy().copy()

    def before(module: Any, inputs: tuple[Any, ...]) -> None:
        """Record actual Projector inputs before its forward executes."""
        snapshot("pre", inputs[0])

    def after(module: Any, inputs: tuple[Any, ...], outputs: Any) -> None:
        """Record each output scale."""
        snapshot("post", outputs)

    handles = [backbone.projector.register_forward_pre_hook(before),
               backbone.projector.register_forward_hook(after)]
    try:
        with torch.inference_mode():
            model.predict(image, include_source_image=False)
    finally:
        for handle in handles:
            handle.remove()
    if not features:
        raise RuntimeError("Projector hooks did not run; use an unoptimized model.")
    return features


def activation_map(feature: np.ndarray) -> np.ndarray:
    """Return channel RMS, avoiding cancellation between signed activations."""
    if feature.ndim != 3 or not np.isfinite(feature).all():
        raise ValueError("Features must be finite CHW arrays.")
    return np.sqrt(np.mean(np.square(feature, dtype=np.float64), axis=0))


def paired_metrics(clear: np.ndarray, haze: np.ndarray) -> dict[str, float | None]:
    """Measure aligned same-layer feature drift, excluding undefined zero-vector cosines."""
    if clear.shape != haze.shape:
        raise ValueError("Aligned features must have identical shapes.")
    a, b = clear.astype(np.float64), haze.astype(np.float64)
    denominator = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    valid = denominator > 1e-12
    cosine = np.sum(a * b, axis=0)[valid] / denominator[valid]
    norm = np.linalg.norm(a)
    return {
        "cosine_mean": float(np.clip(cosine, -1, 1).mean()) if valid.any() else None,
        "cosine_valid_fraction": float(valid.mean()),
        "relative_l2": float(np.linalg.norm(b - a) / norm) if norm > 1e-12 else None,
    }


def save_comparison(
    images: list[Image.Image],
    captures: list[dict[str, np.ndarray]],
    output_dir: Path,
    aligned: bool,
    save_raw: bool,
) -> list[dict[str, Any]]:
    """Save shared-scale overlays, native maps, optional tensors and drift statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = list(captures[0])
    if list(captures[1]) != keys:
        raise ValueError("Projector levels differ between images.")
    rows = 3 if aligned else 2
    fig, axes = plt.subplots(rows, len(keys) + 1, squeeze=False,
                             figsize=(3.4 * (len(keys) + 1), 3.5 * rows), constrained_layout=True)
    statistics: list[dict[str, Any]] = []
    maps: dict[str, np.ndarray] = {}
    try:
        for row, (label, image) in enumerate(zip(("clear", "haze"), images)):
            axes[row, 0].imshow(image)
            axes[row, 0].set_title(label)
        for column, key in enumerate(keys, 1):
            heatmaps = [activation_map(capture[key]) for capture in captures]
            # One scale per layer shared by both images. No independent min-max scaling.
            upper = max(float(heatmap.max()) for heatmap in heatmaps) or 1.0
            for row, (label, image, capture, heatmap) in enumerate(
                zip(("clear", "haze"), images, captures, heatmaps)
            ):
                axis = axes[row, column]
                axis.imshow(image)
                artist = axis.imshow(heatmap, cmap="magma", vmin=0, vmax=upper, alpha=0.65,
                                     extent=(0, image.width, image.height, 0), interpolation="nearest")
                axis.set_title(f"{label} | {key}\nC,H,W={capture[key].shape}")
                fig.colorbar(artist, ax=axis, shrink=0.65, label="channel RMS")
                maps[f"{label}_{key}"] = heatmap
                statistics.append({"weather": label, "level": key, "shape": str(capture[key].shape),
                                   "rms_mean": float(heatmap.mean()), "rms_spatial_std": float(heatmap.std()),
                                   "color_vmin": 0.0, "color_vmax": upper})
            if aligned:
                difference = activation_map(captures[1][key] - captures[0][key])
                maps[f"difference_{key}"] = difference
                metrics = paired_metrics(captures[0][key], captures[1][key])
                statistics.append({"weather": "aligned_pair", "level": key, **metrics})
                artist = axes[2, column].imshow(difference, cmap="inferno", vmin=0)
                axes[2, column].set_title(f"{key}: RMS(haze - clear)")
                fig.colorbar(artist, ax=axes[2, column], shrink=0.65)
        for axis in axes.flat:
            axis.axis("off")
        fig.suptitle("Projector features (activation energy, NOT attention)\n"
                     "Color scales shared across weather within each level; pre/post spaces differ")
        fig.savefig(output_dir / "comparison.png", dpi=160)
    finally:
        plt.close(fig)
    np.savez_compressed(output_dir / "activation_maps.npz", **maps)
    if save_raw:
        for label, capture in zip(("clear", "haze"), captures):
            np.savez_compressed(output_dir / f"{label}_features.npz", **capture)
    columns = sorted({key for row in statistics for key in row})
    with (output_dir / "statistics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(statistics)
    return statistics


def main() -> None:
    """Load an existing checkpoint and compare explicit clear/haze image pairs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("/root/autodl-tmp/rf-ddetr/output/hazydet_small_baseline/checkpoint_best_total.pth"))
    parser.add_argument("--dataset-dir", type=Path, help="COCO dataset root; sample its selected split")
    parser.add_argument("--split", default="valid", choices=("train", "valid", "test"))
    parser.add_argument("--clear", type=Path, nargs="+")
    parser.add_argument("--haze", type=Path, nargs="+")
    parser.add_argument("--image-dir", type=Path, help="Recursively sample a mixed/unlabelled image directory")
    parser.add_argument("--clear-dir", type=Path, help="Directory with known clear-weather images")
    parser.add_argument("--haze-dir", type=Path, help="Directory with known hazy-weather images")
    parser.add_argument("--num-samples", type=int, default=200, help="Total images, not pairs (default: 200)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("output/projector_features"))
    parser.add_argument("--device", default=None, help="cuda or cpu; defaults to available hardware")
    parser.add_argument("--aligned", action="store_true", help="Pairs depict the same pixel-aligned scene")
    parser.add_argument("--save-raw", action="store_true", help="Also save full CHW float32 features")
    args = parser.parse_args()
    explicit_images = any((args.image_dir, args.clear_dir, args.haze_dir, args.clear, args.haze))
    if args.dataset_dir and explicit_images:
        parser.error("--dataset-dir cannot be combined with other image sources")
    if not explicit_images:
        args.image_dir = (args.dataset_dir or Path("/root/autodl-tmp/HazyDet_RFDETR")) / args.split
    batch_mode = any((args.image_dir, args.clear_dir, args.haze_dir))
    if batch_mode:
        if args.clear or args.haze or args.aligned:
            parser.error("Directory sampling cannot be combined with explicit pairs or --aligned")
        from projector_batch import run_batch

        run_batch(args)
        return
    if not args.clear or not args.haze:
        parser.error("Provide --image-dir, both --clear-dir/--haze-dir, or explicit --clear/--haze lists")
    if len(args.clear) != len(args.haze):
        parser.error("--clear and --haze must contain the same number of files, paired in order")
    for path in [args.checkpoint, *args.clear, *args.haze]:
        if not path.is_file():
            parser.error(f"File not found: {path}")
    if not hasattr(torch, "inference_mode"):
        parser.error("A working PyTorch installation is required; run in your RF-DETR training environment.")
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    from rfdetr import from_checkpoint

    # Explicitly omit the training-only HBS branch; its learned influence on backbone weights remains.
    model = from_checkpoint(args.checkpoint, device=args.device, hbs_enabled=False)
    model.model.model.eval()
    backbone = model.model.model.backbone[0]
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()), "hbs_enabled_at_inference": False,
        "resolution": model.model.resolution, "device": args.device, "aligned": args.aligned,
        "model_class": type(model).__name__,
        "encoder_feature_indexes": model.model_config.out_feature_indexes,
        "projector_scales": backbone.projector_scale,
        "note": "HBS-trained weights remain HBS-trained. Heatmaps are not attention or detection accuracy.",
        "pairs": [],
    }
    for index, (clear_path, haze_path) in enumerate(zip(args.clear, args.haze)):
        images = []
        for path in (clear_path, haze_path):
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
        if args.aligned and images[0].size != images[1].size:
            raise ValueError("--aligned requires equal image dimensions and actual pixel alignment.")
        captures = [capture_features(model, image) for image in images]
        pair_dir = args.output_dir / f"pair_{index:03d}"
        save_comparison(images, captures, pair_dir, args.aligned, args.save_raw)
        metadata["pairs"].append({"clear": str(clear_path.resolve()), "haze": str(haze_path.resolve()),
                                  "output": str(pair_dir.resolve())})
        print(f"Saved: {pair_dir / 'comparison.png'}")
    (args.output_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
