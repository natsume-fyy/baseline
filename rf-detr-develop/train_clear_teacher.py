# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Train a clear-image RF-DETR Small teacher without HBS or distillation.

Run on the training server. The generated COCO view references existing images
by absolute path; it does not copy images or modify the source annotations.
"""
# train_clear_teacher.py
import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image


def prepare_clear_dataset(
    source_dir: Path,
    destination: Path,
    clear_train_dir: Path,
    clear_valid_dir: Path | None = None,
    pair_manifest: Path | None = None,
    valid_manifest: Path | None = None,
) -> Path:
    """Create train/valid COCO indexes while preserving annotations and split membership.

    Args:
        source_dir: Existing Roboflow COCO dataset with train and valid splits.
        destination: Separate directory for generated annotation indexes.
        clear_train_dir: Directory of geometrically aligned clear training images.
        clear_valid_dir: Optional clear validation images; otherwise use existing valid images.
        pair_manifest: Optional COCO filename -> {"clear": relative_path} mapping for train.
        valid_manifest: The same mapping format for clear validation images.

    Returns:
        Absolute path to the generated dataset view.

    Raises:
        ValueError: If geometry is incompatible, a mapping is incomplete, or indexes would be overwritten.
        FileNotFoundError: If an annotation or image is missing.
    """
    source_dir, destination = source_dir.resolve(), destination.resolve()
    if destination == source_dir or destination in source_dir.parents or source_dir in destination.parents:
        raise ValueError("Use a separate prepared dataset directory outside the source dataset.")
    if valid_manifest is not None and clear_valid_dir is None:
        raise ValueError("--valid-manifest requires --clear-valid-dir.")
    prepared: dict[str, str] = {}
    split_paths: dict[str, set[Path]] = {}
    for split in ("train", "valid"):
        annotation_path = source_dir / split / "_annotations.coco.json"
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        image_root = clear_train_dir if split == "train" else clear_valid_dir
        manifest_path = pair_manifest if split == "train" else valid_manifest
        mapping = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path else None
        if mapping is not None and not isinstance(mapping, dict):
            raise ValueError(f"Manifest must be a JSON object: {manifest_path}")
        split_paths[split] = set()
        for record in data["images"]:
            filename = record["file_name"]
            if image_root is None:
                path = source_dir / split / filename
            else:
                if mapping is None:
                    relative_name = filename
                    if Path(relative_name).is_absolute():
                        raise ValueError(f"Use a manifest to map absolute COCO filenames to clear images: {filename}")
                else:
                    entry = mapping.get(filename)
                    if not isinstance(entry, dict) or not isinstance(entry.get("clear"), str):
                        raise ValueError(f"Missing 'clear' mapping for {split}/{filename}")
                    relative_name = entry["clear"]
                    if Path(relative_name).is_absolute():
                        raise ValueError(f"Manifest clear paths must be relative to the clear directory: {filename}")
                path = image_root / relative_name
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Missing {split} image: {path}. Use a manifest if filenames differ.")
            with Image.open(path) as image:
                expected_size = (record["width"], record["height"])
                if image.size != expected_size:
                    raise ValueError(
                        f"Image/COCO size mismatch for {path}: {image.size} versus {expected_size}. "
                        "Use annotations in the clear image coordinate system; do not just rename resized exports."
                    )
            record["file_name"] = str(path)
            split_paths[split].add(path)
        prepared[split] = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if split_paths["train"] & split_paths["valid"]:
        raise ValueError("The same image file appears in both training and validation.")
    # Complete all validation before creating any indexes; reruns with identical inputs are safe.
    for split, content in prepared.items():
        path = destination / split / "_annotations.coco.json"
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Existing index differs: {path}. Choose a new --prepared-dataset-dir.")
    for split, content in prepared.items():
        path = destination / split / "_annotations.coco.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(content, encoding="utf-8")
    return destination


def build_parser() -> argparse.ArgumentParser:
    """Build teacher-training options with the current student's architecture defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset-dir", type=Path, default=Path("/root/autodl-tmp/HazyDet_RFDETR"))
    parser.add_argument("--clear-train-dir", type=Path, default=Path("/root/autodl-tmp/HazyDet/train/images"))
    parser.add_argument("--clear-valid-dir", type=Path, help="Optional clear validation directory")
    parser.add_argument(
        "--pair-manifest", type=Path, help="Train filename mapping, compatible with student KD manifest"
    )
    parser.add_argument("--valid-manifest", type=Path, help="Optional clear validation filename mapping")
    parser.add_argument(
        "--prepared-dataset-dir", type=Path, default=Path("/root/autodl-tmp/HazyDet_RFDETR_clear_teacher")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/root/autodl-tmp/rf-detr/output/hazydet_clear_teacher")
    )
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--prepare-only", action="store_true", help="Validate images and create indexes without training"
    )
    return parser


def train_teacher(args: argparse.Namespace, dataset_dir: Path) -> Any:
    """Train the standard Small detector; its checkpoint is compatible with the HBS student's feature teacher."""
    from rfdetr import RFDETRSmall

    # Same default Small architecture/resolution (512) as train_hazydet.py.
    # Teacher HBS is unnecessary because KD only loads backbone + projector.
    model = RFDETRSmall(hbs_enabled=False)
    return model.train(
        dataset_dir=str(dataset_dir),
        dataset_file="roboflow",
        output_dir=str(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        hbs_loss_coef=0.0,
        fg_distill_coef=0.0,
        augmentation_backend="cpu",
        device=args.device,
        num_workers=args.num_workers,
        use_ema=True,
        checkpoint_interval=5,
        early_stopping=False,
    )


def main() -> None:
    """Prepare clear training data and optionally train the teacher."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        dataset_dir = prepare_clear_dataset(
            args.source_dataset_dir,
            args.prepared_dataset_dir,
            args.clear_train_dir,
            args.clear_valid_dir,
            args.pair_manifest,
            args.valid_manifest,
        )
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Prepared dataset: {dataset_dir}")
    print(f"Validation images: {args.clear_valid_dir or args.source_dataset_dir / 'valid'}")
    if args.clear_valid_dir is None:
        print("Validation uses the existing split (typically hazy); checkpoint selection follows that domain.")
    if args.prepare_only:
        return
    train_teacher(args, dataset_dir)
    checkpoint = args.output_dir / "checkpoint_best_total.pth"
    print(f'Next: python train_hazydet.py --teacher-weights "{checkpoint}"')


if __name__ == "__main__":
    main()
