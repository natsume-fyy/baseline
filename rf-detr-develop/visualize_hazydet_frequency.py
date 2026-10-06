# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Compare native-resolution COCO object boxes with matched background patches.

CPU only; no model checkpoint required. See HAZYDET_FREQUENCY.md for interpretation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

COLORS = ("#397CA5", "#C18048")
BANDS = ("low", "mid", "high")
MAX_FREQUENCY = np.sqrt(0.5)


def spectrum(patch: np.ndarray, bins: int, cuts: tuple[float, float]) -> dict[str, Any]:
    """Compute window-corrected AC power and radial energy fractions.

    Subtract the Hann-weighted mean BEFORE windowing. With an orthonormal
    FFT, sum(power) equals sum(windowed_signal**2) / sum(window**2).
    Annular sums (not annular averages) preserve the total energy fraction.
    Frequencies are cycles per original pixel; no resizing or zero padding.
    """
    h, w = patch.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    centered = (patch - np.sum(patch * window) / window.sum()) * window
    power = np.abs(np.fft.fft2(centered, norm="ortho")) ** 2 / np.sum(window**2)
    power[0, 0] = 0.0
    radius = np.hypot(np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :])
    edges = np.linspace(0, MAX_FREQUENCY, bins + 1)
    total = float(power.sum())
    denominator = total if total > 1e-20 else 1.0
    radial = np.histogram(radius, bins=edges, weights=power)[0] / denominator
    band_power = np.array([power[radius < cuts[0]].sum(),
                           power[(radius >= cuts[0]) & (radius < cuts[1])].sum(),
                           power[radius >= cuts[1]].sum()])
    return {"power": power, "energy": total, "radial": radial,
            "frequency": (edges[:-1] + edges[1:]) / 2,
            "bands": band_power / denominator}


def clip_box(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int] | None:
    """Round COCO xywh outward and clip to actual image dimensions."""
    values = np.asarray(bbox, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all() or np.any(values[2:] <= 0):
        return None
    x, y, w, h = values
    x1, y1 = max(0, min(width, int(np.floor(x)))), max(0, min(height, int(np.floor(y))))
    x2, y2 = max(0, min(width, int(np.ceil(x + w)))), max(0, min(height, int(np.ceil(y + h))))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def exclusion_integral(height: int, width: int, boxes: list, margin: int) -> np.ndarray:
    """Build an integral occupancy map from ALL boxes, including ignored boxes."""
    mask = np.zeros((height, width), dtype=np.int32)
    for x1, y1, x2, y2 in boxes:
        mask[max(0, y1 - margin):min(height, y2 + margin),
             max(0, x1 - margin):min(width, x2 + margin)] = 1
    return np.pad(mask.cumsum(0, dtype=np.int64).cumsum(1), ((1, 0), (1, 0)))


def background_box(
    integral: np.ndarray, width: int, height: int, rng: np.random.Generator, tries: int
) -> tuple[int, int, int, int] | None:
    """Uniformly propose same-size rectangles and accept the first unoccupied one."""
    h, w = np.array(integral.shape) - 1
    if width > w or height > h:
        return None
    xs = rng.integers(0, w - width + 1, size=tries)
    ys = rng.integers(0, h - height + 1, size=tries)
    sums = integral[ys + height, xs + width] - integral[ys, xs + width]
    sums = sums - integral[ys + height, xs] + integral[ys, xs]
    valid = np.flatnonzero(sums == 0)
    if not len(valid):
        return None
    i = valid[0]
    return int(xs[i]), int(ys[i]), int(xs[i] + width), int(ys[i] + height)


def save_figure(fig: plt.Figure, destination: Path) -> None:
    """Export a readable PNG and an editable-text PDF, then close the figure."""
    fig.savefig(destination.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def draw_example(
    rgb: np.ndarray, boxes: list, pair: list, results: list, title: str, output: Path
) -> None:
    """Show selected source regions and spectra using matched intensity scales."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    axes[0, 0].imshow(rgb)
    for x1, y1, x2, y2 in boxes:
        axes[0, 0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                      edgecolor="0.7", linewidth=0.6))
    for box, color, label in zip(pair, COLORS, ("Object", "Background")):
        x1, y1, x2, y2 = box
        axes[0, 0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                      edgecolor=color, linewidth=2, label=label))
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title("All boxes + sampled pair")
    logpowers = [np.log10(np.fft.fftshift(r["power"]) + 1e-12) for r in results]
    vmax = max(float(p.max()) for p in logpowers)
    for index, (box, result, label, color) in enumerate(zip(pair, results, ("Object", "Background"), COLORS)):
        x1, y1, x2, y2 = box
        axes[0, index + 1].imshow(rgb[y1:y2, x1:x2], interpolation="nearest")
        axes[0, index + 1].set_title(f"{label}: {x2 - x1} x {y2 - y1} px", color=color)
        fx = np.fft.fftshift(np.fft.fftfreq(x2 - x1))
        fy = np.fft.fftshift(np.fft.fftfreq(y2 - y1))
        dx, dy = 0.5 / (x2 - x1), 0.5 / (y2 - y1)
        handle = axes[1, index].imshow(logpowers[index], cmap="magma", vmin=vmax - 6, vmax=vmax,
                                        extent=(fx[0] - dx, fx[-1] + dx, fy[0] - dy, fy[-1] + dy),
                                        origin="lower")
        axes[1, index].set(title=f"{label}: log10 AC power", xlabel="fx (cycles/pixel)", ylabel="fy (cycles/pixel)")
        axes[1, 2].plot(result["frequency"], result["radial"], color=color, label=label)
    for ax in axes[0]:
        ax.set_axis_off()
    fig.colorbar(handle, ax=list(axes[1, :2]), shrink=0.7, label="Shared log10 power scale")
    axes[1, 2].set(title="Radial energy distribution", xlabel="Radial frequency (cycles/pixel)",
                   ylabel="Fraction of AC energy / bin")
    axes[1, 2].legend()
    fig.suptitle(title, fontsize=11)
    save_figure(fig, output)


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write UTF-8 tables readable in Excel, preserving explicit column names."""
    if rows:
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def summarize(rows: list[dict], radial_pairs: list, args: argparse.Namespace) -> dict:
    """Aggregate within image, then equally across images; draw descriptive plots."""
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[row["image_id"]].append(i)
    metrics = [f"{group}_{metric}" for group in ("object", "background")
               for metric in ("energy", "low", "mid", "high")]
    per_image = []
    curves = []
    for image_id, indices in grouped.items():
        record = {"image_id": image_id, "pairs": len(indices)}
        record.update({key: float(np.mean([rows[i][key] for i in indices])) for key in metrics})
        per_image.append(record)
        curves.append(np.mean([radial_pairs[i] for i in indices], axis=0))
    curves = np.asarray(curves)
    frequency = (np.arange(args.bins) + 0.5) * MAX_FREQUENCY / args.bins
    write_csv(args.output_dir / "per_image.csv", per_image)
    spectral_rows = []
    for image_index, record in enumerate(per_image):
        for group_index, group in enumerate(("object", "background")):
            spectral_rows.extend({"image_id": record["image_id"], "region": group,
                                  "frequency_cycles_per_pixel": float(f),
                                  "energy_fraction": float(value)}
                                 for f, value in zip(frequency, curves[image_index, group_index]))
    write_csv(args.output_dir / "radial_spectra.csv", spectral_rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for i, (group, color) in enumerate(zip(("object", "background"), COLORS)):
        mean = curves[:, i].mean(0)
        sd = curves[:, i].std(0)
        axes[0, 0].plot(frequency, mean, label=group.capitalize(), color=color)
        axes[0, 0].fill_between(frequency, np.maximum(0, mean - sd), mean + sd, color=color, alpha=0.15)
        means = [np.mean([r[f"{group}_{band}"] for r in per_image]) for band in BANDS]
        axes[0, 1].bar(np.arange(3) + (i - 0.5) * 0.35, means, width=0.35, color=color, label=group.capitalize())
    axes[0, 0].set(title="Mean radial spectrum; shading = 1 SD across images",
                   xlabel="Radial frequency (cycles/pixel)", ylabel="Fraction of AC energy / bin")
    axes[0, 0].legend()
    axes[0, 1].set(title="Band fractions: images weighted equally", xticks=np.arange(3),
                   xticklabels=[f"Low\n< {args.low_cut:g}", f"Mid\n{args.low_cut:g} to {args.high_cut:g}",
                                f"High\n>= {args.high_cut:g}"], ylabel="Mean fraction of AC energy", ylim=(0, 1))
    axes[0, 1].legend()
    bg = np.array([r["background_energy"] for r in per_image])
    obj = np.array([r["object_energy"] for r in per_image])
    axes[1, 0].scatter(bg, obj, s=16, alpha=0.6, color=COLORS[0])
    limits = [min(bg.min(), obj.min()) * 0.8, max(bg.max(), obj.max()) * 1.2]
    axes[1, 0].plot(limits, limits, "--", color="0.5", linewidth=1)
    axes[1, 0].set(xscale="log", yscale="log", xlim=limits, ylim=limits,
                   title="Absolute AC energy; each dot = one image",
                   xlabel="Background energy (gray [0,1] squared)", ylabel="Object energy (gray [0,1] squared)")
    axes[1, 1].scatter([r["box_area_px"] for r in rows],
                       [r["object_high"] - r["background_high"] for r in rows],
                       s=10, alpha=0.4, color=COLORS[0])
    axes[1, 1].axhline(0, color="0.5", linestyle="--", linewidth=1)
    axes[1, 1].set(xscale="log", title="Size dependence; each dot = one matched pair",
                   xlabel="Clipped object-box area (pixels)", ylabel="High-frequency fraction: object - background")
    fig.suptitle(f"Object boxes vs unannotated background | {len(per_image)} images, {len(rows)} pairs", fontsize=13)
    save_figure(fig, args.output_dir / "summary")
    return {"images_with_pairs": len(per_image),
            "image_weighted_means": {key: float(np.mean([r[key] for r in per_image])) for key in metrics},
            "mean_paired_high_fraction_difference": float(np.mean([
                r["object_high"] - r["background_high"] for r in per_image]))}


def run(args: argparse.Namespace) -> dict:
    """Read COCO, sample matched pairs, export figures and an auditable run report."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = args.annotations or args.dataset_dir / args.split / "_annotations.coco.json"
    image_dir = args.image_dir or args.dataset_dir / args.split
    raw = annotation_path.read_bytes()
    coco = json.loads(raw)
    annotations = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    categories = {item["id"]: item["name"] for item in coco.get("categories", [])}
    rng = np.random.default_rng(args.seed)
    images = sorted(coco["images"], key=lambda item: item["id"])
    order = rng.permutation(len(images))
    if args.max_images:
        order = order[:args.max_images]
    counts = Counter(images_in_split=len(images), images_selected=len(order), pairs=0)
    rows, radial_pairs = [], []
    examples = 0
    for step, index in enumerate(order):
        info = images[index]
        path = image_dir / info["file_name"]
        with Image.open(path) as source:
            rgb = np.asarray(source.convert("RGB"))
        h, w = rgb.shape[:2]
        if (w, h) != (info["width"], info["height"]):
            raise ValueError(f"Image/COCO size mismatch for {path}: {(w, h)} vs {(info['width'], info['height'])}")
        gray = np.asarray(rgb, dtype=np.float64) @ np.array([0.299, 0.587, 0.114]) / 255.0
        targets, boxes = [], []
        for annotation in annotations[info["id"]]:
            box = clip_box(annotation["bbox"], w, h)
            if box is None:
                counts["invalid_boxes"] += 1
                continue
            boxes.append(box)
            if annotation.get("iscrowd", 0) or annotation.get("ignore", 0):
                counts["ignored_or_crowd_boxes"] += 1
                continue
            if min(box[2] - box[0], box[3] - box[1]) < args.min_size:
                counts["too_small_boxes"] += 1
                continue
            targets.append((annotation, box))
        counts["eligible_boxes"] += len(targets)
        integral = exclusion_integral(h, w, boxes, args.bg_margin)
        target_order = rng.permutation(len(targets))
        if args.max_pairs_per_image:
            target_order = target_order[:args.max_pairs_per_image]
        counts["boxes_not_sampled_due_to_cap"] += len(targets) - len(target_order)
        image_has_pair = False
        example_saved = False
        for target_index in target_order:
            annotation, box = targets[target_index]
            x1, y1, x2, y2 = box
            bg_box = background_box(integral, x2 - x1, y2 - y1, rng, args.bg_tries)
            if bg_box is None:
                counts["background_search_failed"] += 1
                continue
            pair = [box, bg_box]
            results = [spectrum(gray[y:y_end, x:x_end], args.bins, (args.low_cut, args.high_cut))
                       for x, y, x_end, y_end in pair]
            if any(result["energy"] <= 1e-20 for result in results):
                counts["zero_texture_pairs"] += 1
                continue
            row = {"image_id": info["id"], "file_name": info["file_name"],
                   "annotation_id": annotation.get("id", ""), "category_id": annotation["category_id"],
                   "category": categories.get(annotation["category_id"], str(annotation["category_id"])),
                   "box_area_px": (x2 - x1) * (y2 - y1)}
            for group, coordinates, result in zip(("object", "background"), pair, results):
                row.update({f"{group}_{name}": int(value) for name, value in zip(("x1", "y1", "x2", "y2"), coordinates)})
                row[f"{group}_energy"] = result["energy"]
                row.update({f"{group}_{band}": float(value) for band, value in zip(BANDS, result["bands"])})
            rows.append(row)
            radial_pairs.append([result["radial"] for result in results])
            counts["pairs"] += 1
            image_has_pair = True
            if examples < args.examples and not example_saved:
                examples += 1
                example_saved = True
                draw_example(rgb, boxes, pair, results,
                             f"Image {info['id']} | annotation {annotation.get('id', '?')} | {row['category']}",
                             args.output_dir / f"example_{examples:03d}")
        if not image_has_pair:
            counts["images_without_pairs"] += 1
        if (step + 1) % 25 == 0 or step + 1 == len(order):
            print(f"Processed {step + 1}/{len(order)} images; {counts['pairs']} pairs", flush=True)
    report = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "annotation_path": str(annotation_path.resolve()), "image_dir": str(image_dir.resolve()),
        "annotation_sha256": hashlib.sha256(raw).hexdigest(), "counts": dict(counts),
        "method": "Native-size matched rectangles; grayscale [0,1]; Hann-weighted mean removal; Hann window; "
                  "orthonormal FFT; window-energy correction; annular power sums; equal image weighting.",
        "frequency_unit": "cycles per original pixel; radial range [0, sqrt(0.5)]",
        "limits": ["Object boxes include background; background means outside annotations, not guaranteed object-free.",
                   "Only successfully matched, nonconstant pairs are analyzed; small and crowded objects may be underrepresented.",
                   "Same-image backgrounds need not match depth, illumination, texture, or haze density; patches can overlap.",
                   "Band cutoffs are exploratory; fractions are not detection accuracy or evidence of haze causation.",
                   "Radial bins sum energy, not average power density; finite box sizes affect frequency resolution."],
    }
    if rows:
        write_csv(args.output_dir / "pairs.csv", rows)
        report.update(summarize(rows, radial_pairs, args))
    (args.output_dir / "run_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not rows:
        raise ValueError("No valid pairs. See run_summary.json; check annotations, --min-size, --bg-margin and --bg-tries.")
    print(f"Saved to {args.output_dir.resolve()}")
    print(f"Image-weighted high-frequency fraction difference (object - background): "
          f"{report['mean_paired_high_fraction_difference']:+.4f}")
    return report


def parse_args() -> argparse.Namespace:
    """Expose dataset layout, sampling, and physically defined frequency cutoffs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("/root/autodl-tmp/HazyDet_RFDETR"))
    parser.add_argument("--split", default="valid")
    parser.add_argument("--annotations", type=Path, help="Override COCO JSON path")
    parser.add_argument("--image-dir", type=Path, help="Base path for COCO file_name values")
    parser.add_argument("--output-dir", type=Path, default=Path("output/hazydet_frequency"))
    parser.add_argument("--max-images", type=int, default=200, help="Randomly selected images; 0 = all")
    parser.add_argument("--max-pairs-per-image", type=int, default=10, help="Maximum attempted target boxes; 0 = all")
    parser.add_argument("--min-size", type=int, default=16, help="Minimum clipped box side, in original pixels")
    parser.add_argument("--bg-margin", type=int, default=4, help="Exclude this margin around every annotation")
    parser.add_argument("--bg-tries", type=int, default=500, help="Random background proposals per object")
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--low-cut", type=float, default=0.08, help="Low/mid cutoff, cycles/pixel")
    parser.add_argument("--high-cut", type=float, default=0.20, help="Mid/high cutoff, cycles/pixel")
    parser.add_argument("--examples", type=int, default=6, help="At most one illustrated pair per image")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.max_images, args.max_pairs_per_image, args.bg_margin, args.examples, args.seed) < 0:
        parser.error("Counts, margin and seed must be nonnegative")
    if args.min_size < 4 or args.bins < 4 or args.bg_tries < 1:
        parser.error("Require min-size >= 4, bins >= 4, bg-tries >= 1")
    if not 0 < args.low_cut < args.high_cut < MAX_FREQUENCY:
        parser.error("Require 0 < low-cut < high-cut < sqrt(0.5)")
    return args


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    run(parse_args())
