# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Reproducible random sampling and descriptive Projector diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def select_images(sources: dict[str, Path], count: int, seed: int) -> list[dict[str, str]]:
    """Sample without replacement, balancing known weather directories when supplied."""
    if count < len(sources) or not sources:
        raise ValueError("Sample count must be positive and cover every source group.")
    rng = random.Random(seed)
    result = []
    seen: set[Path] = set()
    for index, (group, directory) in enumerate(sources.items()):
        if not directory.is_dir():
            raise FileNotFoundError(f"Image directory not found: {directory}")
        candidates = sorted({p.resolve() for p in directory.rglob("*")
                             if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}})
        if seen.intersection(candidates):
            raise ValueError("Weather directories overlap; each image must have only one group.")
        seen.update(candidates)
        needed = count // len(sources) + (index < count % len(sources))
        if len(candidates) < needed:
            raise ValueError(f"{directory}: need {needed} distinct images, found {len(candidates)}")
        for path in rng.sample(candidates, needed):
            result.append({"path": str(path), "group": group})
    rng.shuffle(result)
    return result


def image_contrast(image: Image.Image) -> float:
    """Measure image contrast as an appearance descriptor, not a weather label."""
    gray = np.asarray(image.convert("L").resize((256, 256)), dtype=np.float64) / 255
    return float(gray.std())


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-image descriptors within each layer/weather group."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["level"], row["group"])].append(row)
    summary = []
    for (level, group), items in sorted(groups.items()):
        record: dict[str, Any] = {"level": level, "group": group, "n": len(items)}
        for metric in ("rms_mean", "spatial_cv"):
            values = np.array([item[metric] for item in items])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_median"] = float(np.median(values))
            record[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
        contrast = np.array([item["image_contrast"] for item in items])
        energy = np.array([item["rms_mean"] for item in items])
        record["contrast_energy_pearson"] = (
            float(np.corrcoef(contrast, energy)[0, 1])
            if len(items) >= 3 and contrast.std() > 1e-12 and energy.std() > 1e-12 else None
        )
        summary.append(record)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Persist heterogeneous records with a stable column union."""
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def write_analysis(output: Path, rows: list[dict[str, Any]], count: int) -> None:
    """Generate data-based Chinese descriptive findings without causal claims."""
    summary = summarize(rows)
    write_csv(output / "summary.csv", summary)
    lines = ["# Projector 随机样本分析", "", f"共分析 {count} 张独立抽样图片；统计单位为图片，不是像素或通道。",
             "HBS 推理分支关闭；使用 HBS 训练过的权重仍保留其训练影响。", "",
             "## 各层描述统计", "", "|层|分组|图片数|RMS 均值|空间变异系数均值|图像对比度与能量 Pearson r|",
             "|---|---|---:|---:|---:|---:|"]
    for item in summary:
        corr = item["contrast_energy_pearson"]
        corr_text = f"{corr:.3f}" if corr is not None else "不可计算"
        lines.append(f"|{item['level']}|{item['group']}|{item['n']}|{item['rms_mean_mean']:.4f}|"
                     f"{item['spatial_cv_mean']:.4f}|{corr_text}|")
    lines += ["", "## 结果解读", ""]
    lookup = {(item["level"], item["group"]): item for item in summary}
    if any(item["group"] == "unknown" for item in summary):
        lines.append("输入为混合目录，没有可靠天气标签，因此本报告不声称进行了清晰/雾天分组比较。"
                     "图像对比度仅是外观指标，不能用低对比度自动认定雾天。可用 --clear-dir 与 --haze-dir 提供真实分组。")
    else:
        for level in sorted({item["level"] for item in summary}):
            clear, haze = lookup[(level, "clear")], lookup[(level, "haze")]
            base = clear["rms_mean_mean"]
            change = f"{(haze['rms_mean_mean'] / base - 1) * 100:+.1f}%" if base > 1e-12 else "不可计算"
            lines.append(f"- {level}：雾天相对清晰组的平均 RMS 变化为 {change}。这是未配对组间差异，"
                         "也可能受到场景、目标密度及光照影响；不等于准确率变化。")
    lines += ["", "空间变异系数 = 特征能量图空间标准差 / 均值；越小只说明能量分布越均匀，不能直接推断目标丢失。",
              "所有样本在同一层使用同一色标。不同 pre/post 层是不同通道空间，禁止直接作逐元素差或将能量大小解释为信息保留率。",
              "", "## 下一步定位", "",
              "1. 查看 images/ 中目标位置在 pre 层是否已缺乏可见结构，结合输入缩放后目标像素数检查编码器与输入分辨率。",
              "2. 如果 pre 层仍有结构而 post 层变弱，将 Projector 列为待验证因素；必须用目标/背景标注验证，不能凭热图确定瓶颈。",
              "3. 如果 post 层仍有结构但检测漏检，继续核查 decoder 查询、分类阈值与定位。",
              "4. 本脚本未计算标注匹配、AP 或召回，也未做显著性检验；要判断检测缺陷，需在固定验证集按天气和目标尺寸评估并做消融。",
              "", "查看 samples.json 复核样本，statistics.csv 复核逐图统计，summary.png 查看分布。"]
    (output / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    levels = list(dict.fromkeys(row["level"] for row in rows))
    groups = sorted({row["group"] for row in rows})
    fig, axes = plt.subplots(2, len(levels), squeeze=False, figsize=(3.5 * len(levels), 7), constrained_layout=True)
    try:
        for column, level in enumerate(levels):
            for row_index, metric in enumerate(("rms_mean", "spatial_cv")):
                values = [[row[metric] for row in rows if row["level"] == level and row["group"] == group]
                          for group in groups]
                axes[row_index, column].boxplot(values)
                axes[row_index, column].set_xticks(range(1, len(groups) + 1), groups)
                axes[row_index, column].set_title(f"{level}: {metric}")
        fig.savefig(output / "summary.png", dpi=150)
    finally:
        plt.close(fig)


def render_sample(image: Image.Image, maps: dict[str, np.ndarray], limits: dict[str, float], path: Path) -> None:
    """Render a sample using per-layer scales shared by the entire random sample."""
    fig, axes = plt.subplots(1, len(maps) + 1, figsize=(3.3 * (len(maps) + 1), 3.6), constrained_layout=True)
    try:
        axes[0].imshow(image)
        axes[0].set_title("Original")
        for axis, (level, heatmap) in zip(axes[1:], maps.items()):
            axis.imshow(image)
            artist = axis.imshow(heatmap, extent=(0, image.width, image.height, 0), alpha=0.65,
                                 cmap="magma", vmin=0, vmax=limits[level], interpolation="nearest")
            axis.set_title(f"{level} | H,W={heatmap.shape}")
            fig.colorbar(artist, ax=axis, shrink=0.65, label="RMS")
        for axis in axes:
            axis.axis("off")
        fig.savefig(path, dpi=140)
    finally:
        plt.close(fig)


def run_batch(args: argparse.Namespace) -> None:
    """Process sampled images sequentially and report measured feature distributions."""
    import torch

    from visualize_projector import activation_map, capture_features

    if args.image_dir:
        if args.clear_dir or args.haze_dir:
            raise ValueError("Use --image-dir OR both --clear-dir/--haze-dir.")
        sources = {"unknown": args.image_dir}
    elif args.clear_dir and args.haze_dir:
        sources = {"clear": args.clear_dir, "haze": args.haze_dir}
    else:
        raise ValueError("Both --clear-dir and --haze-dir are required.")
    samples = select_images(sources, args.num_samples, args.seed)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not hasattr(torch, "inference_mode"):
        raise RuntimeError("Run in your RF-DETR training environment with a working PyTorch installation.")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory to avoid mixing experiments or sampling previous outputs.")
    if any(output == path.resolve() or path.resolve() in output.parents for path in sources.values()):
        raise ValueError("Output directory must be outside the source image directories.")
    output.mkdir(parents=True, exist_ok=True)
    for folder in ("maps", "images"):
        (output / folder).mkdir(exist_ok=True)
    for sample in samples:
        sample["sha256"] = hashlib.sha256(Path(sample["path"]).read_bytes()).hexdigest()
    (output / "samples.json").write_text(json.dumps({"seed": args.seed, "count": len(samples), "samples": samples},
                                                   ensure_ascii=False, indent=2), encoding="utf-8")
    from rfdetr import from_checkpoint

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = from_checkpoint(args.checkpoint, device=device, hbs_enabled=False)
    model.model.model.eval()
    rows = []
    limits: dict[str, float] = {}
    for index, sample in enumerate(samples):
        with Image.open(sample["path"]) as source:
            image = source.convert("RGB")
        captures = capture_features(model, image)
        maps = {level: activation_map(feature) for level, feature in captures.items()}
        if limits and maps.keys() != limits.keys():
            raise ValueError("Projector levels changed between samples.")
        contrast = image_contrast(image)
        for level, heatmap in maps.items():
            limits[level] = max(limits.get(level, 0), float(heatmap.max()), 1e-12)
            rows.append({"sample": index, "path": sample["path"], "group": sample["group"], "level": level,
                         "shape": str(captures[level].shape), "image_contrast": contrast,
                         "rms_mean": float(heatmap.mean()),
                         "spatial_cv": float(heatmap.std() / max(float(heatmap.mean()), 1e-12))})
        np.savez_compressed(output / "maps" / f"{index:04d}.npz", **maps)
        if args.save_raw:
            (output / "raw").mkdir(exist_ok=True)
            np.savez_compressed(output / "raw" / f"{index:04d}.npz", **captures)
        del captures, maps
        print(f"Extracted {index + 1}/{len(samples)}", flush=True)
    for index, sample in enumerate(samples):
        with Image.open(sample["path"]) as source, np.load(output / "maps" / f"{index:04d}.npz") as data:
            render_sample(source.convert("RGB"), dict(data), limits, output / "images" / f"{index:04d}.png")
        if (index + 1) % 10 == 0:
            print(f"Rendered {index + 1}/{len(samples)}", flush=True)
    write_csv(output / "statistics.csv", rows)
    write_analysis(output, rows, len(samples))
    metadata = {"checkpoint": str(args.checkpoint.resolve()), "device": device, "seed": args.seed,
                "count": len(samples), "hbs_enabled": False, "color_vmax": limits,
                "resolution": model.model.resolution, "model_class": type(model).__name__,
                "encoder_feature_indexes": model.model_config.out_feature_indexes,
                "projector_scales": model.model.model.backbone[0].projector_scale}
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Completed. Analysis: {output / 'analysis.md'}", flush=True)
