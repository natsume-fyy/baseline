# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Training-only paired clear-image foreground feature distillation."""

from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812 -- project-conventional alias
from torch import nn
from torchvision.ops import roi_align

from rfdetr.utilities.tensors import NestedTensor


def load_clear_teacher(backbone: nn.Module, checkpoint_path: str) -> nn.Module:
    """Load all teacher backbone/projector weights strictly from a trusted detector checkpoint.

    Args:
        backbone: Student backbone defining the required teacher architecture.
        checkpoint_path: Clear-trained RF-DETR .pth or Lightning .ckpt checkpoint.

    Returns:
        Frozen backbone in evaluation mode. Detection heads and HBS are not copied.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Clear teacher checkpoint not found: {path}")
    # Native RF-DETR checkpoints also contain argparse metadata; same trusted
    # checkpoint convention as models/weights.py.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    weights = {}
    for key, value in state.items():
        while key.startswith(("module.", "model.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        if key.startswith("backbone."):
            weights[key[len("backbone.") :]] = value
    teacher = deepcopy(backbone)
    try:
        teacher.load_state_dict(weights, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "Clear teacher backbone/projector must exactly match the student architecture. "
            "Use a clear-trained RF-DETR Small checkpoint with the same resolution and feature levels."
        ) from exc
    return teacher.requires_grad_(False).eval()


def pack_clear_images(samples: NestedTensor, targets: list[dict]) -> NestedTensor:
    """Pad normalized clear images to exactly the student's canvas and padding mask."""
    clear = torch.zeros_like(samples.tensors)
    for index, target in enumerate(targets):
        image = target["clear_image"]
        height, width = image.shape[-2:]
        if height > clear.shape[-2] or width > clear.shape[-1]:
            raise ValueError("Clear image exceeds student canvas; paired geometry is not synchronized.")
        clear[index, :, :height, :width] = image
    return NestedTensor(clear, samples.mask)


def foreground_distillation_loss(
    student: list[NestedTensor],
    teacher: list[NestedTensor],
    targets: list[dict],
    roi_size: int = 3,
) -> torch.Tensor:
    """Average channel-normalized ROI feature error equally over objects and feature levels.

    Args:
        student: Projected student features before HBS, retaining gradients.
        teacher: Corresponding clear teacher features (detached inside this loss).
        targets: Augmented normalized cxcywh boxes, excluding padded image area.
        roi_size: Output height and width for each ROIAlign crop.

    Returns:
        Scalar loss, or a differentiable zero when no valid foreground remains.
    """
    if not student or len(student) != len(teacher):
        raise ValueError("Teacher and student must have matching nonempty feature levels.")
    losses = []
    for source, reference in zip(student, teacher):
        if source.tensors.shape != reference.tensors.shape:
            raise ValueError("Teacher and student feature shapes differ.")
        rois = []
        for index, target in enumerate(targets):
            boxes = target["boxes"].detach().to(device=source.tensors.device, dtype=torch.float32)
            xyxy = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), dim=1)
            xyxy = xyxy.clamp(0, 1)
            xyxy = xyxy[(xyxy[:, 2:] > xyxy[:, :2]).all(dim=1)]
            height, width = source.tensors.shape[-2:]
            if source.mask is not None:
                valid = ~source.mask[index]
                height = valid.any(dim=1).sum()
                width = valid.any(dim=0).sum()
            scale = torch.stack([torch.as_tensor(v, device=boxes.device) for v in (width, height, width, height)])
            rois.append(torch.cat((xyxy.new_full((len(xyxy), 1), index), xyxy * scale), dim=1))
        rois = torch.cat(rois)
        if not len(rois):
            losses.append(source.tensors.float().sum() * 0.0)
            continue
        # FP32 avoids underflow when normalizing faint tiny-object activations under AMP.
        source_roi = roi_align(source.tensors.float(), rois, roi_size, sampling_ratio=2, aligned=True)
        teacher_roi = roi_align(reference.tensors.detach().float(), rois, roi_size, sampling_ratio=2, aligned=True)
        difference = F.normalize(source_roi, dim=1) - F.normalize(teacher_roi, dim=1)
        losses.append(difference.square().sum(dim=1).mean() / 2)
    return torch.stack(losses).mean()
