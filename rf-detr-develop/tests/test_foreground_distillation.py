# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Regression tests for clear/hazy pairing and foreground-only teacher gradients."""

import json

import albumentations as alb
import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision.transforms.v2 import Compose, ToDtype, ToImage

from rfdetr.config import TrainConfig
from rfdetr.datasets.clear_pairs import ClearImagePairs
from rfdetr.datasets.coco import CocoDetection
from rfdetr.datasets.transforms import AlbumentationsWrapper, Normalize
from rfdetr.models.foreground_distillation import (
    foreground_distillation_loss,
    load_clear_teacher,
    pack_clear_images,
)
from rfdetr.utilities.tensors import NestedTensor


def test_configuration_requires_explicit_teacher():
    """Disabled defaults stay compatible; enabled KD requires complete supported inputs."""
    assert TrainConfig(dataset_dir="example").fg_distill_coef == 0
    with pytest.raises(ValueError, match="fg_teacher_weights"):
        TrainConfig(dataset_dir="example", fg_distill_coef=0.1)
    config = dict(dataset_dir="example", fg_distill_coef=0.1, fg_teacher_weights="teacher.pth", fg_clear_dir="clear")
    assert TrainConfig(**config).fg_distill_warmup_epochs == 3
    with pytest.raises(ValueError, match="augmentation_backend"):
        TrainConfig(**config, augmentation_backend="auto")


def test_object_equal_weight_and_padding_exclusion():
    """A small and large object each contribute one sample regardless of area."""
    source = torch.zeros(1, 2, 32, 32, requires_grad=True)
    with torch.no_grad():
        source[:, 0] = 1
    reference = source.detach().clone()
    reference[:, 0, :8, :8] = 0
    reference[:, 1, :8, :8] = 1
    mask = torch.ones(1, 32, 32, dtype=torch.bool)
    mask[:, :24, :24] = False
    boxes = torch.tensor([[4 / 24, 4 / 24, 2 / 24, 2 / 24], [16 / 24, 16 / 24, 8 / 24, 8 / 24]])
    loss = foreground_distillation_loss(
        [NestedTensor(source, mask)], [NestedTensor(reference, mask)], [{"boxes": boxes}]
    )
    torch.testing.assert_close(loss, torch.tensor(0.5))


def test_roi_loss_gradients_and_empty_targets():
    """Loss trains only student foreground and handles crops containing no objects."""
    source = torch.randn(1, 8, 20, 20, requires_grad=True)
    reference = torch.randn_like(source, requires_grad=True)
    mask = torch.zeros(1, 20, 20, dtype=torch.bool)
    student, teacher = [NestedTensor(source, mask)], [NestedTensor(reference, mask)]
    targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}]
    loss = foreground_distillation_loss(student, teacher, targets)
    loss.backward()
    assert torch.isfinite(loss) and loss > 0
    assert reference.grad is None
    assert source.grad.abs().sum() > 0
    assert source.grad[:, :, :4].abs().sum() == 0
    same = foreground_distillation_loss(student, student, targets)
    assert same == 0
    source.grad = None
    empty = foreground_distillation_loss(student, teacher, [{"boxes": torch.empty(0, 4)}])
    empty.backward()
    assert empty == 0 and source.grad.abs().sum() == 0


@pytest.mark.parametrize("prefix", ["backbone.", "model.backbone.", "module.backbone.", "model._orig_mod.backbone."])
def test_teacher_strict_frozen_loading(tmp_path, prefix):
    """Native and Lightning prefixes load fully; partial teachers fail immediately."""
    backbone = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
    state = {prefix + key: value.clone() for key, value in backbone.state_dict().items()}
    checkpoint = tmp_path / "teacher.pth"
    torch.save({"state_dict": state}, checkpoint)
    teacher = load_clear_teacher(backbone, str(checkpoint))
    assert teacher is not backbone and not teacher.training
    assert not any(parameter.requires_grad for parameter in teacher.parameters())
    del state[prefix + "0.weight"]
    torch.save({"model": state}, checkpoint)
    with pytest.raises(ValueError, match="exactly match"):
        load_clear_teacher(backbone, str(checkpoint))


@pytest.mark.parametrize("box_count", [0, 1, 3])
def test_paired_flip_crop_resize_and_normalization(box_count):
    """All geometry is shared, including empty crops and three-object images."""
    image = Image.fromarray(np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3))
    transform = Compose(
        [
            AlbumentationsWrapper(alb.HorizontalFlip(p=1)),
            AlbumentationsWrapper(alb.RandomCrop(height=18, width=22, p=1)),
            AlbumentationsWrapper(alb.Resize(height=16, width=20, p=1)),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(),
        ]
    )
    target = {
        "boxes": torch.tensor([[5.0, 5.0, 20.0, 18.0]]).repeat(box_count, 1),
        "labels": torch.zeros(box_count, dtype=torch.long),
        "_clear_image": image.copy(),
    }
    output, target = transform(image, target)
    assert "_clear_image" not in target
    torch.testing.assert_close(output, target["clear_image"])
    assert target["clear_image"].shape == (3, 16, 20)


def test_student_only_photometric_augmentation():
    """Pixel-only corruption must not contaminate the clear reference."""
    image = Image.fromarray(np.full((12, 12, 3), 80, dtype=np.uint8))
    transform = Compose(
        [
            AlbumentationsWrapper(alb.RandomBrightnessContrast(brightness_limit=(0.2, 0.2), contrast_limit=0, p=1)),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(),
        ]
    )
    target = {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long), "_clear_image": image.copy()}
    output, target = transform(image, target)
    assert not torch.allclose(output, target["clear_image"])


def test_pair_manifest_and_size_checks(tmp_path):
    """Renamed COCO exports need an explicit mapping; metadata mismatch is rejected."""
    Image.new("RGB", (32, 24)).save(tmp_path / "clear.png")
    Image.new("RGB", (32, 24)).save(tmp_path / "fog.png")
    manifest = tmp_path / "pairs.json"
    manifest.write_text(json.dumps({"export.png": {"clear": "clear.png", "hazy": "fog.png"}}))
    pairs = ClearImagePairs(["export.png"], str(tmp_path), str(tmp_path), str(manifest))
    hazy, clear = pairs.load("export.png", (32, 24))
    assert hazy.size == clear.size
    with pytest.raises(ValueError, match="size mismatch"):
        pairs.load("export.png", (64, 48))
    with pytest.raises(FileNotFoundError):
        ClearImagePairs(["missing.png"], str(tmp_path), str(tmp_path))


def test_coco_dataset_pairing(tmp_path):
    """Pair loading, annotations, v2 transforms, and student padding compose end to end."""
    Image.new("RGB", (32, 24), (80, 80, 80)).save(tmp_path / "image.png")
    annotations = {
        "images": [{"id": 1, "file_name": "image.png", "width": 32, "height": 24}],
        "categories": [{"id": 1, "name": "car"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 8, 8, 6], "area": 48, "iscrowd": 0}],
    }
    annotation_file = tmp_path / "annotations.json"
    annotation_file.write_text(json.dumps(annotations))
    transform = Compose([ToImage(), ToDtype(torch.float32, scale=True), Normalize()])
    dataset = CocoDetection(tmp_path, annotation_file, transform)
    _, normal_target = dataset[0]
    assert "clear_image" not in normal_target
    dataset.clear_pairs = ClearImagePairs(["image.png"], str(tmp_path), str(tmp_path))
    image, target = dataset[0]
    torch.testing.assert_close(image, target["clear_image"])
    samples = NestedTensor(torch.zeros(1, 3, 32, 32), torch.ones(1, 32, 32, dtype=torch.bool))
    samples.mask[:, :24] = False
    packed = pack_clear_images(samples, [target])
    torch.testing.assert_close(packed.tensors[0, :, :24], image)
    assert packed.tensors[:, :, 24:].abs().sum() == 0
    assert packed.mask is samples.mask
