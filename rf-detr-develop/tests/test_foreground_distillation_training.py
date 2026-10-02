# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Exercise the real training hooks with small deterministic stand-in networks."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.utilities.tensors import NestedTensor


class TinyStudent(nn.Module):
    """Expose projected features alongside ordinary and HBS detection losses."""

    def __init__(self):
        super().__init__()
        self.projector = nn.Conv2d(3, 4, 1)

    def forward(self, samples, targets, return_features=False):
        features = self.projector(samples.tensors)
        output = {"pred_logits": features.mean(), "hbs_outputs": features.square().mean()}
        if return_features:
            output["distill_features"] = [NestedTensor(features, samples.mask)]
        return output


class TinyTeacher(nn.Module):
    """BatchNorm makes accidental training mode observable."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
        self.requires_grad_(False)

    def forward(self, samples):
        assert not self.training and not torch.is_grad_enabled()
        return [NestedTensor(self.layers(samples.tensors), samples.mask)], [], None


class TinyCriterion:
    """Check feature outputs are removed before matching and preserve HBS loss."""

    weight_dict = {"loss_ce": 1.0, "loss_ce_hbs": 0.25}

    def __call__(self, output, targets):
        assert "distill_features" not in output
        assert all("clear_image" not in target for target in targets)
        return {"loss_ce": output["pred_logits"], "loss_ce_hbs": output["hbs_outputs"]}


class TrainingHarness(nn.Module):
    """Bind production hooks without constructing a downloaded DINO backbone."""

    on_train_batch_start = RFDETRModelModule.on_train_batch_start
    training_step = RFDETRModelModule.training_step

    def __init__(self, enabled):
        super().__init__()
        self.model = TinyStudent()
        self.fg_teacher = TinyTeacher() if enabled else None
        self.criterion = TinyCriterion()
        self.train_config = SimpleNamespace(
            multi_scale=True,
            do_random_resize_via_padding=False,
            expanded_scales=False,
            fg_roi_size=3,
            fg_distill_coef=0.1,
            fg_distill_warmup_epochs=3,
            train_log_sync_dist=False,
            train_log_on_step=False,
            compute_train_metrics=False,
        )
        self.model_config = SimpleNamespace(resolution=32, patch_size=4, num_windows=1)
        self.trainer = SimpleNamespace(global_step=0, accumulate_grad_batches=2)
        self.current_epoch = 0
        self._use_manual_optimization = False
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.logged = {}

    def optimizers(self):
        return self.optimizer

    def log_dict(self, values, **kwargs):
        self.logged.update(values)

    def log(self, key, value, **kwargs):
        self.logged[key] = value

    def _log_train_progress_metrics(self, *args, **kwargs):
        pass


@pytest.mark.parametrize("enabled", [False, True])
def test_training_hooks_multiscale_loss_and_gradients(enabled):
    """Both paths work; clear/fog resize matches and KD adds the scheduled loss."""
    harness = TrainingHarness(enabled).train()
    image = torch.randn(3, 24, 32)
    canvas = torch.zeros(1, 3, 32, 32)
    canvas[0, :, :24] = image
    mask = torch.ones(1, 32, 32, dtype=torch.bool)
    mask[:, :24] = False
    target = {"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}
    if enabled:
        target["clear_image"] = image.clone()
    batch = (NestedTensor(canvas, mask), [target])
    harness.on_train_batch_start(batch, 0)
    if enabled:
        torch.testing.assert_close(batch[0].tensors[0], target["clear_image"])
    loss = harness.training_step(batch, 0)
    loss.backward()
    assert harness.model.projector.weight.grad.abs().sum() > 0
    expected = harness.logged["train/loss_ce"] + 0.25 * harness.logged["train/loss_ce_hbs"]
    if enabled:
        assert all(parameter.grad is None for parameter in harness.fg_teacher.parameters())
        assert not harness.fg_teacher.training
        kd = harness.logged["train/loss_fg_distill"]
        expected = expected + kd * (0.1 / 3)
        torch.testing.assert_close(harness.logged["train/loss_fg_distill_weighted"], kd * (0.1 / 3))
    else:
        assert "train/loss_fg_distill" not in harness.logged
    torch.testing.assert_close(loss * 2, expected)
