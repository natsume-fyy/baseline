# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Detection and segmentation head subpackage."""

from rfdetr.models.heads.cbam import CBAM
from rfdetr.models.heads.keypoints import ConditionalQueryInitializer
from rfdetr.models.heads.segmentation import DepthwiseConvBlock, MLPBlock, SegmentationHead

__all__ = [
    "CBAM",
    "SegmentationHead",
    "DepthwiseConvBlock",
    "MLPBlock",
    "ConditionalQueryInitializer",
]
