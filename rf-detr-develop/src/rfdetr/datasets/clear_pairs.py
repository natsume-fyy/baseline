# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Explicit clear/hazy pair indexing for COCO foreground distillation."""

import json
from pathlib import Path

from PIL import Image


class ClearImagePairs:
    """Resolve COCO filenames to paired files, checking all paths before training.

    A JSON manifest maps each exact COCO ``file_name`` to an object with ``clear``
    and ``hazy`` relative paths. Without a manifest both use the COCO filename.
    No basename stripping or guessed filename replacement is performed.
    """

    def __init__(self, filenames: list[str], clear_dir: str, hazy_dir: str, manifest: str | None = None) -> None:
        mapping = json.loads(Path(manifest).read_text(encoding="utf-8")) if manifest else None
        if mapping is not None and not isinstance(mapping, dict):
            raise ValueError("Pair manifest must be a JSON object keyed by COCO file_name.")
        self.paths: dict[str, tuple[Path, Path]] = {}
        for filename in filenames:
            if mapping is not None and filename not in mapping:
                raise ValueError(f"Pair manifest has no entry for COCO file_name: {filename}")
            pair = mapping[filename] if mapping is not None else {"clear": filename, "hazy": filename}
            if not isinstance(pair, dict) or not all(isinstance(pair.get(key), str) for key in ("clear", "hazy")):
                raise ValueError(f"Pair entry for {filename} requires string 'clear' and 'hazy' paths.")
            paths = (Path(clear_dir) / pair["clear"], Path(hazy_dir) / pair["hazy"])
            for path in paths:
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing paired image: {path}. Set fg_pair_manifest if COCO and source names differ."
                    )
            self.paths[filename] = paths

    def load(self, filename: str, annotation_size: tuple[int, int]) -> tuple[Image.Image, Image.Image]:
        """Load RGB images and reject geometric mismatch with each other or COCO metadata."""
        clear_path, hazy_path = self.paths[filename]
        with Image.open(clear_path) as image:
            clear = image.convert("RGB")
        with Image.open(hazy_path) as image:
            hazy = image.convert("RGB")
        if clear.size != hazy.size or hazy.size != annotation_size:
            raise ValueError(
                f"Pair/annotation size mismatch for {filename}: clear={clear.size}, "
                f"hazy={hazy.size}, COCO={annotation_size}. Use annotations for the original paired images."
            )
        return hazy, clear
