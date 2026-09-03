# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Fine-tune RF-DETR Small with UFFR on HazyDet."""

from rfdetr import RFDETRSmall


DATASET_DIR = "/root/autodl-tmp/HazyDet_RFDETR"
OUTPUT_DIR = "/root/autodl-tmp/rf-detr/output/hazydet_small_uffr_v2"


def main() -> None:
    """Build the UFFR-enhanced detector and start fine-tuning."""
    model = RFDETRSmall(
        uffr=True,
        uffr_freq_radius=0.6,
        uffr_alpha_low=0.95,
        uffr_alpha_high=1.55,
        uffr_learnable=True,
        uffr_feature_indexes=[1, 2, 3],
    )

    model.train(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        epochs=36,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        device="cuda",
        num_workers=8,
        use_ema=True,
        checkpoint_interval=5,
        early_stopping=False,
        notes={
            "method": "UFFR",
            "source": "doi:10.3389/fmars.2026.1885615",
            "domain": "foggy-object-detection",
        },
    )


if __name__ == "__main__":
    main()
