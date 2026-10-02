import argparse

from rfdetr import RFDETRSmall
from visualize_hazydet import generate_representative_visualization

DATASET_DIR = "/root/autodl-tmp/HazyDet_RFDETR"
OUTPUT_DIR = "/root/autodl-tmp/rf-detr/output/hazydet_small_hbs_fgkd"
VISUALIZE_AFTER_TRAINING = True
SAMPLES_PER_GROUP = 3
FIXED_SAMPLE_FILE = "/root/autodl-tmp/rf-detr/output/hazydet_fixed_samples_valid_3_per_group.json"


def main() -> None:
    """Train HBS with optional paired clear-image foreground supervision."""
    parser = argparse.ArgumentParser(description="HBS + paired clear foreground distillation")
    parser.add_argument("--teacher-weights", help="Clear-trained RF-DETR Small checkpoint (required for KD)")
    parser.add_argument("--clear-dir", default="/root/autodl-tmp/HazyDet/train/images")
    parser.add_argument("--hazy-dir", default="/root/autodl-tmp/HazyDet/train/hazy_images")
    parser.add_argument("--pair-manifest", default=None, help="Optional COCO filename -> clear/hazy paths JSON")
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--fg-distill-coef", type=float, default=0.1)
    args = parser.parse_args()
    if args.fg_distill_coef > 0 and not args.teacher_weights:
        parser.error("--teacher-weights is required when --fg-distill-coef > 0")

    # model = RFDETRSmall()
    model = RFDETRSmall(
        hbs_enabled=True,
        hbs_reduction=4,
    )

    model.train(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,

        epochs=36,

        batch_size=4,
        grad_accum_steps=4,

        lr=1e-3,

        hbs_loss_coef=0.25,
        fg_distill_coef=args.fg_distill_coef,
        fg_teacher_weights=args.teacher_weights,
        fg_clear_dir=args.clear_dir,
        fg_hazy_dir=args.hazy_dir,
        fg_pair_manifest=args.pair_manifest,
        fg_roi_size=3,
        fg_distill_warmup_epochs=3,
        augmentation_backend="cpu",

        device="cuda",

        num_workers=8,

        use_ema=True,

        checkpoint_interval=5,

        early_stopping=False,
    )

    if VISUALIZE_AFTER_TRAINING:
        generate_representative_visualization(
            dataset_dir=args.dataset_dir,
            checkpoint=f"{args.output_dir}/checkpoint_best_total.pth",
            output_dir=f"{args.output_dir}/representative_visualization",
            split="valid",
            confidence_threshold=0.30,
            iou_threshold=0.50,
            sample_file=FIXED_SAMPLE_FILE,
            samples_per_group=SAMPLES_PER_GROUP,
        )


if __name__ == "__main__":
    main()
