from rfdetr import RFDETRSmall

from visualize_hazydet import generate_representative_visualization


DATASET_DIR = "/root/autodl-tmp/HazyDet_RFDETR"
OUTPUT_DIR = "/root/autodl-tmp/rf-detr/output/hazydet_small_hbs_frequency"
VISUALIZE_AFTER_TRAINING = True
FIXED_SAMPLE_FILE = "/root/autodl-tmp/rf-detr/output/hazydet_fixed_samples_valid.json"


def main():

    # model = RFDETRSmall()
    model = RFDETRSmall(
        hbs_enabled=True,
        hbs_reduction=4,
        foreground_frequency_enabled=True,
        foreground_frequency_reduction=4,
    )

    model.train(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,

        epochs=36,

        batch_size=4,
        grad_accum_steps=4,

        lr=1e-3,

        hbs_loss_coef=0.25,
        foreground_frequency_loss_coef=0.1,

        device="cuda",

        num_workers=8,

        use_ema=True,

        checkpoint_interval=5,

        early_stopping=False,
    )

    if VISUALIZE_AFTER_TRAINING:
        generate_representative_visualization(
            dataset_dir=DATASET_DIR,
            checkpoint=f"{OUTPUT_DIR}/checkpoint_best_total.pth",
            output_dir=f"{OUTPUT_DIR}/representative_visualization",
            split="valid",
            confidence_threshold=0.30,
            iou_threshold=0.50,
            sample_file=FIXED_SAMPLE_FILE,
        )


if __name__ == "__main__":
    main()
