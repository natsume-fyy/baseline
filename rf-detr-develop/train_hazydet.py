from rfdetr import RFDETRSmall


DATASET_DIR = "/root/autodl-tmp/HazyDet_RFDETR"
OUTPUT_DIR = "/root/autodl-tmp/rf-detr/output/hazydet_small_hbs"


def main():

    # model = RFDETRSmall()
    model = RFDETRSmall(
        hbs_enabled=True,
        hbs_reduction=4,
        hbs_alpha_init=0.1,
    )

    model.train(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,

        epochs=36,

        batch_size=4,
        grad_accum_steps=4,

        lr=1e-3,

        device="cuda",

        num_workers=8,

        use_ema=True,

        checkpoint_interval=5,

        early_stopping=False,
    )


if __name__ == "__main__":
    main()
