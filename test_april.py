import argparse

from run_april import APRIL_run


def main():
    parser = argparse.ArgumentParser(description="Test APRIL.")
    parser.add_argument("--dataset", default="mosi", choices=["mosi", "mosei"])
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--gpu", type=int, default=0, help="GPU id. Use -1 for CPU.")
    parser.add_argument("--mr", type=float, default=0.3, help="Missing rate.")
    parser.add_argument("--model_save_dir", default="./pt")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--config_file", default="")
    args = parser.parse_args()

    APRIL_run(
        dataset_name=args.dataset,
        config_file=args.config_file,
        seeds=[args.seed],
        gpu_ids=[] if args.gpu < 0 else [args.gpu],
        model_save_dir=args.model_save_dir,
        num_workers=args.num_workers,
        mode="test",
        mr=args.mr,
    )


if __name__ == "__main__":
    main()
