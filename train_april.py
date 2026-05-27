import argparse

import torch

from run_april import APRIL_run, DEFAULT_SEEDS


def parse_seeds(value):
    if value.strip().lower() == "default":
        return DEFAULT_SEEDS
    return [int(seed.strip()) for seed in value.split(",") if seed.strip()]


def main():
    parser = argparse.ArgumentParser(description="Train APRIL.")
    parser.add_argument("--dataset", default="mosi", choices=["mosi", "mosei"])
    parser.add_argument("--seeds", default="default", help="Comma-separated seeds, or 'default' for 1111..1119.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id. Use -1 for CPU.")
    parser.add_argument("--mr", type=float, default=0.3, help="Missing rate.")
    parser.add_argument("--model_save_dir", default="./pt")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--config_file", default="")
    args = parser.parse_args()

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)

    APRIL_run(
        dataset_name=args.dataset,
        config_file=args.config_file,
        seeds=parse_seeds(args.seeds),
        gpu_ids=[] if args.gpu < 0 else [args.gpu],
        model_save_dir=args.model_save_dir,
        num_workers=args.num_workers,
        mode="train",
        mr=args.mr,
    )


if __name__ == "__main__":
    main()
