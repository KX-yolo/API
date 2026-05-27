import gc
import logging
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from config import get_config_regression
from data_loader import MMDataLoader
from trains import ATIO
from trains.singleTask.model import april
from utils import assign_gpu, dict_to_str, setup_seed


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

DEFAULT_SEEDS = list(range(1111, 1120))
logger = logging.getLogger("MMSA")


def _set_logger(verbose_level=1):
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}.get(verbose_level, logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(stream_level)
    handler.setFormatter(logging.Formatter("%(name)s - %(message)s"))
    logger.addHandler(handler)


def _summarize_results(results):
    if not results:
        return {}

    keys = results[0].keys()
    summary = {}
    for key in keys:
        values = [r[key] for r in results if isinstance(r.get(key), (int, float, np.floating))]
        if len(values) == len(results):
            summary[key] = (round(float(np.mean(values)), 4), round(float(np.std(values)), 4))
    return summary


def APRIL_run(
    model_name="april",
    dataset_name="mosi",
    config=None,
    config_file="",
    seeds=None,
    feature_T="",
    feature_A="",
    feature_V="",
    model_save_dir="./pt",
    gpu_ids=None,
    num_workers=2,
    verbose_level=1,
    mode="train",
    mr=0.3,
):
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()
    seeds = DEFAULT_SEEDS if seeds is None else seeds
    gpu_ids = [0] if gpu_ids is None else gpu_ids

    _set_logger(verbose_level)

    config_path = Path(config_file) if config_file else Path(__file__).parent / "config" / "config_april.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    model_save_dir = Path(model_save_dir)
    model_save_dir.mkdir(parents=True, exist_ok=True)

    base_args = get_config_regression(model_name, dataset_name, config_path)
    base_args["mode"] = mode
    base_args["model_save_path"] = model_save_dir / f"{model_name}-{dataset_name}.pth"
    base_args["device"] = assign_gpu(gpu_ids)
    base_args["train_mode"] = "regression"
    base_args["feature_T"] = feature_T
    base_args["feature_A"] = feature_A
    base_args["feature_V"] = feature_V
    base_args["mr"] = mr
    if config:
        base_args.update(config)

    all_results = []
    for seed in seeds:
        setup_seed(seed)
        args = deepcopy(base_args)
        args["cur_seed"] = seed
        logger.info(f"Running APRIL on {dataset_name.upper()} with seed={seed}, mode={mode}, mr={mr}")
        result = _run(args, num_workers)
        all_results.append(result)
        logger.info(f"Seed {seed} TEST >> {dict_to_str(result)}")

    summary = _summarize_results(all_results)
    if len(all_results) > 1 and summary:
        logger.info(f"Summary mean/std >> {summary}")
        return summary
    return all_results[0] if all_results else {}


def _run(args, num_workers=2):
    dataloader = MMDataLoader(args, num_workers)
    model = april.APRIL(args).to(args["device"])
    trainer = ATIO().getTrain(args)

    if args.mode == "test":
        state_dict = torch.load(args["model_save_path"], map_location=args["device"])
        model.load_state_dict(state_dict)
        return trainer.do_test(model, dataloader["test"], mode="TEST")

    trainer.do_train(model, dataloader)
    state_dict = torch.load(args["model_save_path"], map_location=args["device"])
    model.load_state_dict(state_dict)
    results = trainer.do_test(model, dataloader["test"], mode="TEST")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return results
