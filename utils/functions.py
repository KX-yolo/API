import logging
import random

import numpy as np
import torch


logger = logging.getLogger("MMSA")


def dict_to_str(src_dict):
    return "".join(" %s: %.4f " % (key, src_dict[key]) for key in src_dict.keys())


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def assign_gpu(gpu_ids):
    using_cuda = len(gpu_ids) > 0 and torch.cuda.is_available()
    device = torch.device("cuda:%d" % int(gpu_ids[0]) if using_cuda else "cpu")
    if using_cuda:
        torch.cuda.set_device(device)
    logger.info(f"Using device: {device}")
    return device


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
