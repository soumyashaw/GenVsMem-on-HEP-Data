"""Utils for jupyter notebooks."""

import logging

import torch

from gabbro.utils.pylogger import get_pylogger
from gabbro.utils.utils import get_gpu_properties

_logger = get_pylogger(__name__)
_logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)


def print_gpu_memory():
    """Print GPU memory information.

    Running this also sets up the logging if you're running in a notebook.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free, avail = torch.cuda.mem_get_info()
        # print available GPU memory in GB
        _logger.info(f" -- GPU properties: {get_gpu_properties(verbose=False)}")
        _logger.info(f" -- Available GPU memory: {avail / 1024 / 1024 / 1000:.2f} GB")
        _logger.info(
            f" -- Free GPU memory:      {free / 1024 / 1024 / 1000:.2f} GB ({free / avail:.2%})"
        )
        # get number of gpus
        num_gpus = torch.cuda.device_count()
        _logger.info(f" -- Number of GPUs:       {num_gpus}")
    else:
        _logger.info(" -- No GPU available.")
