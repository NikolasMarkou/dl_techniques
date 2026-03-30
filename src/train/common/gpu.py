"""Common GPU setup utilities for training scripts."""

import os
from typing import Optional

import tensorflow as tf

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------

def setup_gpu(gpu_id: Optional[int] = None):
    """Configure GPU settings for optimal training.

    Args:
        gpu_id: Specific GPU device index to use. If provided, sets
            CUDA_VISIBLE_DEVICES to restrict to that GPU. If None,
            enables memory growth on all available GPUs.
    """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logger.info(f"Set CUDA_VISIBLE_DEVICES={gpu_id}")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
        except RuntimeError as e:
            logger.error(f"GPU setup error: {e}")
    else:
        logger.info("No GPUs found, using CPU")
