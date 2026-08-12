"""Common GPU setup utilities for training scripts."""

import os
from typing import Optional

import keras
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

# ---------------------------------------------------------------------

def log_gpu_peak_memory() -> None:
    """Log peak and current memory use for every visible GPU.

    Useful right after ``model.fit`` to record how close a run came to the
    device limit. Purely a reporting path: any failure to read the counters is
    logged and swallowed rather than raised.
    """
    for device in tf.config.list_physical_devices("GPU"):
        try:
            info = tf.config.experimental.get_memory_info(
                f"GPU:{device.name.split(':')[-1]}"
            )
            logger.info(
                "GPU peak on %s: %.1f MiB (current %.1f MiB)",
                device.name,
                info["peak"] / 1024 ** 2,
                info["current"] / 1024 ** 2,
            )
        except Exception as error:  # pragma: no cover - reporting path
            logger.warning("Could not read GPU memory info: %s", error)


# ---------------------------------------------------------------------

def setup_mixed_precision(
        enabled: bool,
        policy: str = "mixed_float16",
) -> bool:
    """Set the global Keras dtype policy and report what was applied.

    Sets ``policy`` when ``enabled``, and ``"float32"`` otherwise -- the
    ``else`` branch matters, because the global policy is process-wide state
    that an earlier call (or an imported module) may already have changed.

    Note this configures the POLICY only. Under ``mixed_float16`` the optimizer
    must additionally be wrapped in ``keras.mixed_precision.LossScaleOptimizer``
    to avoid gradient underflow; do that at the call site, after building the
    optimizer, since not every trainer builds one the same way.

    ``bfloat16`` needs no loss scaling (it keeps float32's exponent range), so
    pass ``policy="mixed_bfloat16"`` and skip the wrap.

    Args:
        enabled: Whether to enable mixed precision.
        policy: Policy name to apply when enabled, e.g. ``"mixed_float16"`` or
            ``"mixed_bfloat16"``.

    Returns:
        ``True`` if mixed precision was enabled, ``False`` otherwise. Returning
        the flag lets callers write
        ``use_loss_scale = setup_mixed_precision(cfg.mixed_precision)``.
    """
    if enabled:
        keras.mixed_precision.set_global_policy(policy)
        logger.info(f"Mixed precision ENABLED: global policy={policy}")
        return True

    keras.mixed_precision.set_global_policy("float32")
    logger.info("Mixed precision disabled: global policy=float32")
    return False
