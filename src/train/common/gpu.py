"""Common GPU setup utilities for training scripts."""

import os
from typing import Optional

import keras
import tensorflow as tf

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------

def visible_device_for(gpu_id: int, visible: Optional[str]) -> str:
    """The ``CUDA_VISIBLE_DEVICES`` value that selects GPU ``gpu_id``.

    Interface contract (callers: :func:`setup_gpu`; any code that must turn a
    ``--gpu`` index into an environment value uses this, not its own rule):

    * ``visible`` is the current ``CUDA_VISIBLE_DEVICES`` (``None`` when the
      variable is unset). When it is set, ``gpu_id`` indexes into its
      comma-separated entries, in order, and the selected entry is returned
      verbatim (an index or a ``GPU-<uuid>``). When it is unset, ``gpu_id``
      names a physical device and ``str(gpu_id)`` is returned.
    * Raises ``ValueError`` for a negative ``gpu_id``, and for an index outside
      the visible list (an empty ``visible`` lists no device).
    * Pure: reads and writes no environment.

    Args:
        gpu_id: Index of the GPU to use.
        visible: The current ``CUDA_VISIBLE_DEVICES``, or ``None`` if unset.

    Returns:
        The single-device ``CUDA_VISIBLE_DEVICES`` value.
    """
    if gpu_id < 0:
        raise ValueError(
            f"GPU index {gpu_id} is negative; to run without a GPU set "
            "CUDA_VISIBLE_DEVICES= (empty) instead.")
    if visible is None:
        return str(gpu_id)
    # DECISION plan-2026-10-14T062616-81938e81/D-110
    # Index INTO an exported CUDA_VISIBLE_DEVICES; do NOT overwrite it with
    # str(gpu_id). Overwriting sent `CUDA_VISIBLE_DEVICES=1 ... --gpu 0` to
    # physical GPU 0 (every Step 13.5 smoke ran on the wrong GPU that way).
    entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    if gpu_id >= len(entries):
        raise ValueError(
            f"GPU index {gpu_id} is outside CUDA_VISIBLE_DEVICES={visible!r}, "
            f"which lists {len(entries)} device(s); the index counts within "
            "that list, not over the physical GPUs.")
    return entries[gpu_id]


def setup_gpu(gpu_id: Optional[int] = None):
    """Configure GPU settings for optimal training.

    Args:
        gpu_id: Index of the GPU to use, resolved by :func:`visible_device_for`:
            it counts within an exported ``CUDA_VISIBLE_DEVICES`` and names a
            physical device only when that variable is unset. The result
            restricts ``CUDA_VISIBLE_DEVICES`` to that one GPU. If None, the
            environment is left alone and memory growth is enabled on every
            visible GPU.

    Raises:
        ValueError: ``gpu_id`` is negative or outside the visible list.
    """
    if gpu_id is not None:
        selected = visible_device_for(gpu_id, os.environ.get("CUDA_VISIBLE_DEVICES"))
        os.environ["CUDA_VISIBLE_DEVICES"] = selected
        logger.info(f"--gpu {gpu_id}: set CUDA_VISIBLE_DEVICES={selected}")

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
