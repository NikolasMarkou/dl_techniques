"""Common GPU setup utilities for training scripts."""

import os
from typing import Dict, Mapping, Optional

import keras
import tensorflow as tf

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------

# The variable this module is the only writer of (census in
# tests/test_train/test_common_setup_gpu_respects_visible_devices.py).
VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"


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


def gpu_or_default(
        gpu_id: Optional[int],
        visible: Optional[str],
        default_gpu: int,
) -> Optional[int]:
    """The GPU index a script with its own default device selects.

    Interface contract (callers: :func:`child_visible_devices`,
    ``bfunet/eval_per_pixel_uncertainty.py``, ``nam/eval_dfsa.py``,
    ``nam/test_extreme.py``, ``cliffordnet/eval_clip_retrieval.py``,
    ``cliffordnet/filter_cc3m_clipscore.py``, ``logic/train_e5_clevr_hans.py``,
    ``mothnet/train_mothnet.py``, ``thera/train_thera.py``; a script with a
    preferred GPU defaults ``--gpu`` to ``None`` and picks it here, D-118,
    D-234):

    * An explicit ``gpu_id`` wins and is returned unchanged.
    * With no ``gpu_id``, an exported ``CUDA_VISIBLE_DEVICES`` (``visible`` is
      not ``None``, including the empty string) is respected: returns ``None``,
      meaning "select nothing, keep the exported list".
    * With neither, returns ``default_gpu``, which then names a physical GPU.
    * Pure: reads and writes no environment.
    """
    if gpu_id is not None:
        return gpu_id
    return None if visible is not None else default_gpu


def child_visible_devices(
        gpu_id: Optional[int],
        visible: Optional[str],
        default_gpu: int,
) -> str:
    """The ``CUDA_VISIBLE_DEVICES`` value to give a child process.

    Interface contract (callers: ``vae/run_sampler_comparison.py``,
    ``logic/multiseed_sweep.py``, ``mothnet/multiseed_sweep.py``,
    ``convnext/run_stochastic_comparison.py``, ``rms_variants_train/sweep.py``,
    ``bfunet/run_gabor_depthwise_depth_sweep.sh``): :func:`gpu_or_default`
    picks the index, and
    :func:`visible_device_for` resolves it against ``visible`` (the launcher's
    own, inherited value). When nothing is selected, ``visible`` is returned
    verbatim. Raises ``ValueError`` as :func:`visible_device_for` does. Pure.
    """
    selection = gpu_or_default(gpu_id, visible, default_gpu)
    if selection is None:
        return visible
    return visible_device_for(selection, visible)


def env_with_visible_devices(
        value: str,
        base: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """A copy of ``base`` whose ``CUDA_VISIBLE_DEVICES`` is ``value``.

    Interface contract (callers: every launcher that starts a training child):

    * ``value`` is an already resolved list -- from :func:`visible_device_for`
      or :func:`child_visible_devices`, or ``""`` for a CPU-only child. Never
      pass ``str(gpu_id)``: that is the rule D-110 replaced.
    * ``base`` defaults to ``os.environ``; it is copied, never modified.
    * Returns the new mapping, ready for ``subprocess.run(env=...)``.
    """
    env = dict(os.environ if base is None else base)
    # DECISION plan-2026-10-14T062616-81938e81/D-116
    # Launchers build a child's environment HERE; do NOT write
    # env["CUDA_VISIBLE_DEVICES"] = str(gpu_id) in a launcher. Six scripts kept
    # that write after D-110 and sent `CUDA_VISIBLE_DEVICES=1 ... --gpu 0`
    # children to physical GPU 0; a census test holds this module as the only
    # writer.
    env[VISIBLE_DEVICES] = value
    return env


def hide_gpus() -> Optional[str]:
    """Make THIS process CPU-only, before TensorFlow lists its devices.

    Interface contract (callers: the comparison drivers' ``main``, which
    orchestrate GPU children and must not hold a GPU context themselves; the
    time-series export scripts' ``main``, whose exported graph must be
    device-neutral, D-235):

    * Sets ``CUDA_VISIBLE_DEVICES`` to the empty string in ``os.environ``.
    * Returns the value it replaced (``None`` if the variable was unset), so a
      driver can still resolve its children's ``--gpu`` against what it
      inherited.
    * Takes effect when called before the first device listing; importing
      TensorFlow (as this module does) lists none (measured in step 13.7).
    * Never call it at import time of a module something else imports: the
      importer loses its GPUs for good (D-118).
    """
    inherited = os.environ.get(VISIBLE_DEVICES)
    os.environ[VISIBLE_DEVICES] = ""
    return inherited


#: Why a selection or a memory-growth call can come too late; both refusals
#: in :func:`setup_gpu` end with it.
_TOO_LATE = (
    "TensorFlow listed or initialised its devices before setup_gpu ran (an op, "
    "a device listing or a model build at import time or earlier in the "
    "process); call setup_gpu first, or select the device by exporting "
    "CUDA_VISIBLE_DEVICES before Python starts")


def setup_gpu(gpu_id: Optional[int] = None):
    """Select the GPU and enable memory growth, or raise.

    Interface contract (callers: the 133 modules under ``src/train`` that call
    it, D-234 census; each once, before building anything):

    * With ``gpu_id``, ``CUDA_VISIBLE_DEVICES`` is restricted to the device
      :func:`visible_device_for` resolves, and the process must then list
      exactly one GPU; otherwise ``RuntimeError`` (the selection did not take
      effect: TensorFlow had already read the variable).
    * With ``None`` the environment is left alone.
    * Memory growth is enabled on every listed GPU. A failure raises
      ``RuntimeError`` naming the import-order cause; it is never logged and
      swallowed. Calling again after a successful call is a no-op (TensorFlow
      returns early for a device whose growth is already enabled).

    Args:
        gpu_id: Index of the GPU to use, resolved by :func:`visible_device_for`:
            it counts within an exported ``CUDA_VISIBLE_DEVICES`` and names a
            physical device only when that variable is unset. If None, the
            environment is left alone and memory growth is enabled on every
            visible GPU.

    Raises:
        ValueError: ``gpu_id`` is negative or outside the visible list.
        RuntimeError: ``gpu_id`` was given and the process does not list
            exactly one GPU, or memory growth could not be enabled.
    """
    selected = None
    if gpu_id is not None:
        selected = visible_device_for(gpu_id, os.environ.get(VISIBLE_DEVICES))
        os.environ[VISIBLE_DEVICES] = selected
        logger.info(f"--gpu {gpu_id}: set CUDA_VISIBLE_DEVICES={selected}")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    # DECISION plan-2026-10-14T062616-81938e81/D-234
    # Raise when the selection or the memory growth did not take effect; do NOT
    # go back to logging the RuntimeError and continuing: that ran a "--gpu 0"
    # trainer on whatever TensorFlow had already picked, with the exit code 0.
    if gpu_id is not None and len(gpus) != 1:
        raise RuntimeError(
            f"--gpu {gpu_id} set CUDA_VISIBLE_DEVICES={selected}, but this "
            f"process lists {len(gpus)} GPU(s), not 1: the selection did not "
            f"take effect. {_TOO_LATE}.")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as error:
            raise RuntimeError(
                f"Could not enable memory growth on {gpu.name}: {error}. "
                f"{_TOO_LATE}.") from error
    if gpus:
        logger.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
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
