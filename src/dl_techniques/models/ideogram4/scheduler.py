"""Logit-normal time schedule and Euler flow-matching sampler parameters.

Faithful Keras-3 port of the PyTorch ``src/ideogram4/scheduler.py`` plus the
named preset registry from ``src/ideogram4/sampler_configs.py``.

Design note (eager / CPU, NOT a differentiable graph)
-----------------------------------------------------
The schedule produces the *time grid* consumed by the sampling loop: a small
set of scalar ``t`` values that index the integration. It is NOT part of the
differentiable model forward (that is the transformer velocity prediction). The
logit-normal warp needs the inverse standard-normal CDF (``ndtri``) and the
logistic sigmoid (``expit``); ``keras.ops`` has no backend-agnostic ``erfinv`` /
``ndtri``, so we compute the warp in NumPy via :func:`scipy.special.ndtri` and
:func:`scipy.special.expit`, matching PyTorch's ``torch.special.ndtri`` /
``torch.special.expit``. Internals run in float64 (as PyTorch does), then cast
to float32 on return. ``__call__`` accepts a Python float or ``np.ndarray`` and
returns the same kind (scalar in -> scalar out, array in -> array out).

These are plain frozen dataclasses / functions mirroring the PyTorch module; no
``keras.layers.Layer`` is involved (nothing here is trainable or serialized into
a ``.keras`` graph).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
from scipy.special import ndtri, expit

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------------
# Type alias: the schedule accepts/returns either a python float or an ndarray.
# ---------------------------------------------------------------------------
FloatOrArray = Union[float, np.ndarray]


@dataclass(frozen=True)
class LogitNormalSchedule:
    """Logit-normal time warp with a log-SNR clamp.

    Warps a uniform time ``t in (0, 1)`` through the inverse-CDF / sigmoid
    logit-normal transform and clamps the result to the time interval implied
    by ``[logsnr_min, logsnr_max]``. Mirrors the PyTorch ``LogitNormalSchedule``
    exactly::

        z   = ndtri(t)                 # inverse standard-normal CDF
        y   = mean + std * z
        t_  = 1 - expit(y)             # 1 - sigmoid
        t_  = clamp(t_, t_min, t_max)

    where ``t_min = 1 / (1 + exp(0.5 * logsnr_max))`` and
    ``t_max = 1 / (1 + exp(0.5 * logsnr_min))``.

    Args:
        mean: Mean of the logit-normal warp (resolution-aware in practice).
        std: Standard deviation of the warp. Defaults to ``1.0``.
        logsnr_min: Minimum log-SNR; sets the upper time bound ``t_max``.
            Defaults to ``-15.0``.
        logsnr_max: Maximum log-SNR; sets the lower time bound ``t_min``.
            Defaults to ``18.0``.

    Example:
        >>> sched = LogitNormalSchedule(mean=1.0, std=1.5)
        >>> float(sched(0.5))  # doctest: +SKIP
    """

    mean: float
    std: float = 1.0
    logsnr_min: float = -15.0
    logsnr_max: float = 18.0

    def __call__(self, t: FloatOrArray) -> FloatOrArray:
        """Apply the logit-normal warp + log-SNR clamp.

        Args:
            t: Uniform time(s) in the open interval ``(0, 1)``; a python float
                or a NumPy array.

        Returns:
            The warped time(s) as float32. Scalar in -> python ``float`` out;
            array in -> ``np.ndarray`` (float32) out.
        """
        scalar_input = np.isscalar(t)
        # float64 internally, matching torch.float64.
        t_arr = np.asarray(t, dtype=np.float64)

        z = ndtri(t_arr)                       # inverse standard-normal CDF
        y = self.mean + self.std * z
        t_ = expit(y)                          # logistic sigmoid
        t_ = 1.0 - t_

        t_min = 1.0 / (1.0 + math.exp(0.5 * self.logsnr_max))
        t_max = 1.0 / (1.0 + math.exp(0.5 * self.logsnr_min))
        t_ = np.clip(t_, t_min, t_max).astype(np.float32)

        if scalar_input:
            return float(t_)
        return t_


def get_schedule_for_resolution(
    image_resolution: Tuple[int, int],
    known_resolution: Tuple[int, int] = (512, 512),
    known_mean: float = 1.0,
    std: float = 1.0,
) -> LogitNormalSchedule:
    """Build a resolution-aware logit-normal schedule (eval-time).

    The mean is shifted by half the log pixel-count ratio relative to a known
    reference resolution::

        mean = known_mean + 0.5 * log(num_pixels / known_pixels)

    Args:
        image_resolution: ``(H, W)`` of the target image.
        known_resolution: ``(H, W)`` reference resolution. Defaults to
            ``(512, 512)``.
        known_mean: Schedule mean at the reference resolution. Defaults to
            ``1.0``.
        std: Standard deviation passed through to the schedule. Defaults to
            ``1.0``.

    Returns:
        A :class:`LogitNormalSchedule` with the resolution-shifted mean.
    """
    num_pixels = image_resolution[0] * image_resolution[1]
    known_pixels = known_resolution[0] * known_resolution[1]
    mean = known_mean + 0.5 * math.log(num_pixels / known_pixels)
    return LogitNormalSchedule(mean=mean, std=std)


def make_step_intervals(num_steps: int) -> np.ndarray:
    """Default linear step schedule used by the v4 eval config.

    Args:
        num_steps: Number of sampling steps.

    Returns:
        A float32 ``np.ndarray`` of shape ``(num_steps + 1,)`` linearly spaced
        on ``[0, 1]`` (endpoints inclusive).
    """
    return np.linspace(0.0, 1.0, num_steps + 1, dtype=np.float32)


@dataclass(frozen=True)
class SamplerParameters:
    """Bundle of sampling hyperparameters for a named preset.

    ``guidance_schedule`` is in LOOP-INDEX order: index ``0`` is the LAST
    sampling step (final polish), index ``num_steps - 1`` is the FIRST sampling
    step. ``mu`` and ``std`` are the mean and stddev of the logit-normal noise
    schedule (passed as ``known_mean`` and ``std`` to
    :func:`get_schedule_for_resolution`).

    Note:
        Declared ``frozen`` (immutable preset). The PyTorch original is also
        ``kw_only``; here we keep field order so positional construction in
        :data:`PRESETS` reads identically to the reference.

    Args:
        num_steps: Number of Euler integration steps.
        guidance_schedule: Per-step CFG guidance weights in loop-index order;
            its length MUST equal ``num_steps``.
        mu: Mean of the logit-normal noise schedule.
        std: Standard deviation of the logit-normal noise schedule. Defaults to
            ``1.0``.

    Raises:
        ValueError: If ``len(guidance_schedule) != num_steps``.
    """

    num_steps: int
    guidance_schedule: Tuple[float, ...]
    mu: float
    std: float = 1.0

    def __post_init__(self) -> None:
        if len(self.guidance_schedule) != self.num_steps:
            raise ValueError(
                f"guidance_schedule has length {len(self.guidance_schedule)}, "
                f"expected num_steps={self.num_steps}"
            )


# ---------------------------------------------------------------------------
# Named preset registry (ported from src/ideogram4/sampler_configs.py).
# guidance_schedule is in loop-INDEX order: index 0 is the LAST (polish) step.
# Each preset does the first N_main sampling steps at gw=7, then N_cleanup
# polish steps at gw=3.
# ---------------------------------------------------------------------------
PRESETS: dict[str, SamplerParameters] = {
    "V4_QUALITY_48": SamplerParameters(
        num_steps=48,
        guidance_schedule=(3.0,) * 3 + (7.0,) * 45,
        mu=0.0,
        std=1.5,
    ),
    "V4_DEFAULT_20": SamplerParameters(
        num_steps=20,
        guidance_schedule=(3.0,) * 2 + (7.0,) * 18,
        mu=0.0,
        std=1.75,
    ),
    "V4_TURBO_12": SamplerParameters(
        num_steps=12,
        guidance_schedule=(3.0,) * 1 + (7.0,) * 11,
        mu=0.5,
        std=1.75,
    ),
}

logger.debug(
    "Ideogram4 scheduler presets loaded: %s",
    ", ".join(PRESETS.keys()),
)
