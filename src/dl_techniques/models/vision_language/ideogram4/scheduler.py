"""Time schedule and sampler presets for Ideogram4's Euler flow-matching sampler.

Holds the logit-normal time warp (:class:`LogitNormalSchedule`), the linear
step grid the warp is applied to (:func:`make_step_intervals`), the sampler
hyperparameter bundle (:class:`SamplerParameters`), and three named presets.
The warp needs the inverse standard-normal CDF and the logistic sigmoid,
which ``keras.ops`` has no backend-agnostic form of, so it runs in NumPy and
SciPy (float64 internally, cast to float32 on return) rather than as part of
the differentiable model graph. None of these are Keras layers: nothing here
is trainable or saved into a `.keras` file.

Callers must respect the time convention: `t = 0` is clean data and `t = 1`
is pure noise, so sampling walks `t` downward. `LogitNormalSchedule` is
strictly decreasing in its uniform input, while `make_step_intervals` returns
an ascending grid, so a sampler steps the uniform grid forward to make `t`
go down; see the pipeline's Euler loop for how the two compose.
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
    by ``[logsnr_min, logsnr_max]``::

        z   = ndtri(t)                 # inverse standard-normal CDF
        y   = mean + std * z
        t_  = 1 - expit(y)             # 1 - sigmoid
        t_  = clamp(t_, t_min, t_max)

    where ``t_min = 1 / (1 + exp(0.5 * logsnr_max))`` and
    ``t_max = 1 / (1 + exp(0.5 * logsnr_min))``. The output is strictly
    decreasing in `t`: `schedule(0)` sits at the noise end, `schedule(1)` at
    the data end.

    :ivar mean: Mean of the logit-normal warp; resolution-aware in practice.
    :ivar std: Standard deviation of the warp. Defaults to ``1.0``.
    :ivar logsnr_min: Minimum log-SNR, sets the upper time bound `t_max`. Defaults to ``-15.0``.
    :ivar logsnr_max: Maximum log-SNR, sets the lower time bound `t_min`. Defaults to ``18.0``.

    Example:
        >>> sched = LogitNormalSchedule(mean=1.0, std=1.5)
        >>> float(sched(0.5))  # doctest: +SKIP
    """

    mean: float
    std: float = 1.0
    logsnr_min: float = -15.0
    logsnr_max: float = 18.0

    def __call__(self, t: FloatOrArray) -> FloatOrArray:
        """Apply the logit-normal warp and log-SNR clamp.

        :param t: Uniform time(s) in the open interval ``(0, 1)``, as a python
            float or a NumPy array.
        :return: The warped time(s) as float32. A scalar input returns a
            python `float`; an array input returns an `np.ndarray`.
        """
        scalar_input = np.isscalar(t)
        t_arr = np.asarray(t, dtype=np.float64)

        # Inverse standard-normal CDF.
        z = ndtri(t_arr)
        y = self.mean + self.std * z
        # Logistic sigmoid, then flip so t=0 is the noise end.
        t_ = expit(y)
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
    """Build a logit-normal schedule whose mean shifts with image resolution.

    The mean is shifted by half the log pixel-count ratio relative to a known
    reference resolution::

        mean = known_mean + 0.5 * log(num_pixels / known_pixels)

    :param image_resolution: ``(H, W)`` of the target image.
    :param known_resolution: ``(H, W)`` reference resolution. Defaults to ``(512, 512)``.
    :param known_mean: Schedule mean at the reference resolution. Defaults to ``1.0``.
    :param std: Standard deviation passed through to the schedule. Defaults to ``1.0``.
    :return: A :class:`LogitNormalSchedule` with the resolution-shifted mean.
    """
    num_pixels = image_resolution[0] * image_resolution[1]
    known_pixels = known_resolution[0] * known_resolution[1]
    mean = known_mean + 0.5 * math.log(num_pixels / known_pixels)
    return LogitNormalSchedule(mean=mean, std=std)


def make_step_intervals(num_steps: int) -> np.ndarray:
    """Build the uniform grid a schedule warps into sampling times.

    This is the ascending uniform grid, not the sampler's time sequence:
    :class:`LogitNormalSchedule` maps it to a strictly decreasing `t`, so its
    own ascending order says nothing about integration direction.

    :param num_steps: Number of sampling steps.
    :return: A float32 array of shape ``(num_steps + 1,)``, linearly spaced
        on ``[0, 1]`` with both endpoints included, strictly ascending.
    :rtype: np.ndarray
    """
    return np.linspace(0.0, 1.0, num_steps + 1, dtype=np.float32)


@dataclass(frozen=True)
class SamplerParameters:
    """Bundle of sampling hyperparameters for a named preset.

    ``guidance_schedule`` is in loop-index order: index ``0`` is the last
    sampling step, the final polish; index ``num_steps - 1`` is the first
    sampling step. ``mu`` and ``std`` are the mean and standard deviation of
    the logit-normal noise schedule, passed as ``known_mean`` and ``std`` to
    :func:`get_schedule_for_resolution`. The dataclass is frozen: a preset is
    immutable once built.

    :ivar num_steps: Number of Euler integration steps.
    :ivar guidance_schedule: Per-step CFG guidance weights in loop-index order;
        its length must equal ``num_steps``.
    :ivar mu: Mean of the logit-normal noise schedule.
    :ivar std: Standard deviation of the logit-normal noise schedule. Defaults to ``1.0``.
    :raises ValueError: If ``len(guidance_schedule) != num_steps``.
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
