"""
DDPM noise schedules and the constant tables a Gaussian diffusion process is built from.

A denoising diffusion probabilistic model is defined entirely by one sequence of
numbers: the per-step variance schedule ``beta_1 ... beta_T``. Everything else --
how much signal survives to step ``t``, how to jump from clean data to a noised
sample in one shot, what the true reverse-step posterior looks like, and the
bounds a learned variance is allowed to interpolate between -- is an algebraic
consequence of that sequence. This module owns that algebra and nothing else. It
holds no model, no sampler and no loss; it is a pure-NumPy value object that the
loss, the sampler and the data pipeline each construct from the same
configuration so that all three provably agree on the same numbers.

The tables are computed and stored in ``float64``, exactly as the upstream
reference does ("Use float64 for accuracy"). ``alphas_cumprod`` is a product of
up to a thousand factors slightly below one and ``posterior_mean_coef2`` divides
by ``1 - alphas_cumprod``, which approaches zero at the head of the chain; both
lose meaningful precision in ``float32``. Consumers cast at the point of use --
this module never does, because a silent narrowing here has no shape symptom and
would be invisible in every downstream test that only checks finiteness.

The forward process and the reverse posterior, and where each table is read:

.. code-block:: text

    forward (fixed, no learning)                q(x_t | x_0)
    ┌──────────────┐                            ┌──────────────────────────┐
    │ x_0  [B,H,W,C]│ ─────────────────────────▶ │ sqrt_alphas_cumprod[t]   │
    └──────────────┘                            │      * x_0               │
                                                │  ⊕                       │
    ┌──────────────┐                            │ sqrt_one_minus_alphas_   │
    │ noise [B,..] │ ─────────────────────────▶ │  cumprod[t] * noise      │
    └──────────────┘                            └────────────┬─────────────┘
                                                             │
                                                             ▼
                                                    ┌──────────────┐
                                                    │ x_t [B,H,W,C]│
                                                    └──────┬───────┘
                                                           │
    reverse (learned)                                      ▼
    ┌────────────────────────────────────────────────────────────────────┐
    │ x_0_hat = sqrt_recip_alphas_cumprod[t]   * x_t                     │
    │           ⊖ sqrt_recipm1_alphas_cumprod[t] * eps_hat               │
    │                                                                    │
    │ q(x_{t-1} | x_t, x_0):                                             │
    │   mean = posterior_mean_coef1[t] * x_0_hat                         │
    │          ⊕ posterior_mean_coef2[t] * x_t                           │
    │   var  = posterior_variance[t]                                     │
    │   log var (clipped at t=0) = posterior_log_variance_clipped[t]     │
    │                                                                    │
    │ LEARNED_RANGE: log var = frac * log(betas[t])                      │
    │                        ⊕ (1 - frac) * posterior_log_variance_      │
    │                                        clipped[t]                  │
    └────────────────────────────────────────────────────────────────────┘

``space_timesteps`` selects a subset of the original ``T`` indices for fast
sampling, and :meth:`DDPMSchedule.respaced` rebuilds a whole consistent table set
over that subset by re-deriving betas from ratios of the ORIGINAL
``alphas_cumprod``. That derivation lives here rather than in the sampler because
it produces schedule tables, and a sampler that re-derived them would be a second
place the schedule algebra could drift.

References:
    - Ho et al., 2020. Denoising Diffusion Probabilistic Models.
      (https://arxiv.org/abs/2006.11239)
    - Nichol & Dhariwal, 2021. Improved Denoising Diffusion Probabilistic
      Models -- the ``squaredcos_cap_v2`` cosine schedule and the LEARNED_RANGE
      variance interpolation. (https://arxiv.org/abs/2102.09672)
    - Song et al., 2020. Denoising Diffusion Implicit Models -- the ``"ddimN"``
      fixed-stride respacing form. (https://arxiv.org/abs/2010.02502)
    - Peebles & Xie, 2022. Scalable Diffusion Models with Transformers.
      (https://arxiv.org/abs/2212.09748)
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Set, Tuple, Union

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

VALID_BETA_SCHEDULES: Tuple[str, ...] = ("linear", "squaredcos_cap_v2")
"""The beta schedule names this module supports, in the order they are listed in errors."""

MAX_BETA: float = 0.999
"""Upper clamp applied by :func:`betas_for_alpha_bar`, preventing a singular final step."""


# ---------------------------------------------------------------------
# Beta schedules
# ---------------------------------------------------------------------


def betas_for_alpha_bar(
        num_diffusion_timesteps: int,
        alpha_bar: Callable[[float], float],
        max_beta: float = MAX_BETA,
) -> np.ndarray:
    """Discretize a continuous ``alpha_bar(t)`` curve into a beta schedule.

    Each beta is the fractional drop in cumulative signal over one step,
    ``1 - alpha_bar(t2) / alpha_bar(t1)``, clamped at ``max_beta`` so the last
    steps cannot become singular.

    :param num_diffusion_timesteps: Number of betas to produce.
    :type num_diffusion_timesteps: int
    :param alpha_bar: Function mapping ``t`` in ``[0, 1]`` to the cumulative
        product of ``(1 - beta)`` up to that point.
    :type alpha_bar: Callable[[float], float]
    :param max_beta: Largest permitted beta.
    :type max_beta: float
    :return: ``float64`` array of shape ``(num_diffusion_timesteps,)``.
    :rtype: np.ndarray
    """
    betas: List[float] = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1.0 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float64)


def get_named_beta_schedule(
        schedule_name: str,
        num_diffusion_timesteps: int,
) -> np.ndarray:
    """Return one of the named DDPM beta schedules.

    ``"linear"`` is Ho et al.'s schedule, defined for 1000 steps and extended to
    any step count by the factor ``scale = 1000 / num_diffusion_timesteps``
    applied to both endpoints. The rescale keeps the total amount of noise
    injected over the whole chain roughly constant as the chain is shortened; at
    exactly 1000 steps ``scale`` is ``1.0`` and the factor is inert. The rescale
    also bounds how short a ``"linear"`` chain may be: ``beta_end`` is
    ``scale * 0.02 == 20 / num_diffusion_timesteps``, so it reaches exactly
    ``1.0`` at ``num_diffusion_timesteps == 20`` and exceeds it below that, and
    such a chain is rejected by :meth:`DDPMSchedule.from_betas` (upstream
    asserts in the same place). This is a boundary, not a floor: at
    ``num_diffusion_timesteps == 1`` ``np.linspace`` returns ``[beta_start]``
    and drops the endpoint entirely, so ``1`` is accepted while ``2`` through
    ``19`` are not. At exactly ``20`` the final ``alphas_cumprod`` is ``0``,
    which makes ``sqrt_recipm1_alphas_cumprod`` infinite, so ``20`` is the last
    accepted value rather than a usable one. Use ``"squaredcos_cap_v2"``, which
    is capped by construction, for short chains. The measured accepted set is
    pinned by ``tests/test_models/test_dit/test_dit_config.py``.

    ``"squaredcos_cap_v2"`` is Nichol & Dhariwal's cosine schedule, obtained by
    discretizing ``alpha_bar(t) = cos((t + 0.008) / 1.008 * pi / 2) ** 2`` with
    :func:`betas_for_alpha_bar` and a ``0.999`` cap.

    :param schedule_name: One of :data:`VALID_BETA_SCHEDULES`.
    :type schedule_name: str
    :param num_diffusion_timesteps: Length of the produced schedule.
    :type num_diffusion_timesteps: int
    :return: ``float64`` array of shape ``(num_diffusion_timesteps,)``.
    :rtype: np.ndarray
    :raises ValueError: If ``schedule_name`` is not a supported name, or if
        ``num_diffusion_timesteps`` is not a positive integer.
    """
    if num_diffusion_timesteps <= 0:
        raise ValueError(
            f"num_diffusion_timesteps must be positive, got {num_diffusion_timesteps}"
        )

    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )

    if schedule_name == "squaredcos_cap_v2":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )

    raise ValueError(
        f"unknown beta schedule: {schedule_name!r}. "
        f"Valid names are: {', '.join(VALID_BETA_SCHEDULES)}"
    )


# ---------------------------------------------------------------------
# The schedule value object
# ---------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class DDPMSchedule:
    """Immutable container for every constant table a Gaussian diffusion process reads.

    Construct one with :meth:`from_name` (the usual path) or :meth:`from_betas`
    (when the betas already exist, e.g. inside :meth:`respaced`). Every field is a
    ``float64`` array of shape ``(num_timesteps,)``; nothing here is ever cast to
    a narrower dtype, because the loss and the sampler cast at the point of use
    and a narrowing at construction time would be invisible.

    ``eq=False`` because the fields are NumPy arrays: a generated ``__eq__``
    would return arrays rather than a bool and silently break any ``==`` on the
    dataclass.

    :param betas: Per-step forward variances, all in ``(0, 1]``.
    :type betas: np.ndarray
    :param alphas: ``1 - betas``.
    :type alphas: np.ndarray
    :param alphas_cumprod: Cumulative product of ``alphas``; the signal fraction
        surviving to step ``t``.
    :type alphas_cumprod: np.ndarray
    :param alphas_cumprod_prev: ``alphas_cumprod`` shifted right, with ``1.0``
        prepended, so index ``0`` means "before any noise".
    :type alphas_cumprod_prev: np.ndarray
    :param alphas_cumprod_next: ``alphas_cumprod`` shifted left, with ``0.0``
        appended (fully destroyed signal past the end of the chain).
    :type alphas_cumprod_next: np.ndarray
    :param sqrt_alphas_cumprod: Signal coefficient of ``q(x_t | x_0)``.
    :type sqrt_alphas_cumprod: np.ndarray
    :param sqrt_one_minus_alphas_cumprod: Noise coefficient of ``q(x_t | x_0)``.
    :type sqrt_one_minus_alphas_cumprod: np.ndarray
    :param log_one_minus_alphas_cumprod: ``log(1 - alphas_cumprod)``, the log
        variance of ``q(x_t | x_0)``.
    :type log_one_minus_alphas_cumprod: np.ndarray
    :param sqrt_recip_alphas_cumprod: ``sqrt(1 / alphas_cumprod)``, used to
        recover ``x_0`` from ``(x_t, eps)``.
    :type sqrt_recip_alphas_cumprod: np.ndarray
    :param sqrt_recipm1_alphas_cumprod: ``sqrt(1 / alphas_cumprod - 1)``, the
        epsilon coefficient of the same recovery.
    :type sqrt_recipm1_alphas_cumprod: np.ndarray
    :param posterior_variance: Variance of ``q(x_{t-1} | x_t, x_0)``; exactly
        ``0.0`` at index ``0``.
    :type posterior_variance: np.ndarray
    :param posterior_log_variance_clipped: ``log(posterior_variance)`` with index
        ``0`` replaced by index ``1`` before the log, because the true value is
        ``log(0) = -inf``. Consequently entries ``0`` and ``1`` are equal.
    :type posterior_log_variance_clipped: np.ndarray
    :param posterior_mean_coef1: Coefficient of ``x_0`` in the posterior mean.
    :type posterior_mean_coef1: np.ndarray
    :param posterior_mean_coef2: Coefficient of ``x_t`` in the posterior mean.
    :type posterior_mean_coef2: np.ndarray
    """

    betas: np.ndarray
    alphas: np.ndarray
    alphas_cumprod: np.ndarray
    alphas_cumprod_prev: np.ndarray
    alphas_cumprod_next: np.ndarray
    sqrt_alphas_cumprod: np.ndarray
    sqrt_one_minus_alphas_cumprod: np.ndarray
    log_one_minus_alphas_cumprod: np.ndarray
    sqrt_recip_alphas_cumprod: np.ndarray
    sqrt_recipm1_alphas_cumprod: np.ndarray
    posterior_variance: np.ndarray
    posterior_log_variance_clipped: np.ndarray
    posterior_mean_coef1: np.ndarray
    posterior_mean_coef2: np.ndarray

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def num_timesteps(self) -> int:
        """Number of diffusion steps this schedule covers.

        :return: ``betas.shape[0]``.
        :rtype: int
        """
        return int(self.betas.shape[0])

    # -----------------------------------------------------------------
    # Factories
    # -----------------------------------------------------------------

    @classmethod
    def from_betas(cls, betas: Union[np.ndarray, Sequence[float]]) -> "DDPMSchedule":
        """Derive every table from a 1-D beta sequence.

        :param betas: The forward variance schedule. Cast to ``float64``.
        :type betas: Union[np.ndarray, Sequence[float]]
        :return: The fully derived schedule.
        :rtype: DDPMSchedule
        :raises ValueError: If ``betas`` is not 1-D, is empty, or has any entry
            outside ``(0, 1]``.
        """
        betas_arr = np.asarray(betas, dtype=np.float64)

        if betas_arr.ndim != 1:
            raise ValueError(f"betas must be 1-D, got shape {betas_arr.shape}")
        if betas_arr.shape[0] == 0:
            raise ValueError("betas must be non-empty")
        if not (betas_arr > 0).all():
            raise ValueError(
                f"every beta must be > 0, got min {float(betas_arr.min())}"
            )
        if not (betas_arr <= 1).all():
            raise ValueError(
                f"every beta must be <= 1, got max {float(betas_arr.max())}"
            )

        alphas = 1.0 - betas_arr
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        alphas_cumprod_next = np.append(alphas_cumprod[1:], 0.0)

        posterior_variance = (
                betas_arr * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # The posterior variance is exactly 0 at t=0, so its log is -inf. Upstream
        # clips by substituting index 1 for index 0 BEFORE taking the log, which
        # makes entries 0 and 1 equal.
        if posterior_variance.shape[0] > 1:
            posterior_log_variance_clipped = np.log(
                np.append(posterior_variance[1], posterior_variance[1:])
            )
        else:
            posterior_log_variance_clipped = np.array([], dtype=np.float64)

        return cls(
            betas=betas_arr,
            alphas=alphas,
            alphas_cumprod=alphas_cumprod,
            alphas_cumprod_prev=alphas_cumprod_prev,
            alphas_cumprod_next=alphas_cumprod_next,
            sqrt_alphas_cumprod=np.sqrt(alphas_cumprod),
            sqrt_one_minus_alphas_cumprod=np.sqrt(1.0 - alphas_cumprod),
            log_one_minus_alphas_cumprod=np.log(1.0 - alphas_cumprod),
            sqrt_recip_alphas_cumprod=np.sqrt(1.0 / alphas_cumprod),
            sqrt_recipm1_alphas_cumprod=np.sqrt(1.0 / alphas_cumprod - 1),
            posterior_variance=posterior_variance,
            posterior_log_variance_clipped=posterior_log_variance_clipped,
            posterior_mean_coef1=(
                    betas_arr * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            ),
            posterior_mean_coef2=(
                    (1.0 - alphas_cumprod_prev)
                    * np.sqrt(alphas)
                    / (1.0 - alphas_cumprod)
            ),
        )

    @classmethod
    def from_name(cls, schedule_name: str, num_timesteps: int) -> "DDPMSchedule":
        """Build a schedule from one of the named beta schedules.

        :param schedule_name: One of :data:`VALID_BETA_SCHEDULES`.
        :type schedule_name: str
        :param num_timesteps: Length of the diffusion chain.
        :type num_timesteps: int
        :return: The fully derived schedule.
        :rtype: DDPMSchedule
        :raises ValueError: If the name is unknown or the derived betas are
            outside ``(0, 1]``.
        """
        betas = get_named_beta_schedule(schedule_name, num_timesteps)
        logger.info(
            f"DDPMSchedule: '{schedule_name}' over {num_timesteps} timesteps "
            f"(beta {float(betas[0]):.6g} -> {float(betas[-1]):.6g})"
        )
        return cls.from_betas(betas)

    # -----------------------------------------------------------------
    # Respacing
    # -----------------------------------------------------------------

    def respaced(
            self,
            use_timesteps: Iterable[int],
    ) -> Tuple["DDPMSchedule", np.ndarray]:
        """Rebuild the tables over a retained subset of this schedule's timesteps.

        The new betas are NOT a subsample of ``self.betas``. They are re-derived
        from ratios of the ORIGINAL ``alphas_cumprod`` so that the shortened
        chain reaches the same cumulative signal levels at the retained indices:
        walking the retained indices in order, each new beta is
        ``1 - alphas_cumprod[i] / last_alphas_cumprod``.

        The returned map takes an index into the shortened chain and yields the
        original timestep, which is the value that must be passed to a model
        trained on the full chain.

        :param use_timesteps: Original timestep indices to retain, in any order
            (typically the output of :func:`space_timesteps`).
        :type use_timesteps: Iterable[int]
        :return: ``(schedule, timestep_map)`` where ``schedule`` covers
            ``len(timestep_map)`` steps and ``timestep_map`` is an ``int64``
            array of the retained original indices in increasing order.
        :rtype: Tuple[DDPMSchedule, np.ndarray]
        :raises ValueError: If ``use_timesteps`` is empty or contains an index
            outside ``[0, num_timesteps)``.
        """
        retained: Set[int] = {int(i) for i in use_timesteps}
        if not retained:
            raise ValueError("use_timesteps must retain at least one timestep")
        out_of_range = sorted(
            i for i in retained if i < 0 or i >= self.num_timesteps
        )
        if out_of_range:
            raise ValueError(
                f"use_timesteps entries out of range [0, {self.num_timesteps}): "
                f"{out_of_range}"
            )

        last_alpha_cumprod = 1.0
        new_betas: List[float] = []
        timestep_map: List[int] = []
        for i, alpha_cumprod in enumerate(self.alphas_cumprod):
            if i in retained:
                new_betas.append(1.0 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                timestep_map.append(i)

        return (
            DDPMSchedule.from_betas(np.array(new_betas, dtype=np.float64)),
            np.array(timestep_map, dtype=np.int64),
        )


# ---------------------------------------------------------------------
# Timestep respacing
# ---------------------------------------------------------------------


def space_timesteps(
        num_timesteps: int,
        section_counts: Union[int, str, Sequence[int]],
) -> Set[int]:
    """Choose which of ``num_timesteps`` original steps a shortened chain keeps.

    ``section_counts`` splits the chain into equally sized sections and states
    how many steps to take from each, evenly spaced within the section. It may be

    - an ``int`` or a list of ``int`` (one entry per section),
    - a comma-separated string such as ``"10,15,20"``,
    - ``"ddimN"``, which asks for a single fixed stride ``i`` such that
      ``len(range(0, num_timesteps, i)) == N`` exactly.

    :param num_timesteps: Length of the original chain.
    :type num_timesteps: int
    :param section_counts: Section specification, as described above.
    :type section_counts: Union[int, str, Sequence[int]]
    :return: The retained original timestep indices, as a set.
    :rtype: Set[int]
    :raises ValueError: If a ``"ddimN"`` request admits no integer stride, or if
        a section is asked for more steps than it contains.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    elif isinstance(section_counts, int):
        section_counts = [section_counts]
    else:
        section_counts = [int(x) for x in section_counts]

    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps: List[int] = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps: List[int] = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


# ---------------------------------------------------------------------
