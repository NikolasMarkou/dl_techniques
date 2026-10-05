"""Config-driven construction for the Spectrum-to-Signal Principle.

:class:`SSPStrategyConfig` holds every scalar knob of both phases, and
:class:`SSPStrategy` binds the config to the four operations a pipeline actually
performs: profile a pool, select specialists, fuse them, weight a group. The factory
:func:`ssp_builder` constructs one from the ``{"type": ..., "config": {...}}`` shape
the rest of this package uses (``sled_builder``, ``deep_supervision_schedule_builder``),
so a whole SSP recipe lives in JSON like every other recipe here.

Why a strategy object at all
----------------------------
The four operations are separately useful and separately testable; the value of the
object is that it carries ONE set of defaults for all of them. The failure this
prevails is specific and has no other guard: a pipeline profiles pools at
``pass@16`` and then samples the training set at ``pass@8``, because the two numbers
were written at different times by different code paths, and nobody notices until the
sampler and the profiler disagree about what "broad" means.

Scope
-----
This module orchestrates and configures. It does not train, sample rollouts, verify
outputs, or write checkpoints -- there is no SSP trainer in this repository, and the
strategy object is a set of configured functions rather than a pipeline that runs
one.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import keras
import numpy as np

from dl_techniques.utils.constants import CONFIG_STR, TYPE_STR
from dl_techniques.utils.logger import logger

from ..constants import (
    DEFAULT_MGPO_CLIP_EPS,
    DEFAULT_MGPO_DDOF,
    DEFAULT_MGPO_EPS,
    DEFAULT_MGPO_LAMBDA,
    DEFAULT_MGPO_NORMALIZE_WEIGHTS,
    DEFAULT_MGPO_P0,
    DEFAULT_SSP_ENABLED,
    DEFAULT_SSP_ESTIMATOR,
    DEFAULT_SSP_FUSION_COEFFICIENT,
    DEFAULT_SSP_FUSION_MODE,
    DEFAULT_SSP_FUSION_SCHEME,
    DEFAULT_SSP_FUSION_TEMPERATURE,
    DEFAULT_SSP_PASS_AT_K,
    DEFAULT_SSP_SAMPLING_MODE,
    DEFAULT_SSP_TOKEN_REDUCTION,
    DEFAULT_SSP_ZERO_VARIANCE,
)
from .fusion import (
    FUSION_MODES,
    FUSION_WEIGHT_SCHEMES,
    fusion_weights_from_scores,
    fuse_specialists,
    greedy_soup,
)
from .signal import (
    TOKEN_REDUCTIONS,
    ZERO_VARIANCE_POLICIES,
    group_relative_advantages,
    max_entropy_weight,
    mgpo_advantages,
)
from .spectrum import (
    SAMPLING_MODES,
    select_specialists,
    spectrum_profile,
    spectrum_sampling_weights,
)

# ---------------------------------------------------------------------


class SSPType(str, Enum):
    """Enumeration of available SSP strategy versions."""

    SSP_V1 = "ssp_v1"


# ---------------------------------------------------------------------


class SSPStrategyConfig:
    """Scalar configuration for both phases of the Spectrum-to-Signal Principle.

    Holds only plain scalars so it round-trips cleanly through ``get_config`` /
    ``from_config``, following the ``WWTailConfig`` precedent in this package.

    Spectrum-phase args:
        enable: Master switch. Default ``False`` (the repo's opt-in convention):
            when ``False`` every operation degenerates to its pass@1 / unweighted
            form, so a recipe can carry the config unconditionally.
        pass_at_k: The ``k`` used when profiling a candidate pool. Default ``8``.
        estimator: ``"unbiased"`` or ``"plug_in"``; see
            :mod:`dl_techniques.metrics.pass_at_k` for why the two differ and why
            quoting both is a false comparison.
        fusion_mode: ``"linear"`` or ``"task_arithmetic"``. The two coincide when
            the specialists share a base and the coefficient is 1.
        fusion_temperature: Softness of ``score_softmax`` fusion weights. Larger is
            closer to uniform.
        fusion_coefficient: Task-arithmetic scale ``c``.
        sampling_mode: How coverage becomes a sampling distribution over the
            training set; see :data:`ssp.spectrum.SAMPLING_MODES`.

    Signal-phase args:
        mgpo_lambda: Max-entropy sharpening. ``0.0`` disables the weighting
            entirely, recovering the unweighted group-relative advantage exactly.
        mgpo_p0: Target success probability, strictly inside ``(0, 1)``.
        mgpo_eps: Advantage denominator stabilizer; also what keeps a
            zero-variance group from dividing by zero.
        mgpo_ddof: Delta degrees of freedom for the group standard deviation.
        mgpo_normalize_weights: Rescale weights to mean 1 so the reweighting is
            purely relative. Changes the batch's advantage scale; see
            :func:`dl_techniques.optimization.ssp.signal.mgpo_advantages`.
        clip_eps: PPO clip range half-width, used by
            :func:`dl_techniques.optimization.ssp.signal.mgpo_surrogate`.
        token_reduction: The paper's ``"per_sequence_mean"``, or the flatter
            ``"per_token_mean"`` / unreduced ``"none"``.
        zero_variance: ``"zero"`` or ``"raise"`` for a group whose rollouts all
            earn the same reward.

    Args:
        enable: See ``enable`` above.
        pass_at_k: See ``pass_at_k`` above.
        estimator: See ``estimator`` above.
        fusion_mode: See ``fusion_mode`` above.
        fusion_weight_scheme: ``"uniform"`` (the paper's ``1 / N``) or
            ``"score_softmax"``. A separate field from ``sampling_mode`` on
            purpose: one picks how the SPECIALISTS are weighted, the other how the
            TRAINING SET is sampled, and deriving one from the other couples two
            decisions that are independent.
        fusion_temperature: See ``fusion_temperature`` above.
        fusion_coefficient: See ``fusion_coefficient`` above.
        sampling_mode: See ``sampling_mode`` above.
        mgpo_lambda: See ``mgpo_lambda`` above.
        mgpo_p0: See ``mgpo_p0`` above.
        mgpo_eps: See ``mgpo_eps`` above.
        mgpo_ddof: See ``mgpo_ddof`` above.
        mgpo_normalize_weights: See ``mgpo_normalize_weights`` above.
        clip_eps: See ``clip_eps`` above.
        token_reduction: See ``token_reduction`` above.
        zero_variance: See ``zero_variance`` above.

    Raises:
        ValueError: If any enumerated field names an unknown member, or a numeric
            field is outside its domain. Validation is at construction so a bad
            recipe fails when it is PARSED rather than three stages into a run.

    Example:
        >>> from dl_techniques.optimization.ssp import SSPStrategyConfig
        >>> cfg = SSPStrategyConfig(enable=True, pass_at_k=16, mgpo_lambda=2.0)
        >>> cfg.pass_at_k, cfg.mgpo_lambda
        (16, 2.0)
    """

    def __init__(
            self,
            enable: bool = DEFAULT_SSP_ENABLED,
            pass_at_k: int = DEFAULT_SSP_PASS_AT_K,
            estimator: str = DEFAULT_SSP_ESTIMATOR,
            fusion_mode: str = DEFAULT_SSP_FUSION_MODE,
            fusion_weight_scheme: str = DEFAULT_SSP_FUSION_SCHEME,
            fusion_temperature: float = DEFAULT_SSP_FUSION_TEMPERATURE,
            fusion_coefficient: float = DEFAULT_SSP_FUSION_COEFFICIENT,
            sampling_mode: str = DEFAULT_SSP_SAMPLING_MODE,
            mgpo_lambda: float = DEFAULT_MGPO_LAMBDA,
            mgpo_p0: float = DEFAULT_MGPO_P0,
            mgpo_eps: float = DEFAULT_MGPO_EPS,
            mgpo_ddof: int = DEFAULT_MGPO_DDOF,
            mgpo_normalize_weights: bool = DEFAULT_MGPO_NORMALIZE_WEIGHTS,
            clip_eps: float = DEFAULT_MGPO_CLIP_EPS,
            token_reduction: str = DEFAULT_SSP_TOKEN_REDUCTION,
            zero_variance: str = DEFAULT_SSP_ZERO_VARIANCE,
    ) -> None:
        self.enable = bool(enable)
        self.pass_at_k = int(pass_at_k)
        self.estimator = _one_of(estimator, ("unbiased", "plug_in"), "estimator")
        self.fusion_mode = _one_of(fusion_mode, FUSION_MODES, "fusion_mode")
        self.fusion_weight_scheme = _one_of(
            fusion_weight_scheme, FUSION_WEIGHT_SCHEMES, "fusion_weight_scheme"
        )
        self.fusion_temperature = _positive(fusion_temperature, "fusion_temperature")
        self.fusion_coefficient = float(fusion_coefficient)
        self.sampling_mode = _one_of(sampling_mode, SAMPLING_MODES, "sampling_mode")
        self.mgpo_lambda = _non_negative(mgpo_lambda, "mgpo_lambda")
        self.mgpo_p0 = _open_unit_interval(mgpo_p0, "mgpo_p0")
        self.mgpo_eps = _positive(mgpo_eps, "mgpo_eps")
        self.mgpo_ddof = _non_negative_int(mgpo_ddof, "mgpo_ddof")
        self.mgpo_normalize_weights = bool(mgpo_normalize_weights)
        self.clip_eps = _positive(clip_eps, "clip_eps")
        self.token_reduction = _one_of(
            token_reduction, TOKEN_REDUCTIONS, "token_reduction"
        )
        self.zero_variance = _one_of(
            zero_variance, ZERO_VARIANCE_POLICIES, "zero_variance"
        )

        if self.pass_at_k < 1:
            raise ValueError(f"pass_at_k must be >= 1; got {self.pass_at_k}")

    @property
    def ks(self) -> List[int]:
        """The default ``k`` ladder implied by ``pass_at_k``: powers of two up to it.

        Returns:
            ``[1, 2, 4, ..., pass_at_k]``, de-duplicated, ascending. Used by
            :meth:`SSPStrategy.profile` when no explicit ladder is given, so the
            headline ``pass_at_k`` and the curve share one ``k``.
        """
        return sorted({2**e for e in range(0, int(np.floor(np.log2(self.pass_at_k))) + 1)})

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments needed to rebuild this config.

        :return: A dict carrying every constructor argument.
        """
        return {
            "enable": self.enable,
            "pass_at_k": self.pass_at_k,
            "estimator": self.estimator,
            "fusion_mode": self.fusion_mode,
            "fusion_weight_scheme": self.fusion_weight_scheme,
            "fusion_temperature": self.fusion_temperature,
            "fusion_coefficient": self.fusion_coefficient,
            "sampling_mode": self.sampling_mode,
            "mgpo_lambda": self.mgpo_lambda,
            "mgpo_p0": self.mgpo_p0,
            "mgpo_eps": self.mgpo_eps,
            "mgpo_ddof": self.mgpo_ddof,
            "mgpo_normalize_weights": self.mgpo_normalize_weights,
            "clip_eps": self.clip_eps,
            "token_reduction": self.token_reduction,
            "zero_variance": self.zero_variance,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SSPStrategyConfig":
        """Rebuild a config from :meth:`get_config` output.

        :param config: A dict of constructor arguments.
        :return: The reconstructed :class:`SSPStrategyConfig`.
        :raises TypeError: If ``config`` is not a dict.
        """
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dict; got {type(config).__name__}")
        return cls(**config)


# ---------------------------------------------------------------------
# field validators (construction-time, so a bad recipe fails when parsed)
# ---------------------------------------------------------------------


def _one_of(value: str, allowed: Sequence[str], name: str) -> str:
    """Normalise and validate a string field against a closed set.

    Args:
        value: The candidate value; case and surrounding whitespace are tolerated.
        allowed: The permitted lower-case members.
        name: Field name, used in the error message.

    Returns:
        The lower-cased value.

    :raises ValueError: If the value is not a string, or is not a member.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"{name} must be a string; got {type(value).__name__} ({value!r})"
        )
    text = value.strip().lower()
    if text not in allowed:
        raise ValueError(
            f"Unknown {name} {value!r}. Supported: {list(allowed)}"
        )
    return text


def _positive(value: float, name: str) -> float:
    """Validate a strictly positive, finite float field.

    :param value: The candidate value.
    :param name: Field name, used in the error message.
    :return: ``value`` as a ``float``.
    :raises ValueError: If it is not finite and strictly positive.
    """
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and > 0; got {value!r}")
    return number


def _non_negative(value: float, name: str) -> float:
    """Validate a non-negative, finite float field.

    :param value: The candidate value.
    :param name: Field name, used in the error message.
    :return: ``value`` as a ``float``.
    :raises ValueError: If it is not finite and non-negative.
    """
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and >= 0; got {value!r}")
    return number


def _non_negative_int(value: int, name: str) -> int:
    """Validate a non-negative integer field.

    :param value: The candidate value.
    :param name: Field name, used in the error message.
    :return: ``value`` as an ``int``.
    :raises ValueError: If it is not a non-negative integer.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"{name} must be an integer; got {type(value).__name__} ({value!r})"
        )
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be >= 0; got {number}")
    return number


def _open_unit_interval(value: float, name: str) -> float:
    """Validate a float field lying strictly inside ``(0, 1)``.

    :param value: The candidate value.
    :param name: Field name, used in the error message.
    :return: ``value`` as a ``float``.
    :raises ValueError: If it is not strictly inside ``(0, 1)``.
    """
    number = float(value)
    if not 0.0 < number < 1.0:
        raise ValueError(
            f"{name} must lie strictly inside (0, 1); got {value!r}. The endpoints "
            f"are degenerate Bernoullis, whose KL against a non-degenerate one is "
            f"+inf, so every weight would collapse to zero."
        )
    return number


# ---------------------------------------------------------------------


class SSPStrategy:
    """A configured Spectrum-to-Signal Principle, as four callable operations.

    Construct one through :func:`ssp_builder` rather than directly, so the config
    dict shape is the single entry point.

    The master switch has a specific meaning, stated because "off" must be a no-op
    and not a silent behaviour change: with ``config.enable == False``,
    :meth:`profile` reports ``pass@1`` as its headline ``pass_at_k``, :meth:`weights`
    returns ones, and :meth:`advantages` is the plain group-relative advantage. So a
    recipe can carry an ``SSPStrategy`` unconditionally and flip one flag to A/B the
    principle against itself.

    Args:
        config: The :class:`SSPStrategyConfig` governing every operation.

    Example:
        >>> import numpy as np
        >>> from dl_techniques.optimization.ssp import ssp_builder
        >>> ssp = ssp_builder({"type": "ssp_v1",
        ...                    "config": {"enable": True, "pass_at_k": 4}})
        >>> rewards = np.array([[1., 1., 0., 0.]], dtype="float32")   # p_c = 0.5
        >>> adv_on = np.asarray(ssp.advantages(rewards))
        >>> adv_off = np.asarray(ssp_builder({"type": "ssp_v1", "config": {}}).advantages(rewards))
        >>> bool(np.allclose(adv_on, adv_off))     # p_c is already 0.5 -> weight 1
        True
    """

    def __init__(self, config: SSPStrategyConfig) -> None:
        if not isinstance(config, SSPStrategyConfig):
            raise TypeError(
                f"config must be an SSPStrategyConfig; got {type(config).__name__}"
            )
        self.config = config
        logger.info(
            f"SSPStrategy: enable={config.enable}, pass_at_k={config.pass_at_k}, "
            f"estimator={config.estimator}, fusion_mode={config.fusion_mode}, "
            f"mgpo_lambda={config.mgpo_lambda}, "
            f"mgpo_normalize_weights={config.mgpo_normalize_weights}"
        )

    # -- Spectrum phase ------------------------------------------------

    def profile(self, outcomes, ks: Optional[Sequence[int]] = None) -> Dict[str, object]:
        """Profile one candidate pool. See :func:`ssp.spectrum.spectrum_profile`.

        :param outcomes: ``(n_problems, n_samples)`` outcome matrix.
        :param ks: Optional ``k`` ladder; defaults to the config's power-of-two
            ladder up to ``config.pass_at_k``, clipped to ``n_samples``.
        :return: The profile dict.
        """
        ladder = list(ks) if ks is not None else [
            k for k in self.config.ks if k <= int(np.asarray(outcomes).shape[1])
        ] or None
        return spectrum_profile(
            outcomes, ks=ladder, estimator=self.config.estimator
        )

    def select(self, score_matrix, subdomains: Optional[Sequence[str]] = None):
        """Select one specialist per subdomain. See :func:`ssp.spectrum.select_specialists`.

        :param score_matrix: ``(n_checkpoints, n_subdomains)`` scores, higher is broader.
        :param subdomains: Optional per-column labels.
        :return: The selection dict.
        """
        return select_specialists(score_matrix, subdomains=subdomains)

    def fusion_weights(self, scores: Sequence[float]) -> np.ndarray:
        """Fusion weights from measured specialist scores.

        Reads ``config.fusion_weight_scheme`` and nothing else. In particular it
        does NOT read ``config.sampling_mode``: that field chooses how the TRAINING
        SET is sampled, this one chooses how the SPECIALISTS are weighted, and
        deriving either from the other couples two independent decisions behind one
        flag. With ``enable=False`` the scheme is forced to ``"uniform"``, so the
        OFF path is the paper's ``1 / N``.

        :param scores: One score per specialist, in specialist order.
        :return: A ``(N,)`` ``float64`` weight vector summing to 1.
        :raises ValueError: On any problem with
            :func:`dl_techniques.optimization.ssp.fusion.fusion_weights_from_scores`.
        """
        scheme = self.config.fusion_weight_scheme if self.config.enable else "uniform"
        return fusion_weights_from_scores(
            scores, scheme=scheme, temperature=self.config.fusion_temperature
        )

    def fuse(self, models_or_weights, weights: Optional[Sequence[float]] = None,
             base=None):
        """Fuse specialists into one weight set. See :func:`ssp.fusion.fuse_specialists`.

        :param models_or_weights: The specialists.
        :param weights: Optional explicit weights; defaults to uniform.
        :param base: Shared ancestor weights for ``task_arithmetic``.
        :return: A list of NEW numpy arrays. Nothing is mutated.
        """
        return fuse_specialists(
            models_or_weights,
            weights=weights,
            mode=self.config.fusion_mode,
            base=base,
            coefficient=self.config.fusion_coefficient,
        )

    def soup(self, pool, score_fn, max_size: Optional[int] = None):
        """Greedy soup over a candidate pool. See :func:`ssp.fusion.greedy_soup`.

        :param pool: Candidate specialists.
        :param score_fn: Callable returning a float to MAXIMISE.
        :param max_size: Optional cap on the number selected.
        :return: A ``(indices, weights)`` tuple.
        """
        return greedy_soup(pool, score_fn=score_fn, max_size=max_size)

    def sampling_weights(self, coverage) -> np.ndarray:
        """Coverage -> sampling distribution. See :func:`ssp.spectrum.spectrum_sampling_weights`.

        :param coverage: Per-item coverage in ``[0, 1]``.
        :return: A ``(N,)`` ``float64`` probability vector.
        """
        if not self.config.enable:
            return np.full(np.asarray(coverage).shape, 1.0 / np.asarray(coverage).size)
        return spectrum_sampling_weights(
            coverage,
            mode=self.config.sampling_mode,
            lam=self.config.mgpo_lambda,
            p0=self.config.mgpo_p0,
        )

    # -- Signal phase ---------------------------------------------------

    def weights(self, p_c):
        """The max-entropy weight for each group's success rate.

        :param p_c: Per-group success rates in ``[0, 1]``.
        :return: A ``keras`` tensor of weights; all ones when ``enable`` is False.
        :raises ValueError: On any problem with
            :func:`dl_techniques.optimization.ssp.signal.max_entropy_weight`.
        """
        values = keras.ops.convert_to_tensor(p_c, dtype="float32")
        if not self.config.enable:
            return keras.ops.ones_like(values)
        return max_entropy_weight(
            values, lam=self.config.mgpo_lambda, p0=self.config.mgpo_p0
        )

    def advantages(self, rewards, group_axis: int = -1):
        """MGPO-weighted group-relative advantages.

        See :func:`ssp.signal.mgpo_advantages`. With ``enable=False`` this is the
        unweighted group-relative advantage, which is the A/B control the principle
        is meant to be measured against.

        :param rewards: Rollout rewards in ``[0, 1]``.
        :param group_axis: The axis holding each query's rollouts.
        :return: A ``keras`` tensor shaped like ``rewards``.
        :raises ValueError: On any problem with
            :func:`dl_techniques.optimization.ssp.signal.mgpo_advantages`.
        """
        values = keras.ops.convert_to_tensor(rewards, dtype="float32")
        if not self.config.enable:
            return group_relative_advantages(
                values,
                eps=self.config.mgpo_eps,
                ddof=self.config.mgpo_ddof,
                group_axis=group_axis,
                zero_variance=self.config.zero_variance,
            )
        return mgpo_advantages(
            values,
            lam=self.config.mgpo_lambda,
            p0=self.config.mgpo_p0,
            eps=self.config.mgpo_eps,
            ddof=self.config.mgpo_ddof,
            group_axis=group_axis,
            zero_variance=self.config.zero_variance,
            normalize_weights=self.config.mgpo_normalize_weights,
        )

    def get_config(self) -> Dict[str, Any]:
        """Return this strategy's config.

        :return: The :class:`SSPStrategyConfig`'s ``get_config()`` dict.
        """
        return self.config.get_config()


# ---------------------------------------------------------------------


def ssp_builder(config: Dict[str, Any]) -> SSPStrategy:
    """Build an :class:`SSPStrategy` from a configuration dictionary.

    Args:
        config: A dict of the shape
            ``{"type": "ssp_v1", "config": {<SSPStrategyConfig kwargs>}}``.
            ``"config"`` may be omitted and every field then takes its default --
            which is the OFF configuration, so the no-arg form is a strict no-op
            rather than a partially-enabled one.

    Returns:
        A configured :class:`SSPStrategy`.

    Raises:
        TypeError: If ``config`` is not a dict, or ``"config"`` is present and not a
            dict.
        ValueError: If ``"type"`` is missing, not a string, or unknown; or if any
            field inside ``"config"`` fails :class:`SSPStrategyConfig` validation.

    Example:
        >>> from dl_techniques.optimization.ssp import ssp_builder
        >>> ssp = ssp_builder({
        ...     "type": "ssp_v1",
        ...     "config": {"enable": True, "pass_at_k": 16, "mgpo_lambda": 2.0},
        ... })
        >>> ssp.config.pass_at_k
        16

    Note:
        An unknown key inside ``"config"`` raises ``TypeError`` from
        :meth:`SSPStrategyConfig.from_config` rather than being ignored. A silently
        dropped ``mgpo_lamda`` typo would leave the weighting at its default and
        read as "SSP did not help".
    """
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dictionary; got {type(config).__name__}")

    ssp_type = config.get(TYPE_STR)
    if ssp_type is None:
        raise ValueError(
            f"SSP type cannot be None - must specify {TYPE_STR!r} in config"
        )
    if not isinstance(ssp_type, str):
        raise TypeError(f"SSP type must be a string; got {type(ssp_type).__name__}")

    ssp_type = ssp_type.strip().lower()
    if ssp_type != SSPType.SSP_V1:
        raise ValueError(
            f"Unknown SSP type: [{ssp_type}]. "
            f"Supported types: {[member.value for member in SSPType]}"
        )

    params = config.get(CONFIG_STR, {})
    if not isinstance(params, dict):
        raise TypeError(
            f"{CONFIG_STR!r} must be a dictionary containing SSP parameters; got "
            f"{type(params).__name__}"
        )

    logger.info(f"Building SSP strategy: type=[{ssp_type}], params=[{params}]")
    strategy_config = SSPStrategyConfig.from_config(params)
    return SSPStrategy(strategy_config)
