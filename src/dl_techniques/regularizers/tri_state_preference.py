"""
Triple-well regularizer that encourages weights toward {-target, 0, +target}.

This regularizer adds a penalty with three zero-gradient minima, at ``-target``,
``0`` and ``+target``, separated by two barriers. Weights sitting exactly at a
target feel no force; weights in between are pushed toward whichever target
their basin belongs to. It is a smooth, differentiable stand-in for a hard
ternary quantization constraint.

Foundational mathematics
------------------------
For target magnitude ``t`` the per-weight penalty is:

    L(w) = m * (27 / 4) * w^2 * (w - t)^2 * (w + t)^2 / t^6

Substituting ``u = (w/t)^2`` into the unnormalized form ``w^2(w^2-t^2)^2`` gives
``u(u-1)^2``, whose derivative ``(3u-1)(u-1)`` puts the interior maxima at
``u = 1/3``, that is:

    w = +/- t / sqrt(3)  ~  +/- 0.5774 t,   where the unnormalized value is 4/27.

The leading ``27/4`` is therefore the constant that makes the barrier height
exactly ``m``. NOTE: an earlier version of this module used ``32/4.5 = 64/9``,
which is ``1 / f(0.5)`` and follows from the incorrect assumption that the
maxima sit at ``+/- 0.5``. That constant overshoots the intended barrier height
by 5.35% and mislocates the watershed. The watershed is at ``0.5774 t``, not
``0.5 t``: a weight at ``0.55 t`` falls into the ZERO well, not the ``+t`` well.

Key quantities (per weight):

    barrier height, at |w| = t/sqrt(3)   = m
    L''(0)                               = 27 m / (2 t^2)
    L''(+/- t)                           = 54 m / t^2
    L2-equivalent lambda near 0          = 27 m / (4 t^2)      (6.75 m at t=1)
    L2-equivalent lambda near +/- t      = 27 m / t^2          (27 m at t=1)
    max |L'|, outer side (|w| = 0.840 t) ~ 3.73 m / t
    max |L'|, inner side (|w| = 0.307 t) ~ 2.69 m / t

Read this before using it
-------------------------
1.  THE WELLS ARE NOT EQUIVALENT. The outer wells are 4x stiffer than the zero
    well, while the zero well owns the wider basin (0.577 t per side against
    0.423 t). Both asymmetries fall out of the polynomial rather than being
    chosen, and both bias training. Do not assume the three states are treated
    symmetrically.

2.  THIS DOES NOT PREFER SPARSITY. L(0) = L(+/-t) = 0, so zero is not cheaper
    than a nonzero target. Any sparsity you observe comes from basin width plus
    initialization, not from the penalty. Add an explicit L1 term if you want a
    real sparsity bias, and accept that doing so breaks the zero-gradient
    property at +/- t.

3.  PUT THE TARGETS WHERE THE WEIGHTS ARE. Glorot/He initialization gives
    std ~ sqrt(2/fan_in), e.g. 0.09 for fan_in 256. With ``target=1.0`` every
    weight starts deep inside the zero basin and would have to climb a barrier
    of height m to reach +/-1. Nothing crosses, and the result is pure collapse
    to zero rather than ternarization. Set ``target`` to the scale the weights
    actually occupy (Ternary Weight Networks uses ~0.7 * E|w|); see
    `from_weight_scale`.

4.  ANNEAL THE MULTIPLIER. The loss is non-convex and the barriers freeze each
    weight into whichever basin its initializer landed in, before the task loss
    has a say. `multiplier` is stored as a non-trainable ``keras.Variable`` by
    default so a callback can ramp it; see `TriStatePressureScheduler`.

5.  REDUCTION MATTERS. ``sum`` (the default, matching Keras built-ins) keeps the
    per-weight gradient independent of layer size. ``mean`` normalizes the loss
    across layers but divides the per-weight gradient by the parameter count, so
    a `multiplier` tuned on a small layer silently does nothing on a large one.

6.  THE TAILS ARE SEXTIC. Gradients grow quintically outside [-t, t]: at w = 3t
    the penalty is 3888 m and the gradient 8424 m/t. This is a divergence risk,
    not a feature. ``quadratic_tails=True`` swaps in the C2-continuous quadratic
    extension ``27 m (|w| - t)^2 / t^2``, capping gradient growth at linear.

References
----------
The potential is a standard symmetric triple-well from Landau theory, not a
first-principles construction. Its use for ternary weights has direct prior art:

- Li, Zhang, Liu. "Ternary Weight Networks." 2016.
  (the 0.7 * E|w| threshold heuristic)
- Zhu, Han, Mao, Dally. "Trained Ternary Quantization." ICLR 2017.
  (learned, asymmetric target magnitudes)
- Courbariaux, Bengio, David. "BinaryConnect." NeurIPS 2015.
- Bai, Wang, Liberty. "ProxQuant." ICLR 2019.
  (piecewise-linear M-shaped penalty, bounded gradient, proximal formulation;
  better conditioned than a sextic when stability matters)
- Yin et al. "BinaryRelax." 2018.
- Louizos, Welling, Kingma. "Learning Sparse Neural Networks through L0
  Regularization." ICLR 2018. (if actual sparsity is the goal)
"""

import keras
from keras import ops
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

DEFAULT_MULTIPLIER: float = 1.0
DEFAULT_TARGET: float = 1.0
DEFAULT_REDUCTION: str = "sum"

VALID_REDUCTIONS = ("sum", "mean")

# Normalizes the barrier height to exactly 1.0 for multiplier=1.
# Derivation: max of x^2(x^2-1)^2 is 4/27 at x = 1/sqrt(3); 1 / (4/27) = 27/4.
BARRIER_NORMALIZATION: float = 27.0 / 4.0

# Curvature coefficient of the C2 quadratic tail extension beyond +/- target:
# L''(+/- t) = 54 m / t^2, so the matching quadratic is 27 m d^2 / t^2.
TAIL_COEFFICIENT: float = 27.0

# Watershed between the zero well and the outer wells, in units of `target`.
WATERSHED_FRACTION: float = 3.0 ** -0.5  # 0.57735...

# String constants for serialization
STR_MULTIPLIER: str = "multiplier"
STR_TARGET: str = "target"
STR_REDUCTION: str = "reduction"
STR_QUADRATIC_TAILS: str = "quadratic_tails"
STR_ANNEALABLE: str = "annealable"
STR_NAME: str = "name"

# Legacy key, kept only so old serialized models still load.
STR_LEGACY_SCALE: str = "scale"


# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.regularizers.tri_state_preference")
class TriStatePreferenceRegularizer(keras.regularizers.Regularizer):
    """Triple-well penalty with zero-gradient minima at -target, 0 and +target.

        L(w) = multiplier * (27/4) * w^2 (w-t)^2 (w+t)^2 / t^6,   t = target

    The barrier height is exactly `multiplier` for any `target`. Locally the
    penalty behaves like L2 pull toward the nearest target, with coefficient
    ``27*multiplier/(4*t^2)`` at zero and ``27*multiplier/t^2`` at +/- t, so the
    outer wells are four times stiffer than the central one.

    Parameters
    ----------
    multiplier : float, optional
        Barrier height and overall penalty strength. Must be non-negative.
        Check `equivalent_l2_lambda_at_zero` before picking a value; with
        ``target=1`` a multiplier of 1.0 is equivalent to L2 with lambda 6.75,
        which is very strong. Default ``DEFAULT_MULTIPLIER``.
    target : float, optional
        Magnitude of the outer targets. Wells sit at ``-target``, ``0`` and
        ``+target``. Must be positive. Match this to the actual scale of the
        weights being regularized. Default ``DEFAULT_TARGET``.
    reduction : {"sum", "mean"}, optional
        How per-weight costs are combined. ``"sum"`` matches the Keras built-in
        regularizers and keeps the per-weight gradient independent of layer
        size. ``"mean"`` divides that gradient by the parameter count.
        Default ``DEFAULT_REDUCTION``.
    quadratic_tails : bool, optional
        If True, replace the sextic growth outside [-target, target] with the
        C2-continuous quadratic ``27 * multiplier * (|w|-t)^2 / t^2``. Value,
        slope and curvature all match at the targets, so the wells are
        unchanged; only the far tails soften, from quintic to linear gradients.
        Strongly recommended. Default False (preserves the pure polynomial).
    annealable : bool, optional
        If True, `multiplier` is held in a non-trainable ``keras.Variable`` so
        it can be updated during training via `set_multiplier`. If False it is
        a Python float folded into the graph as a constant. Default True.
    name : str, optional
        Name for the multiplier variable. Ignored when ``annealable=False``.

    Raises
    ------
    ValueError
        If `multiplier` is negative, `target` is not positive, or `reduction`
        is not one of ``"sum"`` / ``"mean"``.

    Examples
    --------
    >>> # Ternary kernel, targets matched to the initializer scale, annealed in.
    >>> reg = TriStatePreferenceRegularizer.from_weight_scale(
    ...     fan_in=256, multiplier=0.0
    ... )
    >>> layer = keras.layers.Dense(64, kernel_regularizer=reg)
    >>> model.fit(..., callbacks=[TriStatePressureScheduler(reg, target=0.05)])

    >>> # Explicit targets at {-0.1, 0, 0.1}.
    >>> reg = TriStatePreferenceRegularizer(multiplier=0.01, target=0.1,
    ...                                     quadratic_tails=True)
    """

    def __init__(
        self,
        multiplier: float = DEFAULT_MULTIPLIER,
        target: float = DEFAULT_TARGET,
        reduction: str = DEFAULT_REDUCTION,
        quadratic_tails: bool = False,
        annealable: bool = True,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # `keras.regularizers.Regularizer` defines no __init__, so forwarding
        # **kwargs to super() would hit object.__init__ and raise. Accept the
        # legacy `scale` argument, reject everything else loudly.
        legacy_scale = kwargs.pop(STR_LEGACY_SCALE, None)
        if kwargs:
            raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")
        if legacy_scale is not None:
            # Old semantics: weights were pre-multiplied by `scale`, so the
            # wells actually sat at 0 and +/- 1/scale. Translate rather than
            # silently ignore, so old checkpoints keep their behaviour.
            if legacy_scale <= 0.0:
                raise ValueError(f"scale must be positive, got {legacy_scale}")
            target = 1.0 / float(legacy_scale)
            logger.warning(
                f"`scale` is deprecated; mapped scale={legacy_scale} to "
                f"target={target}. Use `target` directly. Note that the old "
                f"implementation also used a barrier constant of 32/4.5, which "
                f"was 5.35% too large; this version uses the correct 27/4, so "
                f"the effective strength drops slightly for the same multiplier."
            )

        # -- validation ---------------------------------------------------
        if multiplier < 0.0:
            raise ValueError(f"multiplier must be non-negative, got {multiplier}")
        if target <= 0.0:
            raise ValueError(f"target must be positive, got {target}")
        if reduction not in VALID_REDUCTIONS:
            raise ValueError(
                f"reduction must be one of {VALID_REDUCTIONS}, got {reduction!r}"
            )

        self.target = float(target)
        self.reduction = reduction
        self.quadratic_tails = bool(quadratic_tails)
        self.annealable = bool(annealable)
        self.name = name

        # Precomputed shape constants. t^6 normalizes the sextic core so the
        # barrier equals `multiplier`; t^2 normalizes the quadratic tails.
        self._t2 = self.target ** 2
        self._t6 = self.target ** 6

        if self.annealable:
            # Non-trainable so the optimizer ignores it; a callback assigns to
            # it mid-training to ramp the quantization pressure.
            self.multiplier = keras.Variable(
                initializer=float(multiplier),
                shape=(),
                dtype="float32",
                trainable=False,
                name=name or "tri_state_preference_multiplier",
            )
        else:
            self.multiplier = float(multiplier)

        logger.debug(
            f"Initialized TriStatePreferenceRegularizer(multiplier={multiplier}, "
            f"target={self.target}, reduction={self.reduction}, "
            f"quadratic_tails={self.quadratic_tails}); wells at "
            f"{{-{self.target:g}, 0, {self.target:g}}}, watershed at "
            f"+/-{WATERSHED_FRACTION * self.target:.4g}, L2-equivalent lambda "
            f"{BARRIER_NORMALIZATION * multiplier / self._t2:.4g} at zero and "
            f"{TAIL_COEFFICIENT * multiplier / self._t2:.4g} at +/-target"
        )

    # -- construction helpers ---------------------------------------------

    @classmethod
    def from_weight_scale(
        cls,
        fan_in: int,
        multiplier: float = 0.0,
        threshold_ratio: float = 0.7,
        gain: float = 2.0,
        **kwargs: Any,
    ) -> "TriStatePreferenceRegularizer":
        """Build with targets matched to a He/Glorot-initialized kernel.

        Placing the outer wells at ``target=1`` on a kernel whose weights have
        std ~0.09 guarantees they are never reached. This helper instead sets
        ``target = threshold_ratio * E|w|`` following Ternary Weight Networks,
        where ``E|w| = sqrt(2/pi) * std`` for a Gaussian initializer with
        ``std = sqrt(gain / fan_in)``.

        Parameters
        ----------
        fan_in : int
            Input dimension of the layer being regularized.
        multiplier : float, optional
            Initial penalty strength, default 0.0 on the assumption it will be
            annealed up from zero.
        threshold_ratio : float, optional
            Fraction of ``E|w|`` to place the outer wells at. Default 0.7.
        gain : float, optional
            Initializer variance gain: 2.0 for He, 1.0 for LeCun. Default 2.0.
        """
        if fan_in <= 0:
            raise ValueError(f"fan_in must be positive, got {fan_in}")
        std = (gain / fan_in) ** 0.5
        mean_abs = 0.7978845608 * std  # sqrt(2/pi) * std, for a Gaussian
        target = threshold_ratio * mean_abs
        kwargs.setdefault(STR_QUADRATIC_TAILS, True)
        logger.debug(
            f"from_weight_scale(fan_in={fan_in}) -> target={target:.5g}"
        )
        return cls(multiplier=multiplier, target=target, **kwargs)

    # -- runtime control ---------------------------------------------------

    def set_multiplier(self, value: float) -> None:
        """Update the penalty strength during training.

        Only available when ``annealable=True``. Intended to be called from a
        callback to ramp quantization pressure in over training.
        """
        if not self.annealable:
            raise RuntimeError(
                "multiplier is a constant; construct with annealable=True to "
                "make it updatable"
            )
        if value < 0.0:
            raise ValueError(f"multiplier must be non-negative, got {value}")
        self.multiplier.assign(float(value))

    @property
    def multiplier_value(self) -> float:
        """Current penalty strength as a Python float."""
        if self.annealable:
            return float(ops.convert_to_numpy(self.multiplier))
        return float(self.multiplier)

    @property
    def watershed(self) -> float:
        """Boundary between the zero basin and the outer basins."""
        return WATERSHED_FRACTION * self.target

    @property
    def equivalent_l2_lambda_at_zero(self) -> float:
        """L2 coefficient this is locally equivalent to inside the zero well."""
        return BARRIER_NORMALIZATION * self.multiplier_value / self._t2

    @property
    def equivalent_l2_lambda_at_target(self) -> float:
        """L2 coefficient this is locally equivalent to inside an outer well.

        Four times the value at zero: the outer wells are stiffer.
        """
        return TAIL_COEFFICIENT * self.multiplier_value / self._t2

    # -- penalty -----------------------------------------------------------

    def __call__(self, weights: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the triple-well penalty for `weights`.

        Uses the fully factored form ``w^2 (w-t)^2 (w+t)^2``. This is both the
        cheapest and the most accurate arrangement: every factor is a plain
        difference that is exact in floating point near the corresponding root,
        so there is no cancellation at the minima, which is exactly where the
        weights end up.
        """
        dtype = weights.dtype
        target = ops.cast(self.target, dtype)

        # Clip before forming the sextic so the core cannot overflow on
        # out-of-range weights (fp16 overflows around |w| > 4.6 t). Outside
        # [-t, t] the clipped core is identically zero and contributes no
        # gradient; the tail branch supplies the penalty there instead.
        w_core = ops.clip(weights, -target, target) if self.quadratic_tails else weights

        # (w^2 - t^2) collects the (w-t)(w+t) pair; squaring the product of
        # that with w gives w^2 (w-t)^2 (w+t)^2 in three multiplies.
        w_sq = ops.square(w_core)
        gap = ops.subtract(w_sq, ops.cast(self._t2, dtype))
        unit_cost = ops.divide(
            ops.multiply(
                ops.cast(BARRIER_NORMALIZATION, dtype),
                ops.multiply(w_sq, ops.square(gap)),
            ),
            ops.cast(self._t6, dtype),
        )

        if self.quadratic_tails:
            # C2-continuous extension: matches the sextic's value (0), slope (0)
            # and curvature (54 m / t^2) at +/- t, so the wells are untouched
            # and only the far tails are softened.
            overshoot = ops.subtract(ops.abs(weights), target)
            tail_cost = ops.multiply(
                ops.cast(TAIL_COEFFICIENT / self._t2, dtype),
                ops.square(overshoot),
            )
            unit_cost = ops.where(ops.greater(overshoot, 0.0), tail_cost, unit_cost)

        reduced = (
            ops.sum(unit_cost) if self.reduction == "sum" else ops.mean(unit_cost)
        )
        return ops.multiply(ops.cast(self.multiplier, dtype), reduced)

    # -- serialization -----------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments needed to rebuild this instance."""
        return {
            STR_MULTIPLIER: self.multiplier_value,
            STR_TARGET: self.target,
            STR_REDUCTION: self.reduction,
            STR_QUADRATIC_TAILS: self.quadratic_tails,
            STR_ANNEALABLE: self.annealable,
            STR_NAME: self.name,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TriStatePreferenceRegularizer":
        """Rebuild from config, tolerating configs written by the old version."""
        return cls(**dict(config))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(multiplier={self.multiplier_value:g}, "
            f"target={self.target:g}, reduction={self.reduction!r}, "
            f"quadratic_tails={self.quadratic_tails})"
        )


# ---------------------------------------------------------------------

def create_tri_state_preference_regularizer(
    multiplier: float = DEFAULT_MULTIPLIER,
    target: float = DEFAULT_TARGET,
    **kwargs: Any,
) -> TriStatePreferenceRegularizer:
    """Thin forwarder kept for interface consistency.

    Prefer constructing `TriStatePreferenceRegularizer` directly, or use
    `from_weight_scale`. All validation lives in the constructor.
    """
    return TriStatePreferenceRegularizer(
        multiplier=multiplier, target=target, **kwargs
    )


# ---------------------------------------------------------------------

class TriStatePressureScheduler(keras.callbacks.Callback):
    """Linearly ramp a regularizer's multiplier from 0 to `target`.

    Annealing is not optional here. Applied at full strength from step zero,
    the barriers freeze every weight into whichever basin its initializer
    happened to place it in, which for a zero-centered initializer is the zero
    basin for all of them. The outer wells are only reachable while the penalty
    is still weak enough for the task loss to move weights across the watershed.

    Parameters
    ----------
    regularizer : TriStatePreferenceRegularizer
        Must have been constructed with ``annealable=True``.
    target : float
        Final multiplier value. Not to be confused with the regularizer's
        `target` parameter, which is a weight magnitude.
    warmup_epochs : int
        Epochs of pure task-loss training before the ramp begins.
    ramp_epochs : int
        Epochs over which the multiplier climbs from 0 to `target`.
    """

    def __init__(
        self,
        regularizer: TriStatePreferenceRegularizer,
        target: float,
        warmup_epochs: int = 0,
        ramp_epochs: int = 10,
    ) -> None:
        super().__init__()
        if ramp_epochs <= 0:
            raise ValueError(f"ramp_epochs must be positive, got {ramp_epochs}")
        self.regularizer = regularizer
        self.target = float(target)
        self.warmup_epochs = int(warmup_epochs)
        self.ramp_epochs = int(ramp_epochs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        progress = (epoch - self.warmup_epochs) / self.ramp_epochs
        value = self.target * min(1.0, max(0.0, progress))
        self.regularizer.set_multiplier(value)
        logger.debug(f"epoch {epoch}: tri-state pressure multiplier -> {value:.4g}")


# ---------------------------------------------------------------------
