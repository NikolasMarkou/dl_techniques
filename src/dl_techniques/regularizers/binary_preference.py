"""
Double-well regularizer that encourages weights toward two target values.

This regularizer adds a penalty with two zero-gradient minima, at `low` and
`high`, and a barrier between them. Weights sitting exactly at a target feel
no force; weights between the targets are pushed toward whichever one they
are closer to. It is a smooth, differentiable stand-in for a hard binarizing
constraint.

Foundational mathematics
------------------------
For half-gap ``h = (high - low) / 2`` the per-weight penalty is:

    L(w) = m * (w - low)^2 * (w - high)^2 / h^4

For the canonical ``low=0, high=1`` case this reduces to ``L(w) = 16 m w^2 (1-w)^2``,
the standard Cahn-Hilliard / Ginzburg-Landau double-well potential. The
normalization by ``h^4`` fixes the barrier height at the midpoint to exactly
``m``, independent of the chosen targets, so `multiplier` means the same thing
whether the targets are {0, 1} or {-1, +1}.

Key quantities (per weight):

    L'(w)                      = 2m (w-low)(w-high)(2w-low-high) / h^4
    L(midpoint)                = m                      (barrier height)
    L''(low) = L''(high)       = 8m / h^2               (curvature at minima)
    max |L'|                   ~ 3.08 m / h             (steepest slope)
    local L2-equivalent lambda = 4m / h^2

That last line is the number to reason about when picking `multiplier`. Near a
minimum this regularizer behaves exactly like L2 pull toward that target with
``lambda = 4m/h^2``. For {0, 1} targets that is ``16 m``, so ``multiplier=1.0``
is a very strong penalty, not a mild one.

Read this before using it
-------------------------
1.  DO NOT attach the {0, 1} configuration to a `Dense` or `Conv2D` kernel.
    Glorot/He initializers are zero-centered, so every weight starts in the
    ``w=0`` well and cannot climb the barrier to reach ``w=1``. The result is
    not binarization, it is aggressive L2 that collapses the layer to zero and
    forbids negative weights. For kernels use ``low=-1.0, high=1.0``
    (see `for_bipolar_weights`). For gates and masks initialized inside
    [0, 1] the {0, 1} configuration is the right one (see `for_gates`).

2.  ANNEAL THE MULTIPLIER. The total loss is non-convex: the barrier means
    each weight's well assignment gets frozen early by initialization noise.
    Standard practice is to ramp `multiplier` from 0 over training, which is
    why it is stored as a non-trainable `keras.Variable` by default and can be
    updated from a callback via `set_multiplier`.

3.  REDUCTION MATTERS. `sum` (the default, matching Keras built-ins) keeps the
    per-weight gradient independent of layer size. `mean` normalizes the loss
    across layers but divides the per-weight gradient by the parameter count,
    so a `multiplier` tuned on a small layer silently does nothing on a large
    one. Choose deliberately.

4.  The tails are quartic and unbounded: gradients grow cubically outside
    [low, high]. Set ``quadratic_tails=True`` to swap in the C2-continuous
    quadratic extension beyond the targets, which caps gradient growth at
    linear and is much safer without gradient clipping.

References
----------
The potential itself is the classic double-well of Ginzburg-Landau /
Cahn-Hilliard theory, not a novel construction. Its use for weight
binarization has direct prior art:

- Courbariaux, Bengio, David. "BinaryConnect." NeurIPS 2015.
- Hubara et al. "Binarized Neural Networks." NeurIPS 2016.
- Darabi et al. "BNN+: Improved Binary Network Training." 2018.
  (explicit double-well regularizer with minima at +/-1)
- Bai, Wang, Liberty. "ProxQuant." ICLR 2019.
  (W-shaped penalty, bounded gradient, proximal formulation; preferable to a
  quartic when stability matters)
- Louizos, Welling, Kingma. "Learning Sparse Neural Networks through L0
  Regularization." ICLR 2018. (the gate reparameterization this pairs with)
- Bengio, Leonard, Courville. "Estimating or Propagating Gradients Through
  Stochastic Neurons." 2013. (straight-through estimator)
"""

import keras
from keras import ops
from typing import Any, Dict, Optional, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

DEFAULT_MULTIPLIER: float = 1.0
DEFAULT_LOW: float = 0.0
DEFAULT_HIGH: float = 1.0
DEFAULT_REDUCTION: str = "sum"

VALID_REDUCTIONS = ("sum", "mean")

# String constants for serialization
STR_MULTIPLIER: str = "multiplier"
STR_LOW: str = "low"
STR_HIGH: str = "high"
STR_REDUCTION: str = "reduction"
STR_QUADRATIC_TAILS: str = "quadratic_tails"
STR_ANNEALABLE: str = "annealable"
STR_NAME: str = "name"

# Legacy key, kept only so old serialized models still load.
STR_LEGACY_SCALE: str = "scale"


# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.regularizers.binary_preference")
class BinaryPreferenceRegularizer(keras.regularizers.Regularizer):
    """Double-well penalty with zero-gradient minima at `low` and `high`.

        L(w) = multiplier * (w - low)^2 * (w - high)^2 / h^4,   h = (high - low) / 2

    The barrier height at the midpoint equals `multiplier` for any choice of
    targets. Curvature at each minimum is ``8 * multiplier / h^2``, i.e. the
    local behaviour is L2-toward-the-target with ``lambda = 4 * multiplier / h^2``
    (``16 * multiplier`` for the default {0, 1} targets).

    Parameters
    ----------
    multiplier : float, optional
        Barrier height, and the overall strength of the penalty. Must be
        non-negative. Note the L2-equivalence above before picking a value;
        1.0 is strong. Default ``DEFAULT_MULTIPLIER``.
    low : float, optional
        Lower target value, a zero of the penalty. Default ``DEFAULT_LOW``.
    high : float, optional
        Upper target value, a zero of the penalty. Must exceed `low`.
        Default ``DEFAULT_HIGH``.
    reduction : {"sum", "mean"}, optional
        How per-weight costs are combined. ``"sum"`` matches the Keras
        built-in regularizers and keeps the per-weight gradient independent of
        layer size. ``"mean"`` divides that gradient by the parameter count.
        Default ``DEFAULT_REDUCTION``.
    quadratic_tails : bool, optional
        If True, replace the quartic growth outside [low, high] with the
        C2-continuous quadratic ``4 * multiplier * d^2 / h^2`` (where ``d`` is
        the signed distance past the nearer target). Value, slope and
        curvature all match at the targets, so the well shape is untouched;
        only the far tails change, from cubic to linear gradients. Recommended
        when weights can leave the target interval. Default False.
    annealable : bool, optional
        If True, `multiplier` is held in a non-trainable ``keras.Variable`` so
        it can be updated during training via `set_multiplier`. If False it is
        a Python float folded into the graph as a constant. Default True.
    name : str, optional
        Name for the multiplier variable. Ignored when ``annealable=False``.

    Raises
    ------
    ValueError
        If `multiplier` is negative, `high` is not greater than `low`, or
        `reduction` is not one of ``"sum"`` / ``"mean"``.

    Warnings
    --------
    The default {0, 1} targets are intended for gates and masks initialized
    inside [0, 1], NOT for zero-centered layer kernels. Applied to a
    Glorot-initialized kernel this cannot binarize (no weight can cross the
    barrier to reach 1) and instead acts as strong L2 with a hard floor at
    zero. Use `for_bipolar_weights` for kernels.

    Examples
    --------
    >>> # Learnable feature-selection gates, pressure annealed in by a callback.
    >>> reg = BinaryPreferenceRegularizer.for_gates(multiplier=0.0)
    >>> gate = layer.add_weight(
    ...     shape=(units,), initializer="random_uniform", regularizer=reg
    ... )
    >>> reg.set_multiplier(0.5)  # call from on_epoch_begin

    >>> # Binarizing a kernel toward {-1, +1}, with safe tails.
    >>> reg = BinaryPreferenceRegularizer.for_bipolar_weights(multiplier=0.01)
    >>> layer = keras.layers.Dense(64, kernel_regularizer=reg)
    """

    def __init__(
        self,
        multiplier: float = DEFAULT_MULTIPLIER,
        low: float = DEFAULT_LOW,
        high: float = DEFAULT_HIGH,
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
            raise TypeError(
                f"unexpected keyword arguments: {sorted(kwargs)}"
            )
        if legacy_scale is not None:
            # Old semantics: zeros at 0 and 1/scale. Translate, do not silently
            # ignore, so previously serialized models keep their behaviour.
            if legacy_scale <= 0.0:
                raise ValueError(f"scale must be positive, got {legacy_scale}")
            low, high = 0.0, 1.0 / float(legacy_scale)
            logger.warning(
                f"`scale` is deprecated; mapped scale={legacy_scale} to "
                f"low=0.0, high={high}. Use `low`/`high` directly."
            )

        # -- validation ---------------------------------------------------
        if multiplier < 0.0:
            raise ValueError(f"multiplier must be non-negative, got {multiplier}")
        if high <= low:
            raise ValueError(
                f"high must be strictly greater than low, got low={low}, high={high}"
            )
        if reduction not in VALID_REDUCTIONS:
            raise ValueError(
                f"reduction must be one of {VALID_REDUCTIONS}, got {reduction!r}"
            )

        self.low = float(low)
        self.high = float(high)
        self.reduction = reduction
        self.quadratic_tails = bool(quadratic_tails)
        self.annealable = bool(annealable)
        self.name = name

        # Precomputed shape constants. h is the half-gap; h^4 normalizes the
        # barrier height to `multiplier`, h^2 normalizes the quadratic tails.
        half_gap = (self.high - self.low) / 2.0
        self._h2 = half_gap ** 2
        self._h4 = half_gap ** 4

        if self.annealable:
            # Non-trainable so the optimizer ignores it; a callback can assign
            # to it mid-training to ramp the binarization pressure.
            self.multiplier = keras.Variable(
                initializer=float(multiplier),
                shape=(),
                dtype="float32",
                trainable=False,
                name=name or "binary_preference_multiplier",
            )
        else:
            self.multiplier = float(multiplier)

        logger.debug(
            f"Initialized BinaryPreferenceRegularizer(multiplier={multiplier}, "
            f"low={self.low}, high={self.high}, reduction={self.reduction}, "
            f"quadratic_tails={self.quadratic_tails}) -> "
            f"local L2-equivalent lambda={4.0 * multiplier / self._h2:.4g}"
        )

    # -- construction helpers ---------------------------------------------

    @classmethod
    def for_gates(
        cls, multiplier: float = 0.0, **kwargs: Any
    ) -> "BinaryPreferenceRegularizer":
        """Preset for {0, 1} gates or masks initialized inside [0, 1].

        Uses ``reduction="mean"`` so the loss contribution does not scale with
        the number of gates, and defaults `multiplier` to 0.0 on the assumption
        that it will be annealed up from zero.
        """
        kwargs.setdefault(STR_REDUCTION, "mean")
        kwargs.setdefault(STR_QUADRATIC_TAILS, True)
        return cls(multiplier=multiplier, low=0.0, high=1.0, **kwargs)

    @classmethod
    def for_bipolar_weights(
        cls, multiplier: float = 0.0, **kwargs: Any
    ) -> "BinaryPreferenceRegularizer":
        """Preset for {-1, +1} kernel binarization.

        This is the configuration to use on `Dense` / `Conv2D` kernels: it is
        symmetric about zero, so a standard zero-centered initializer places
        weights at the top of the barrier and lets task gradients decide which
        well each one falls into. Half-gap is 1, so the local L2-equivalent
        lambda is ``4 * multiplier``.
        """
        kwargs.setdefault(STR_QUADRATIC_TAILS, True)
        return cls(multiplier=multiplier, low=-1.0, high=1.0, **kwargs)

    # -- runtime control ---------------------------------------------------

    def set_multiplier(self, value: float) -> None:
        """Update the penalty strength during training.

        Only available when ``annealable=True``. Intended to be called from a
        callback to ramp binarization pressure in over training, e.g.::

            reg.set_multiplier(target * min(1.0, epoch / warmup_epochs))
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
    def equivalent_l2_lambda(self) -> float:
        """L2 coefficient this is locally equivalent to near either minimum."""
        return 4.0 * self.multiplier_value / self._h2

    # -- penalty -----------------------------------------------------------

    def __call__(self, weights: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the double-well penalty for `weights`.

        Uses the factored form ``((w-low)(w-high))^2 / h^4`` rather than the
        algebraically identical ``(1 - (w-c)^2/h^2)^2``. Both are the same
        polynomial, but the factored one is cheaper and, more importantly, has
        no subtractive cancellation: the expanded form subtracts near-equal
        floats right at the minima, which is exactly where the weights end up.
        """
        dtype = weights.dtype
        low = ops.cast(self.low, dtype)
        high = ops.cast(self.high, dtype)

        # Clip before forming the quartic so the core term cannot overflow on
        # out-of-range weights (relevant in fp16). Outside [low, high] the
        # clipped core is identically zero and contributes no gradient; the
        # tail branch below supplies the penalty there instead.
        w_core = ops.clip(weights, low, high) if self.quadratic_tails else weights

        d_low_core = ops.subtract(w_core, low)
        d_high_core = ops.subtract(w_core, high)
        unit_cost = ops.divide(
            ops.square(ops.multiply(d_low_core, d_high_core)),
            ops.cast(self._h4, dtype),
        )

        if self.quadratic_tails:
            # C2-continuous extension: matches the quartic's value (0), slope
            # (0) and curvature (8m/h^2) at each target, so the well is
            # unchanged and only the far tails are softened.
            inv_h2 = ops.cast(4.0 / self._h2, dtype)
            below = ops.multiply(inv_h2, ops.square(ops.subtract(weights, low)))
            above = ops.multiply(inv_h2, ops.square(ops.subtract(weights, high)))
            unit_cost = ops.where(
                ops.less(weights, low),
                below,
                ops.where(ops.greater(weights, high), above, unit_cost),
            )

        reduced = (
            ops.sum(unit_cost) if self.reduction == "sum" else ops.mean(unit_cost)
        )
        return ops.multiply(ops.cast(self.multiplier, dtype), reduced)

    # -- serialization -----------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments needed to rebuild this instance."""
        return {
            STR_MULTIPLIER: self.multiplier_value,
            STR_LOW: self.low,
            STR_HIGH: self.high,
            STR_REDUCTION: self.reduction,
            STR_QUADRATIC_TAILS: self.quadratic_tails,
            STR_ANNEALABLE: self.annealable,
            STR_NAME: self.name,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BinaryPreferenceRegularizer":
        """Rebuild from config, tolerating configs written by the old version."""
        return cls(**dict(config))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(multiplier={self.multiplier_value:g}, "
            f"low={self.low:g}, high={self.high:g}, "
            f"reduction={self.reduction!r}, "
            f"quadratic_tails={self.quadratic_tails})"
        )


# ---------------------------------------------------------------------

def create_binary_preference_regularizer(
    multiplier: float = DEFAULT_MULTIPLIER,
    low: float = DEFAULT_LOW,
    high: float = DEFAULT_HIGH,
    **kwargs: Any,
) -> BinaryPreferenceRegularizer:
    """Thin forwarder kept for backwards compatibility.

    Prefer constructing `BinaryPreferenceRegularizer` directly, or use the
    `for_gates` / `for_bipolar_weights` presets. All validation lives in the
    constructor; this function adds nothing but the call.
    """
    return BinaryPreferenceRegularizer(
        multiplier=multiplier, low=low, high=high, **kwargs
    )


# ---------------------------------------------------------------------

class BinaryPressureScheduler(keras.callbacks.Callback):
    """Linearly ramp a regularizer's multiplier from 0 to `target`.

    Annealing is not optional in practice: applied at full strength from step
    zero, the barrier freezes each weight into whichever well its initializer
    happened to place it in, before the task loss has had any say.

    Parameters
    ----------
    regularizer : BinaryPreferenceRegularizer
        Must have been constructed with ``annealable=True``.
    target : float
        Final multiplier value.
    warmup_epochs : int
        Epochs of pure task-loss training before the ramp begins.
    ramp_epochs : int
        Epochs over which the multiplier climbs from 0 to `target`.
    """

    def __init__(
        self,
        regularizer: BinaryPreferenceRegularizer,
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
        logger.debug(f"epoch {epoch}: binary pressure multiplier -> {value:.4g}")


# ---------------------------------------------------------------------
