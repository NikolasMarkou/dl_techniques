"""Double-well regularizer that pulls weights toward two target values.

Provides :class:`BinaryPreferenceRegularizer`, a penalty with zero-gradient
minima at ``low`` and ``high`` and a barrier between them,
:class:`BinaryPressureScheduler`, a callback that anneals its strength, and
:func:`create_binary_preference_regularizer`, a thin constructor forwarder.

A weight sitting exactly at a target feels no force; a weight between the
targets is pushed toward whichever one it is closer to. It is a smooth,
differentiable stand-in for a hard binarizing constraint.

Foundational mathematics
------------------------
For half-gap ``h = (high - low) / 2`` the per-weight penalty is::

    L(w) = m * (w - low)^2 * (w - high)^2 / h^4

For the canonical ``low=0, high=1`` case this reduces to
``L(w) = 16 m w^2 (1-w)^2``, the standard Cahn-Hilliard / Ginzburg-Landau
double-well potential. Dividing by ``h^4`` fixes the barrier height at the
midpoint to exactly ``m`` whatever the targets, so `multiplier` means the same
thing for {0, 1} as for {-1, +1}.

Key quantities, per weight::

    L'(w)                      = 2m (w-low)(w-high)(2w-low-high) / h^4
    L(midpoint)                = m                      (barrier height)
    L''(low) = L''(high)       = 8m / h^2               (curvature at minima)
    max |L'|                   ~ 3.08 m / h             (steepest slope)
    local L2-equivalent lambda = 4m / h^2

That last line is the number to reason about when picking `multiplier`. Near a
minimum this behaves exactly like L2 pull toward that target with
``lambda = 4m/h^2``. For {0, 1} targets that is ``16 m``, so
``multiplier=1.0`` is a very strong penalty, not a mild one.

Read this before using it
-------------------------
1.  Do not attach the {0, 1} configuration to a `Dense` or `Conv2D` kernel.
    Glorot/He initializers are zero-centered, so every weight starts in the
    ``w=0`` well and cannot climb the barrier to reach ``w=1``. The result is
    not binarization; it is aggressive L2 that collapses the layer to zero and
    forbids negative weights. For kernels use ``low=-1.0, high=1.0`` (see
    `for_bipolar_weights`). For gates and masks initialized inside [0, 1] the
    {0, 1} configuration is the right one (see `for_gates`).

2.  Anneal the multiplier. The total loss is non-convex: the barrier freezes
    each weight's well assignment early, from initialization noise alone. Ramp
    `multiplier` from 0 over training. It is stored as a non-trainable
    `keras.Variable` by default so a callback can update it via
    `set_multiplier`.

3.  Reduction matters. `sum` (the default, matching Keras built-ins) keeps the
    per-weight gradient independent of layer size. `mean` normalizes the loss
    across layers but divides the per-weight gradient by the parameter count,
    so a `multiplier` tuned on a small layer does nothing on a large one.
    Choose knowingly.

4.  The tails are quartic and unbounded: gradients grow cubically outside
    [low, high]. Set ``quadratic_tails=True`` for the C2-continuous quadratic
    extension beyond the targets, which caps gradient growth at linear and is
    much safer without gradient clipping.

References
----------
The potential is the classic double-well of Ginzburg-Landau / Cahn-Hilliard
theory, not a novel construction. Its use for weight binarization has direct
prior art:

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

    The per-weight penalty is::

        L(w) = multiplier * (w - low)^2 * (w - high)^2 / h^4,  h = (high-low)/2

    The barrier height at the midpoint equals `multiplier` for any targets.
    Curvature at each minimum is ``8 * multiplier / h^2``, so locally this is
    L2-toward-the-target with ``lambda = 4 * multiplier / h^2``
    (``16 * multiplier`` for the default {0, 1} targets).

    **Penalty shape:**

    .. code-block:: text

        L(w)
          |
        m +- - - - - - -*- - - - - - -     barrier = multiplier
          |    quartic /|\\ quartic
          |   tail    / | \\    tail
          |          /  |  \\
          |  ___    /   |   \\    ___
          | .   '--'    |    '--'   '.
        0 +------*------+------*-------> w
                low   midpoint high

        quadratic_tails=True replaces both outer arms with
        4*multiplier*d^2/h^2, matching value, slope and curvature
        at the targets, so only the far tails change.

    **Presets:**

    .. code-block:: text

        constructor            targets    reduction  tails      use on
        --------------------   --------   ---------  ---------  ------------
        for_gates()            {0, 1}     mean       quadratic  gates, masks
        for_bipolar_weights()  {-1, +1}   sum        quadratic  layer kernels
        BinaryPreference...()  {0, 1}     sum        quartic    (raw default)

    :param multiplier: Barrier height, and the overall strength of the penalty.
        Must be non-negative. Read the L2-equivalence above before picking a
        value; 1.0 is strong.
    :type multiplier: float
    :param low: Lower target value, a zero of the penalty.
    :type low: float
    :param high: Upper target value, a zero of the penalty. Must exceed `low`.
    :type high: float
    :param reduction: How per-weight costs are combined, ``"sum"`` or
        ``"mean"``. ``"sum"`` matches the Keras built-in regularizers and keeps
        the per-weight gradient independent of layer size; ``"mean"`` divides
        that gradient by the parameter count.
    :type reduction: str
    :param quadratic_tails: If ``True``, replace the quartic growth outside
        [low, high] with the C2-continuous quadratic
        ``4 * multiplier * d^2 / h^2``, where ``d`` is the signed distance past
        the nearer target. Value, slope and curvature all match at the targets,
        so the well shape is untouched and only the far tails change, from
        cubic to linear gradients. Recommended when weights can leave the
        target interval.
    :type quadratic_tails: bool
    :param annealable: If ``True``, `multiplier` lives in a non-trainable
        ``keras.Variable`` and can be updated during training via
        :meth:`set_multiplier`. If ``False`` it is a Python float folded into
        the graph as a constant.
    :type annealable: bool
    :param name: Name for the multiplier variable. Ignored when
        ``annealable=False``.
    :type name: str or None
    :param kwargs: Only the deprecated ``scale`` key is accepted, and it is
        translated to ``low=0.0, high=1.0/scale``. Any other keyword raises.

    :ivar low: The lower target.
    :vartype low: float
    :ivar high: The upper target.
    :vartype high: float
    :ivar reduction: The selected reduction.
    :vartype reduction: str
    :ivar quadratic_tails: Whether the softened tails are in use.
    :vartype quadratic_tails: bool
    :ivar annealable: Whether the multiplier is a variable.
    :vartype annealable: bool
    :ivar multiplier: The penalty strength, a ``keras.Variable`` when
        ``annealable`` is ``True`` and a ``float`` otherwise.
    :vartype multiplier: keras.Variable or float

    :raises ValueError: If `multiplier` is negative, `high` is not greater than
        `low`, `reduction` is not ``"sum"`` or ``"mean"``, or a deprecated
        ``scale`` is not positive.
    :raises TypeError: If an unrecognized keyword argument is supplied.

    Warning:
        The default {0, 1} targets are for gates and masks initialized inside
        [0, 1], not for zero-centered layer kernels. On a Glorot-initialized
        kernel this cannot binarize, since no weight can cross the barrier to
        reach 1, and instead acts as strong L2 with a hard floor at zero. Use
        :meth:`for_bipolar_weights` for kernels.

    Example:
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
        """Validate the targets and set up the multiplier.

        :param multiplier: Non-negative barrier height and penalty strength.
        :type multiplier: float
        :param low: Lower target value.
        :type low: float
        :param high: Upper target value; must exceed `low`.
        :type high: float
        :param reduction: ``"sum"`` or ``"mean"``.
        :type reduction: str
        :param quadratic_tails: Whether to soften the tails beyond the targets.
        :type quadratic_tails: bool
        :param annealable: Whether the multiplier is an updatable variable.
        :type annealable: bool
        :param name: Name for the multiplier variable.
        :type name: str or None
        :param kwargs: Only the deprecated ``scale`` key is accepted.
        :raises ValueError: See the class docstring.
        :raises TypeError: If an unrecognized keyword argument is supplied.
        """
        # `keras.regularizers.Regularizer` defines no __init__, so forwarding
        # **kwargs to super() would hit object.__init__ and raise. Accept the
        # legacy `scale` argument, reject everything else loudly.
        legacy_scale = kwargs.pop(STR_LEGACY_SCALE, None)
        if kwargs:
            raise TypeError(
                f"unexpected keyword arguments: {sorted(kwargs)}"
            )
        if legacy_scale is not None:
            # Old semantics: zeros at 0 and 1/scale. Translate rather than
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

        # h is the half-gap; h^4 normalizes the barrier height to `multiplier`,
        # h^2 normalizes the quadratic tails.
        half_gap = (self.high - self.low) / 2.0
        self._h2 = half_gap ** 2
        self._h4 = half_gap ** 4

        if self.annealable:
            # Non-trainable so the optimizer ignores it; a callback assigns to
            # it mid-training to ramp the binarization pressure.
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
        """Build the {0, 1} preset for gates or masks initialized inside [0, 1].

        Uses ``reduction="mean"`` so the loss contribution does not scale with
        the number of gates, and defaults `multiplier` to 0.0 on the assumption
        that it will be annealed up from zero.

        :param multiplier: Initial penalty strength.
        :type multiplier: float
        :param kwargs: Forwarded to the constructor; ``reduction`` and
            ``quadratic_tails`` are only defaults here and can be overridden.
        :return: The configured regularizer.
        :rtype: BinaryPreferenceRegularizer
        """
        kwargs.setdefault(STR_REDUCTION, "mean")
        kwargs.setdefault(STR_QUADRATIC_TAILS, True)
        return cls(multiplier=multiplier, low=0.0, high=1.0, **kwargs)

    @classmethod
    def for_bipolar_weights(
        cls, multiplier: float = 0.0, **kwargs: Any
    ) -> "BinaryPreferenceRegularizer":
        """Build the {-1, +1} preset for kernel binarization.

        This is the configuration for `Dense` and `Conv2D` kernels. It is
        symmetric about zero, so a standard zero-centered initializer places
        weights at the top of the barrier and task gradients decide which well
        each one falls into. Half-gap is 1, so the local L2-equivalent lambda
        is ``4 * multiplier``.

        :param multiplier: Initial penalty strength.
        :type multiplier: float
        :param kwargs: Forwarded to the constructor.
        :return: The configured regularizer.
        :rtype: BinaryPreferenceRegularizer
        """
        kwargs.setdefault(STR_QUADRATIC_TAILS, True)
        return cls(multiplier=multiplier, low=-1.0, high=1.0, **kwargs)

    # -- runtime control ---------------------------------------------------

    def set_multiplier(self, value: float) -> None:
        """Update the penalty strength during training.

        Available only when ``annealable=True``. Call it from a callback to
        ramp binarization pressure in over training::

            reg.set_multiplier(target * min(1.0, epoch / warmup_epochs))

        :param value: New non-negative penalty strength.
        :type value: float
        :return: Nothing.
        :rtype: None
        :raises RuntimeError: If the instance was built with
            ``annealable=False``.
        :raises ValueError: If ``value`` is negative.
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
        """Current penalty strength as a Python float.

        :return: The multiplier, read out of the variable when annealable.
        :rtype: float
        """
        if self.annealable:
            return float(keras.ops.convert_to_numpy(self.multiplier))
        return float(self.multiplier)

    @property
    def equivalent_l2_lambda(self) -> float:
        """L2 coefficient this is locally equivalent to near either minimum.

        :return: ``4 * multiplier / h^2``.
        :rtype: float
        """
        return 4.0 * self.multiplier_value / self._h2

    # -- penalty -----------------------------------------------------------

    def __call__(self, weights: keras.KerasTensor) -> keras.KerasTensor:
        """Compute the double-well penalty for `weights`.

        Uses the factored form ``((w-low)(w-high))^2 / h^4`` rather than the
        algebraically identical ``(1 - (w-c)^2/h^2)^2``. Both are the same
        polynomial, but the factored one is cheaper and has no subtractive
        cancellation: the expanded form subtracts near-equal floats right at
        the minima, which is where the weights end up.

        :param weights: Weight tensor to regularize.
        :type weights: tensor
        :return: The scalar penalty.
        :rtype: tensor
        """
        dtype = weights.dtype
        low = keras.ops.cast(self.low, dtype)
        high = keras.ops.cast(self.high, dtype)

        # Clip before forming the quartic so the core term cannot overflow on
        # out-of-range weights (relevant in fp16). Outside [low, high] the
        # clipped core is identically zero and contributes no gradient; the
        # tail branch below supplies the penalty there instead.
        w_core = keras.ops.clip(weights, low, high) if self.quadratic_tails else weights

        d_low_core = keras.ops.subtract(w_core, low)
        d_high_core = keras.ops.subtract(w_core, high)
        unit_cost = keras.ops.divide(
            keras.ops.square(keras.ops.multiply(d_low_core, d_high_core)),
            keras.ops.cast(self._h4, dtype),
        )

        if self.quadratic_tails:
            # C2-continuous extension: matches the quartic's value (0), slope
            # (0) and curvature (8m/h^2) at each target, so the well is
            # unchanged and only the far tails are softened.
            inv_h2 = keras.ops.cast(4.0 / self._h2, dtype)
            below = keras.ops.multiply(inv_h2, keras.ops.square(keras.ops.subtract(weights, low)))
            above = keras.ops.multiply(inv_h2, keras.ops.square(keras.ops.subtract(weights, high)))
            unit_cost = keras.ops.where(
                keras.ops.less(weights, low),
                below,
                keras.ops.where(keras.ops.greater(weights, high), above, unit_cost),
            )

        reduced = (
            keras.ops.sum(unit_cost) if self.reduction == "sum" else keras.ops.mean(unit_cost)
        )
        return keras.ops.multiply(keras.ops.cast(self.multiplier, dtype), reduced)

    # -- serialization -----------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments for serialization.

        :return: A dict holding the multiplier, targets, reduction, tail mode,
            annealability and name.
        :rtype: dict
        """
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
        """Rebuild a regularizer from a config dict.

        A config carrying the deprecated ``scale`` key is accepted and
        translated by the constructor.

        :param config: Configuration dictionary from :meth:`get_config`.
        :type config: dict
        :return: A new regularizer.
        :rtype: BinaryPreferenceRegularizer
        """
        return cls(**dict(config))

    def __repr__(self) -> str:
        """Return the constructor-like representation.

        :return: A string naming the multiplier, targets, reduction and tail
            mode.
        :rtype: str
        """
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
    """Build a :class:`BinaryPreferenceRegularizer`.

    Prefer constructing the class directly, or use the
    :meth:`BinaryPreferenceRegularizer.for_gates` /
    :meth:`BinaryPreferenceRegularizer.for_bipolar_weights` presets. All
    validation lives in the constructor; this adds nothing but the call.

    :param multiplier: Barrier height and penalty strength.
    :type multiplier: float
    :param low: Lower target value.
    :type low: float
    :param high: Upper target value.
    :type high: float
    :param kwargs: Forwarded to the constructor.
    :return: The configured regularizer.
    :rtype: BinaryPreferenceRegularizer
    """
    return BinaryPreferenceRegularizer(
        multiplier=multiplier, low=low, high=high, **kwargs
    )


# ---------------------------------------------------------------------

class BinaryPressureScheduler(keras.callbacks.Callback):
    """Linearly ramp a regularizer's multiplier from 0 to `target`.

    Annealing is not optional in practice: at full strength from step zero the
    barrier freezes each weight into whichever well its initializer happened to
    place it in, before the task loss has had any say.

    **Ramp:**

    .. code-block:: text

        multiplier
             |                 ______________
             |                /
        target - - - - - - - +
             |              /
             |             /
           0 +------------+--------------------> epoch
             0      warmup_epochs
                          |<-ramp_epochs->|

        value = target * clamp((epoch - warmup_epochs) / ramp_epochs, 0, 1)

    :param regularizer: The regularizer to drive. Must have been constructed
        with ``annealable=True``.
    :type regularizer: BinaryPreferenceRegularizer
    :param target: Final multiplier value.
    :type target: float
    :param warmup_epochs: Epochs of pure task-loss training before the ramp
        begins.
    :type warmup_epochs: int
    :param ramp_epochs: Epochs over which the multiplier climbs from 0 to
        `target`. Must be positive.
    :type ramp_epochs: int

    :raises ValueError: If ``ramp_epochs`` is not positive.
    """

    def __init__(
        self,
        regularizer: BinaryPreferenceRegularizer,
        target: float,
        warmup_epochs: int = 0,
        ramp_epochs: int = 10,
    ) -> None:
        """Store the regularizer and the ramp schedule.

        :param regularizer: The annealable regularizer to drive.
        :type regularizer: BinaryPreferenceRegularizer
        :param target: Final multiplier value.
        :type target: float
        :param warmup_epochs: Epochs before the ramp begins.
        :type warmup_epochs: int
        :param ramp_epochs: Positive number of epochs the ramp spans.
        :type ramp_epochs: int
        :raises ValueError: If ``ramp_epochs`` is not positive.
        """
        super().__init__()
        if ramp_epochs <= 0:
            raise ValueError(f"ramp_epochs must be positive, got {ramp_epochs}")
        self.regularizer = regularizer
        self.target = float(target)
        self.warmup_epochs = int(warmup_epochs)
        self.ramp_epochs = int(ramp_epochs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Set the multiplier for the epoch about to start.

        :param epoch: Index of the epoch about to begin.
        :type epoch: int
        :param logs: Keras metrics dict; unused.
        :type logs: dict or None
        :return: Nothing.
        :rtype: None
        """
        progress = (epoch - self.warmup_epochs) / self.ramp_epochs
        value = self.target * min(1.0, max(0.0, progress))
        self.regularizer.set_multiplier(value)
        logger.debug(f"epoch {epoch}: binary pressure multiplier -> {value:.4g}")


# ---------------------------------------------------------------------
