"""Constrain weights to a value range with a smooth projection.

Provides :class:`SoftValueRangeConstraint`, a drop-in alternative to
:class:`~dl_techniques.constraints.value_range_constraint.ValueRangeConstraint`
that replaces the hard clip with a monotone softplus composition.

``ValueRangeConstraint`` projects onto the box with
``w' = max(min_value, min(w, max_value))``. That projection is exact, and it is
piecewise constant outside the box. A weight the optimizer pushed past a bound
is snapped back to the bound, and every later step sees the same value, so how
far outside it wanted to be is destroyed. In a WGAN critic, whose weights are
clipped to a small box after every update, this is the documented failure mode:
the clipped weights pile up on the two bound values and the critic degenerates
toward a much simpler function than the box allows.

This constraint applies the softplus composition from
``dl_techniques.layers.activations.soft_value_range`` instead::

    sp(u) = softplus(beta * u) / beta
    w'    = lo + sp(w - lo)          # lower bound, always applied
    w'    = hi - sp(hi - w')         # upper bound, only when max_value is given

The map is monotone, never crosses a bound, and is the identity in the interior
up to a bias of at most ``log(2) / beta``. Where the hard clip is flat, this one
still moves: two weights at different distances outside the box land at
different, if very close, values, so their ordering survives the projection.
The formula lives in one place, the activations module; this class is a thin
role adapter over it and re-derives nothing.

This is still a post-hoc projection, exactly like the hard clip. Keras applies
it through ``variable.assign(variable.constraint(variable))`` after the
optimizer has applied the gradients, outside any gradient tape
(``keras/src/optimizers/base_optimizer.py:447-452``). Nothing here is
differentiated; the class shapes the value the next forward pass sees, not any
gradient of the current step. For a differentiable use of the same map inside a
forward pass, use ``SoftValueRange`` or ``soft_value_range`` from the
activations module.

References:
    - Arjovsky et al., 2017. Wasserstein GAN
      (https://arxiv.org/abs/1701.07875) -- the weight-clipping critic.
    - Gulrajani et al., 2017. Improved Training of Wasserstein GANs
      (https://arxiv.org/abs/1704.00028) -- documents the pathologies of the
      hard weight clip that motivate a smooth projection.
    - Bertsekas, 1999. Nonlinear Programming (for Projected Gradient Methods,
      and for the smooth exact-penalty reformulation of a box constraint).
"""

import keras
from typing import Dict, Union, Optional, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.layers.activations.soft_value_range import soft_value_range

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.constraints.soft_value_range_constraint")
class SoftValueRangeConstraint(keras.constraints.Constraint):
    """Project weights into a value range with a smooth, monotone map.

    Replaces the hard clip ``max(lo, min(w, hi))`` with
    ``w' = hi - sp(hi - (lo + sp(w - lo)))``, where
    ``sp(u) = softplus(beta * u) / beta``. The projection is monotone and stays
    inside the box, and unlike the hard clip it is not flat outside it: weights
    landing at different distances beyond a bound are still mapped to different
    values, so their ordering is preserved rather than collapsed onto the bound.

    **Transfer function, against the hard clip:**

    .. code-block:: text

            w'
             |
          hi +- - - - - - -,---=========   soft: approaches hi, never reaches
             |           ,'                hard: flat AT hi
             |          /
             |         /                   identity for lo <= w <= hi,
             |        /                    up to a bias of log(2)/beta
             |      ,'
          lo +====,'- - - - - - - - - -    soft: still rising below lo
             |    |            |           hard: flat AT lo
             +----+------------+---------> w
                 lo           hi

    **Pipeline:**

    .. code-block:: text

        weights
           |
           v
        ┌────────────────────────────────────────┐
        │ soft_value_range(...)                  │  the single shared
        │   w' = lo + sp(w - lo)                 │  definition, in
        │   w' = hi - sp(hi - w')  (two-sided)   │  layers/activations/
        └──────────────────┬─────────────────────┘
                           v
        ┌────────────────────────────────────────┐
        │ maximum(w', lo)                        │  ('enforce_hard_bounds')
        │ minimum(w', hi)   (two-sided only)     │  makes the bounds exact
        └──────────────────┬─────────────────────┘
                           v
                     constrained weights

    Typical uses: WGAN critics, where hard clipping piles weights onto the two
    bound values and degenerates the critic; any bounded parameter such as a
    slope, a temperature or a gate scale where saturating exactly at a bound
    stalls it there; and a positivity floor with no flat region at the floor.

    The math is not implemented here. :meth:`__call__` delegates to
    ``dl_techniques.layers.activations.soft_value_range``, the single definition
    shared by the plain function, the ``SoftValueRange`` layer and this
    constraint.

    :param min_value: Minimum allowed value for weights. Always applied.
    :type min_value: float
    :param max_value: Maximum allowed value for weights. ``None`` applies only
        the smooth minimum, leaving no ceiling.
    :type max_value: float or None
    :param sharpness: Knee steepness of the softplus. Larger values sit closer
        to a hard clip and shrink the interior bias, which is bounded by
        ``log(2) / beta``. Must be strictly positive. The default of 50.0 gives
        a two-sided interior bias of ``log(2) * (hi - lo) / 50``, for example
        ``2.8e-04`` on ``[-0.01, 0.01]``.
    :type sharpness: float
    :param relative_sharpness: When ``True``,
        ``beta = sharpness / (hi - lo)``, so ``sharpness`` is expressed in
        interval widths and transfers unchanged between a ``[-0.01, 0.01]`` box
        and a ``[-1, 1]`` box. Ignored, not an error, when ``max_value`` is
        ``None``, where ``beta = sharpness`` in the weights' own units.
    :type relative_sharpness: bool
    :param enforce_hard_bounds: When ``True``, an exact ``keras.ops.maximum`` /
        ``keras.ops.minimum`` guard runs after the smooth map, making
        ``lo <= w' <= hi`` exact rather than merely bounded. See the note below.
    :type enforce_hard_bounds: bool
    :param kwargs: Must be empty. ``keras.constraints.Constraint`` defines no
        ``__init__``, so any keyword forwarded here reaches ``object.__init__``
        and raises ``TypeError``.

    :ivar min_value: The coerced lower bound.
    :vartype min_value: float
    :ivar max_value: The coerced upper bound, or ``None``.
    :vartype max_value: float or None
    :ivar sharpness: The coerced knee steepness.
    :vartype sharpness: float
    :ivar relative_sharpness: Whether sharpness is in interval widths.
    :vartype relative_sharpness: bool
    :ivar enforce_hard_bounds: Whether the exact guard is applied.
    :vartype enforce_hard_bounds: bool

    :raises ValueError: If ``sharpness`` is not strictly positive, or if
        ``min_value`` is greater than ``max_value`` when ``max_value`` is given.
    :raises TypeError: If any keyword argument is supplied.

    Note:
        This class carries no ``clip_gradients`` parameter.
        ``ValueRangeConstraint`` has one, and it is a no-op that its own test
        suite proves inert. Constraints are applied outside any gradient tape,
        so no such flag could do anything.

    Note:
        On ``enforce_hard_bounds``: ``w' <= hi`` is already structural, since
        ``sp`` is non-negative and ``hi - sp(...)`` cannot exceed ``hi``. The
        lower bound is the one that can be missed. The two branches are
        composed, so the upper branch reads the already-lifted value and pulls
        it back down by up to
        ``log(1 + exp(-beta * (hi - lo))) / beta``. That undershoot is a
        property of the real-valued map, not of floating point, and it is only
        visible at low sharpness. Measured on ``[-1, 1]`` over 20001 points
        spanning +-50, maximum undershoot below ``lo``:

        ===================  ==================
        sharpness (relative) measured undershoot
        ===================  ==================
        1.0                  6.265e-01
        2.0                  1.269e-01
        5.0                  2.686e-03
        10.0                 9.060e-06
        20.0                 0.0 (exact)
        50.0 (the default)   0.0 (exact)
        ===================  ==================

        At the default sharpness the guard changes nothing; at
        ``sharpness=1.0`` on ``[-1, 1]`` it changes output bits. Set it to
        ``False`` only to get exactly the smooth map, undershoot included.

        The guard costs nothing in this role, because no gradient is ever taken
        through this call and there is no gradient for an exact clamp to zero
        out. It would matter for autodiff if this object were called inside a
        forward pass, since an exact clamp is flat outside the box, which is
        what this map exists to avoid. For that case use ``SoftValueRange`` (a
        layer) or ``soft_value_range`` (a plain function) from
        ``dl_techniques.layers.activations.soft_value_range``.

    Example:
        >>> # A WGAN critic's weight box, smoothly projected
        >>> constraint = SoftValueRangeConstraint(min_value=-0.01, max_value=0.01)
        >>> layer = keras.layers.Dense(units=64, kernel_constraint=constraint)

        >>> # A positivity floor with no flat region at the floor. One-sided mode, so
        >>> # `sharpness` is in the weights' own units and `relative_sharpness` is moot.
        >>> constraint = SoftValueRangeConstraint(
        ...     min_value=1e-3, sharpness=1000.0, relative_sharpness=False
        ... )
        >>> layer = keras.layers.Dense(units=32, kernel_constraint=constraint)
    """

    def __init__(
            self,
            min_value: float,
            max_value: Optional[float] = None,
            sharpness: float = 50.0,
            relative_sharpness: bool = True,
            enforce_hard_bounds: bool = True,
            **kwargs: Any
    ) -> None:
        """Validate the range and knee parameters and store them.

        :param min_value: Minimum allowed value for weights.
        :type min_value: float
        :param max_value: Maximum allowed value for weights, or ``None`` for
            one-sided mode.
        :type max_value: float or None
        :param sharpness: Knee steepness; must be strictly positive.
        :type sharpness: float
        :param relative_sharpness: Whether ``sharpness`` is expressed in
            interval widths. Ignored when ``max_value`` is ``None``.
        :type relative_sharpness: bool
        :param enforce_hard_bounds: Whether to apply the exact bound guard after
            the smooth map.
        :type enforce_hard_bounds: bool
        :param kwargs: Must be empty; see the class docstring.
        :raises ValueError: If ``sharpness`` is not strictly positive, or if
            ``min_value`` is greater than ``max_value`` when ``max_value`` is
            given.
        :raises TypeError: If any keyword argument is supplied.
        """
        super().__init__(**kwargs)

        # DECISION plan-2026-09-01T175024-5a32e889/D-003: validate at
        # construction, and keep these checks rather than importing the
        # activations module's private `_validated_bounds`. TestValidation
        # asserts this class and soft_value_range reject the same parameter
        # sets, so a one-sided edit reddens. See D-003.
        if sharpness <= 0.0:
            raise ValueError(
                f"sharpness must be strictly positive, got {sharpness}. It is the "
                f"knee steepness of the smooth ramp; a non-positive value inverts or "
                f"annihilates the projection."
            )

        if max_value is not None and min_value > max_value:
            raise ValueError(
                f"min_value ({min_value}) cannot be greater than max_value "
                f"({max_value})"
            )

        self.min_value = float(min_value)
        self.max_value = float(max_value) if max_value is not None else None
        self.sharpness = float(sharpness)
        self.relative_sharpness = bool(relative_sharpness)
        self.enforce_hard_bounds = bool(enforce_hard_bounds)

        logger.debug(
            f"Initialized SoftValueRangeConstraint with min_value={self.min_value}, "
            f"max_value={self.max_value}, sharpness={self.sharpness}, "
            f"relative_sharpness={self.relative_sharpness}, "
            f"enforce_hard_bounds={self.enforce_hard_bounds}"
        )

    def __call__(self, weights: keras.KerasTensor) -> keras.KerasTensor:
        """Project weights into the value range with the smooth map.

        :param weights: Weight tensor to constrain.
        :type weights: keras.KerasTensor
        :return: The constrained weights, same shape and dtype as the input.
        :rtype: keras.KerasTensor
        """
        # One definition, three roles: the composition is not restated here.
        constrained = soft_value_range(
            weights,
            min_value=self.min_value,
            max_value=self.max_value,
            sharpness=self.sharpness,
            relative_sharpness=self.relative_sharpness,
        )

        # DECISION plan-2026-09-01T175024-5a32e889/D-002: the exact guard lives
        # here and nowhere else. Do NOT add a clamp to soft_value_range, which
        # runs in a forward pass and would regain the flat region. Here it is
        # free (no tape) and buys an exact lower bound: measured undershoot
        # 6.265e-01 at relative sharpness 1.0 on [-1, 1]. See D-002.
        if self.enforce_hard_bounds:
            constrained = keras.ops.maximum(constrained, self.min_value)
            if self.max_value is not None:
                constrained = keras.ops.minimum(constrained, self.max_value)

        return constrained

    def get_config(self) -> Dict[str, Union[float, None, bool]]:
        """Return the constructor arguments for serialization.

        :return: A dict holding the two bounds, ``sharpness``,
            ``relative_sharpness`` and ``enforce_hard_bounds``.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            'min_value': self.min_value,
            'max_value': self.max_value,
            'sharpness': self.sharpness,
            'relative_sharpness': self.relative_sharpness,
            'enforce_hard_bounds': self.enforce_hard_bounds,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SoftValueRangeConstraint':
        """Rebuild a constraint from a config dict.

        :param config: Configuration dictionary from :meth:`get_config`.
        :type config: dict
        :return: A new constraint.
        :rtype: SoftValueRangeConstraint
        """
        return cls(**config)

    def __repr__(self) -> str:
        """Return the constructor-like representation.

        ``max_value`` is omitted in one-sided mode, matching
        ``ValueRangeConstraint``.

        :return: A string naming the bounds and the knee parameters.
        :rtype: str
        """
        if self.max_value is not None:
            return (f"SoftValueRangeConstraint(min_value={self.min_value}, "
                    f"max_value={self.max_value}, sharpness={self.sharpness}, "
                    f"relative_sharpness={self.relative_sharpness}, "
                    f"enforce_hard_bounds={self.enforce_hard_bounds})")
        else:
            return (f"SoftValueRangeConstraint(min_value={self.min_value}, "
                    f"sharpness={self.sharpness}, "
                    f"relative_sharpness={self.relative_sharpness}, "
                    f"enforce_hard_bounds={self.enforce_hard_bounds})")

# ---------------------------------------------------------------------
