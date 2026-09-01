"""
Constrain weights to a value range with a smooth projection instead of a hard clip.

`ValueRangeConstraint` projects a weight tensor onto the box `[min_value, max_value]`
with `w' = max(min_value, min(w, max_value))`. That projection is exact, and it has
one structural cost: it is piecewise constant outside the box. A weight the optimizer
pushed past a bound is snapped back to the bound and every subsequent step sees the
same value, so the information about how far outside it wanted to be is destroyed.
In a WGAN critic, whose weights are clipped to a small box after every update, this
is the documented failure mode: the clipped weights pile up on the two bound values
and the critic degenerates towards a much simpler function than the box allows.

This constraint applies the softplus composition from
`dl_techniques.layers.activations.soft_value_range` instead:

    sp(u) = softplus(beta * u) / beta
    w'    = lo + sp(w - lo)          # lower bound, always applied
    w'    = hi - sp(hi - w')         # upper bound, only when max_value is given

The map is monotone, never crosses a bound, and is the identity in the interior up to
a bias of at most `log(2) / beta`. Where the hard clip is flat, this one still moves:
two weights at different distances outside the box land at different -- if very close
-- values, so their ordering survives the projection. The formula itself lives in one
place, the activations module; this class is a thin role adapter over it and re-derives
nothing.

Architecturally this is still a post-hoc projection, exactly like the hard clip. Keras
applies it via `variable.assign(variable.constraint(variable))` after the optimizer has
already applied the gradients, outside any gradient tape
(`keras/src/optimizers/base_optimizer.py:447-452`). Nothing here is differentiated;
the class shapes the value that the NEXT forward pass sees, not any gradient of the
current step. For a differentiable use of the same map inside a forward pass, use
`SoftValueRange` or `soft_value_range` from the activations module.

References:
    - Arjovsky et al., 2017. Wasserstein GAN
      (https://arxiv.org/abs/1701.07875) -- the weight-clipping critic.
    - Gulrajani et al., 2017. Improved Training of Wasserstein GANs
      (https://arxiv.org/abs/1704.00028) -- documents the pathologies of the hard
      weight clip that motivate a smooth projection.
    - Bertsekas, 1999. Nonlinear Programming (for Projected Gradient Methods, and for
      the smooth exact-penalty reformulation of a box constraint).

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
    """Constrains weights to a value range with a smooth, monotone projection.

    A drop-in alternative to `ValueRangeConstraint` that replaces the hard clip
    `max(lo, min(w, hi))` with the softplus composition
    `w' = hi - sp(hi - (lo + sp(w - lo)))`, `sp(u) = softplus(beta * u) / beta`.
    The projection is monotone and stays inside the box, but unlike the hard clip it
    is not flat outside it: weights that land at different distances beyond a bound
    are still mapped to different values, so their ordering is preserved rather than
    collapsed onto the bound. Typical uses:

    * WGAN critics, where hard clipping is known to pile weights onto the two bound
      values and degenerate the critic
    * Any bounded parameter (a slope, a temperature, a gate scale) where saturating
      exactly at a bound stalls it there
    * Keeping a positivity floor without a flat region at the floor

    The math is not implemented here. `__call__` delegates to
    `dl_techniques.layers.activations.soft_value_range`, which is the single
    definition shared by the plain function, the `SoftValueRange` layer and this
    constraint.

    Note:
        This class deliberately does NOT carry a `clip_gradients` parameter.
        `ValueRangeConstraint` has one, and it is a documented no-op that its own
        test suite proves inert (`tests/test_constraints/test_value_range_constraint.py`
        asserts both values give identical output). Constraints are applied outside
        any gradient tape, so no such flag could do anything.

    Args:
        min_value (float): Minimum allowed value for weights. Always applied.
        max_value (Optional[float]): Maximum allowed value for weights. If None, only
            the smooth minimum is applied and there is no ceiling. Defaults to None.
        sharpness (float): Knee steepness of the softplus. Larger values sit closer to
            a hard clip and shrink the interior bias, which is bounded by
            `log(2) / beta`. Must be strictly positive. Defaults to 50.0, which for a
            two-sided range gives an interior bias of `log(2) * (hi - lo) / 50`, e.g.
            `2.8e-04` on `[-0.01, 0.01]`.
        relative_sharpness (bool): When True (the default), `beta = sharpness / (hi - lo)`,
            so `sharpness` is expressed in interval widths and transfers unchanged
            between a `[-0.01, 0.01]` box and a `[-1, 1]` box. Ignored -- not an error
            -- when `max_value` is None, where `beta = sharpness` in the weights' own
            units. Defaults to True.
        enforce_hard_bounds (bool): When True (the default), an exact
            `keras.ops.maximum` / `keras.ops.minimum` guard is applied AFTER the smooth
            map, making `lo <= w' <= hi` exact rather than merely bounded. See below.
            Defaults to True.
        **kwargs: Additional keyword arguments passed to the parent class.

    About `enforce_hard_bounds`:
        `w' <= hi` is already structural -- `sp` is non-negative, so `hi - sp(...)`
        cannot exceed `hi`. The LOWER bound is the one that can be missed. The two
        branches are composed, so the upper branch reads the already-lifted value and
        pulls it back down by up to `log(1 + exp(-beta * (hi - lo))) / beta`. That
        undershoot is a property of the real-valued map, not of floating point, and it
        is only visible at low sharpness. Measured on `[-1, 1]` over 20001 points
        spanning +-50, maximum undershoot below `lo`:

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

        So at the default sharpness the guard changes nothing at all, and at
        `sharpness=1.0` on `[-1, 1]` it changes output bits. Set it to False only if
        you want the projection to be exactly the smooth map, undershoot included.

        The guard costs nothing in this role. Keras applies a constraint via
        `variable.assign(variable.constraint(variable))` after `_backend_apply_gradients`
        and outside any gradient tape (`keras/src/optimizers/base_optimizer.py:447-452`),
        so no gradient is ever taken through this call and there is no gradient for an
        exact clamp to zero out. The flag WOULD matter for autodiff if someone called
        this object inside a forward pass -- an exact clamp is flat outside the box,
        which is the whole thing this map exists to avoid -- but that is not this
        class's role. Use `SoftValueRange` (a layer) or `soft_value_range` (a plain
        function) from `dl_techniques.layers.activations.soft_value_range` for the
        differentiable forward-pass case.

    Raises:
        ValueError: If `sharpness` is not strictly positive, or if `min_value` is
            greater than `max_value` when `max_value` is provided.

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
        """Initialize the constraint with its range and knee parameters.

        Args:
            min_value (float): Minimum allowed value for weights.
            max_value (Optional[float]): Maximum allowed value for weights, or None
                for one-sided mode. Defaults to None.
            sharpness (float): Knee steepness. Must be strictly positive.
                Defaults to 50.0.
            relative_sharpness (bool): Whether `sharpness` is expressed in interval
                widths. Ignored when `max_value` is None. Defaults to True.
            enforce_hard_bounds (bool): Whether to apply an exact bound guard after
                the smooth map. Defaults to True.
            **kwargs: Additional keyword arguments passed to parent class.

        Raises:
            ValueError: If sharpness is not strictly positive, or if min_value is
                greater than max_value when max_value is provided.
        """
        super().__init__(**kwargs)

        # DECISION plan-2026-09-01T175024-5a32e889/D-003
        # These two checks also exist in `_validated_bounds` in the activations
        # module, and this class deliberately does NOT import that private sibling --
        # a constraint reaching into another package's underscore-prefixed helper is
        # a worse coupling than the copy. The copy is kept honest MECHANICALLY, not
        # by a comment asking someone to remember: `TestValidation` in this class's
        # test file parametrizes the invalid parameter sets and asserts that
        # `SoftValueRangeConstraint(...)` and `soft_value_range(...)` reject exactly
        # the same ones, so a one-sided edit reddens. Do NOT delete these checks in
        # favour of letting `__call__` raise later: the house exemplar
        # (`value_range_constraint.py`) raises at construction, and a constraint that
        # only fails on the first optimizer step fails deep inside `fit`.
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

        Args:
            weights (keras.KerasTensor): Input tensor of weights to be constrained.

        Returns:
            keras.KerasTensor: Tensor with constrained weights, of the same shape and
                dtype as the input.
        """
        # The composition is NOT restated here. One definition, three roles.
        constrained = soft_value_range(
            weights,
            min_value=self.min_value,
            max_value=self.max_value,
            sharpness=self.sharpness,
            relative_sharpness=self.relative_sharpness,
        )

        # DECISION plan-2026-09-01T175024-5a32e889/D-002
        # The exact guard lives HERE and nowhere else. `soft_value_range` never
        # clips, because a clamp in a forward pass reintroduces the flat region the
        # map exists to remove; but a constraint is assigned outside any tape
        # (base_optimizer.py:447-452), so the clamp is free in this role. What it buys
        # is an exact lower bound: the composed upper branch can pull the value below
        # `lo` by up to log(1 + exp(-beta*(hi-lo)))/beta -- measured 6.265e-01 at
        # relative sharpness 1.0 on [-1, 1], and exactly 0.0 from sharpness 20 upward.
        # See decisions.md D-002 for why the brief's "reintroduces zero gradients"
        # framing is wrong: there is no gradient here to reintroduce.
        if self.enforce_hard_bounds:
            constrained = keras.ops.maximum(constrained, self.min_value)
            if self.max_value is not None:
                constrained = keras.ops.minimum(constrained, self.max_value)

        return constrained

    def get_config(self) -> Dict[str, Union[float, None, bool]]:
        """Return the configuration of the constraint for serialization.

        Returns:
            Dict[str, Union[float, None, bool]]: Dictionary containing the five
                configuration parameters needed to recreate this constraint.
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
        """Creates a constraint from its configuration dictionary.

        Args:
            config (Dict[str, Any]): Dictionary containing configuration parameters.

        Returns:
            SoftValueRangeConstraint: A new instance initialized with the provided
                configuration.
        """
        return cls(**config)

    def __repr__(self) -> str:
        """Return string representation of the constraint.

        Returns:
            str: String representation showing the constraint parameters. `max_value`
                is omitted in one-sided mode, matching `ValueRangeConstraint`.
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
