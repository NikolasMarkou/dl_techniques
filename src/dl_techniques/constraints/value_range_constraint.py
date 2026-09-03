"""Constrain weights to a value range with a hard clip.

Provides :class:`ValueRangeConstraint`, which projects a weight tensor
element-wise onto the interval ``[min_value, max_value]`` after every optimizer
update.

The projection is::

    w' = max(min_value, min(w, max_value))

That is a projection onto the convex hyperrectangle
``[min_value, max_value]^d``, so applying it after each update makes the whole
step a projected gradient descent step.

Restricting the hypothesis space this way keeps weights from growing
indefinitely, which reduces the risk of exploding gradients and helps numerical
stability in deep and recurrent architectures. It also encodes prior knowledge
directly: ``min_value=0`` enforces non-negativity, which is what models of
physical quantities and Non-negative Matrix Factorization need.

The best-known application is the Wasserstein GAN critic, where clipping the
weights to a small box enforces the Lipschitz condition that the Wasserstein
distance approximation requires. For that use see
:class:`~dl_techniques.constraints.soft_value_range_constraint.SoftValueRangeConstraint`,
whose smooth projection avoids the weight pile-up a hard clip produces.

References:
    - Arjovsky et al., 2017. Wasserstein GAN
      (https://arxiv.org/abs/1701.07875)
    - Bertsekas, 1999. Nonlinear Programming (for Projected Gradient Methods).
"""

import keras
from keras import ops
from typing import Dict, Union, Optional, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.constraints.value_range_constraint")
class ValueRangeConstraint(keras.constraints.Constraint):
    """Clip weights element-wise into ``[min_value, max_value]``.

    Keras applies this after each optimizer update, so weights that left the
    interval are snapped back to the nearest bound. Use it to prevent
    vanishing or exploding gradients, to meet an architectural requirement, to
    keep a layer numerically stable, or to enforce non-negative weights for an
    NMF-like decomposition.

    **Transfer function:**

    .. code-block:: text

            w'
             |
          hi +- - - - - -*==========      clipped at max_value
             |          /                 (skipped when max_value is None)
             |         /
             |        /                   identity for lo <= w <= hi
             |       /
          lo +======*- - - - - - - -      clipped at min_value
             |      |          |
             +------+----------+--------> w
                   lo         hi

        The map is flat outside the interval, so how far a weight went past a
        bound is discarded. SoftValueRangeConstraint keeps that ordering.

    :param min_value: Minimum allowed value for weights. Always applied.
    :type min_value: float
    :param max_value: Maximum allowed value for weights. ``None`` applies only
        the minimum, leaving no ceiling.
    :type max_value: float or None
    :param clip_gradients: Accepted and inert. Clipping is inherent to the
        constraint operation, and constraints run outside any gradient tape, so
        neither value changes the result.
        ``tests/test_constraints/test_value_range_constraint.py`` asserts both
        give identical output. The parameter is kept so existing configs still
        deserialize.
    :type clip_gradients: bool
    :param kwargs: Must be empty. ``keras.constraints.Constraint`` defines no
        ``__init__``, so any keyword forwarded here reaches ``object.__init__``
        and raises ``TypeError``.

    :ivar min_value: The coerced lower bound.
    :vartype min_value: float
    :ivar max_value: The coerced upper bound, or ``None``.
    :vartype max_value: float or None
    :ivar clip_gradients: The flag as passed, carried into the config.
    :vartype clip_gradients: bool

    :raises ValueError: If ``min_value`` is greater than ``max_value`` when
        ``max_value`` is given.
    :raises TypeError: If any keyword argument is supplied.

    Example:
        >>> # Constrain weights between 0.01 and 1.0
        >>> constraint = ValueRangeConstraint(min_value=0.01, max_value=1.0)
        >>> layer = keras.layers.Dense(
        ...     units=64,
        ...     kernel_constraint=constraint
        ... )

        >>> # Non-negative weights, with no ceiling
        >>> constraint = ValueRangeConstraint(min_value=0.0)
        >>> layer = keras.layers.Dense(
        ...     units=32,
        ...     kernel_constraint=constraint
        ... )
    """

    def __init__(
            self,
            min_value: float,
            max_value: Optional[float] = None,
            clip_gradients: bool = True,
            **kwargs: Any
    ) -> None:
        """Validate and store the bounds.

        :param min_value: Minimum allowed value for weights.
        :type min_value: float
        :param max_value: Maximum allowed value for weights, or ``None`` for no
            ceiling.
        :type max_value: float or None
        :param clip_gradients: Accepted and inert; see the class docstring.
        :type clip_gradients: bool
        :param kwargs: Must be empty; see the class docstring.
        :raises ValueError: If ``min_value`` is greater than ``max_value`` when
            ``max_value`` is given.
        :raises TypeError: If any keyword argument is supplied.
        """
        super().__init__(**kwargs)

        if max_value is not None and min_value > max_value:
            raise ValueError(
                f"min_value ({min_value}) cannot be greater than max_value ({max_value})"
            )

        self.min_value = float(min_value)
        self.max_value = float(max_value) if max_value is not None else None
        self.clip_gradients = clip_gradients

        logger.debug(
            f"Initialized ValueRangeConstraint with min_value={self.min_value}, "
            f"max_value={self.max_value}, clip_gradients={self.clip_gradients}"
        )

    def __call__(self, weights: keras.KerasTensor) -> keras.KerasTensor:
        """Clip the weights into the allowed range.

        :param weights: Weight tensor to constrain.
        :type weights: keras.KerasTensor
        :return: The weights clipped to the valid range, same shape and dtype.
        :rtype: keras.KerasTensor
        """
        constrained = ops.maximum(weights, self.min_value)

        if self.max_value is not None:
            constrained = ops.minimum(constrained, self.max_value)

        return constrained

    def get_config(self) -> Dict[str, Union[float, None, bool]]:
        """Return the constructor arguments for serialization.

        :return: A dict holding ``min_value``, ``max_value`` and
            ``clip_gradients``.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            'min_value': self.min_value,
            'max_value': self.max_value,
            'clip_gradients': self.clip_gradients,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ValueRangeConstraint':
        """Rebuild a constraint from a config dict.

        :param config: Configuration dictionary from :meth:`get_config`.
        :type config: dict
        :return: A new constraint.
        :rtype: ValueRangeConstraint
        """
        return cls(**config)

    def __repr__(self) -> str:
        """Return the constructor-like representation.

        ``max_value`` is omitted in one-sided mode.

        :return: A string naming the bounds and the inert flag.
        :rtype: str
        """
        if self.max_value is not None:
            return (f"ValueRangeConstraint(min_value={self.min_value}, "
                    f"max_value={self.max_value}, clip_gradients={self.clip_gradients})")
        else:
            return (f"ValueRangeConstraint(min_value={self.min_value}, "
                    f"clip_gradients={self.clip_gradients})")

# ---------------------------------------------------------------------
