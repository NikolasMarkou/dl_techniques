"""
Vector squashing, the activation of a Capsule Network.

This activation works on vectors, not scalars. It rescales each vector's
length into [0, 1) and leaves its direction alone. In a capsule network the
length is read as "does this entity exist" and the direction as "what are its
pose and other properties", so the two have to be decoupled: squashing changes
the first and never touches the second.

The function is::

    squash(v) = (||v||^2 / (1 + ||v||^2)) * (v / ||v||)

The right factor is the unit vector, which carries the direction. The left
factor is a scalar in [0, 1) that depends only on the squared norm. It rises
monotonically: near zero for a short vector, close to one for a long one. So
short vectors are almost annihilated and long ones are capped just under
length 1.

Measured with the default ``epsilon`` of 1e-7: an input vector of norm 0.1
comes out at 0.0099, norm 1.0 comes out at 0.5000, norm 10.0 at 0.9901, and
norm 1000.0 at 0.999999. Putting every capsule on the same bounded scale is
what makes the routing step downstream comparable across capsules.

References:
    - Sabour, S., Frosst, N., & Hinton, G. E. (2017). "Dynamic routing
      between capsules."
    - Hinton, G. E., Krizhevsky, A., & Wang, S. D. (2011). "Transforming
      auto-encoders." (Introduced the concept of capsules).

"""

import keras
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique


# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.activations.squash")
class SquashLayer(keras.layers.Layer):
    """Squashing non-linearity for Capsule Network vectors.

    Applies ``squash(v) = (||v||^2 / (1 + ||v||^2)) * (v / ||v||)`` along one
    axis. Vectors along that axis come out with the same direction and a
    length in [0, 1). Output shape equals input shape.

    The layer owns no weights. ``axis`` and ``epsilon`` are validated in
    ``__init__``, so a bad value fails at construction, not at the first call.

    **Architecture Overview:**

    .. code-block:: text

                            v  [..., D, ...]
                                    │
                                    ▼
                ┌───────────────────────────────────────┐
                │ s = sum(v*v) over axis, keepdims=True │
                └───────────────────┬───────────────────┘
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌───────────────────────┐       ┌───────────────────────┐
        │ scale = s / (1 + s)   │       │ unit = v/sqrt(s+eps)  │
        │ in [0, 1)             │       │ same direction as v   │
        └───────────┬───────────┘       └───────────┬───────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    │  scale * unit
                                    ▼
                             y  [same shape]
                                ||y|| < 1

    ``D`` is the size of the axis given by ``axis``; every other axis is
    untouched. Both branches read the same ``s``, computed once.

    :param axis: Axis holding each capsule vector. The norm is reduced over
        it. Defaults to -1, the last axis.
    :type axis: int
    :param epsilon: Small constant added under the square root so a zero
        vector does not divide by zero. If None, ``keras.config.epsilon()``
        is used, which is 1e-7. Must be positive.
    :type epsilon: Optional[float]
    :param kwargs: Additional keyword arguments passed to the Layer base class,
        such as ``name``, ``dtype``, ``trainable``, etc.

    :raises ValueError: If ``axis`` is not an ``int``, or if ``epsilon`` is
        given and is not positive.

    Note:
        The epsilon sits inside the square root, as ``sqrt(s + epsilon)``, not
        beside the norm as ``||v|| + epsilon``. An all-zero input vector
        therefore comes out exactly zero rather than ``NaN``, because the
        ``scale`` factor is zero too.

    References:
        - Sabour, S., Frosst, N., & Hinton, G. E. (2017). Dynamic routing between
          capsules. In Advances in neural information processing systems.
        - Hinton, G. E., Krizhevsky, A., & Wang, S. D. (2011). Transforming
          auto-encoders. In International conference on artificial neural networks.
    """

    def __init__(
            self,
            axis: int = -1,
            epsilon: Optional[float] = None,
            **kwargs: Any
    ) -> None:
        """Validate and store ``axis`` and ``epsilon``.

        :param axis: Axis holding each capsule vector.
        :type axis: int
        :param epsilon: Small constant for numerical stability. If None,
            ``keras.config.epsilon()`` (1e-7) is used. Must be positive.
        :type epsilon: Optional[float]
        :param kwargs: Additional keyword arguments for the Layer base class.
        :raises ValueError: If ``axis`` is not an ``int``, or if ``epsilon``
            is given and is not positive.
        """
        super().__init__(**kwargs)

        if not isinstance(axis, int):
            raise ValueError(f"axis must be an integer, got {type(axis).__name__}")
        if epsilon is not None and epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.axis = axis
        # Resolve None to a number here rather than at call time, so
        # get_config() records the value the layer was actually built with.
        self.epsilon = epsilon if epsilon is not None else keras.config.epsilon()

        logger.debug(f"Initialized SquashLayer with axis={axis}, epsilon={self.epsilon}")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Squash the vectors lying along ``self.axis``.

        Computes ``squash(v) = (||v||^2 / (1 + ||v||^2)) * (v / ||v||)``.

        :param inputs: Input tensor. Vectors are read along ``self.axis``.
        :type inputs: keras.KerasTensor
        :param training: Training or inference mode. Unused here; the layer
            behaves the same either way. Kept for API consistency.
        :type training: Optional[bool]
        :return: Tensor of the same shape as ``inputs``, whose vectors along
            ``self.axis`` have length in [0, 1).
        :rtype: keras.KerasTensor
        """
        # Squared L2 norm. keepdims=True keeps the reduced axis at size 1,
        # which is what lets scale and safe_norm broadcast back over inputs.
        squared_norm = keras.ops.sum(
            keras.ops.square(inputs),
            axis=self.axis,
            keepdims=True
        )

        # epsilon goes INSIDE the sqrt, so a zero vector gives a finite
        # denominator instead of a division by zero.
        safe_norm = keras.ops.sqrt(squared_norm + self.epsilon)

        # Bounded scale factor, always in [0, 1).
        scale = squared_norm / (1.0 + squared_norm)

        unit_vector = inputs / safe_norm

        return scale * unit_vector

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape, which equals the input shape.

        :param input_shape: Shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The same shape tuple, unchanged.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the config needed to rebuild the layer.

        ``epsilon`` is stored as a resolved number, never as ``None``, so a
        reloaded layer keeps the value it was built with even if the backend
        epsilon changes.

        :return: The base Layer config plus ``axis`` and ``epsilon``.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "epsilon": self.epsilon,
        })
        return config

    def __repr__(self) -> str:
        """Return a short representation showing the config and layer name.

        :return: A string such as
            ``SquashLayer(axis=-1, epsilon=1e-07, name='squash')``.
        :rtype: str
        """
        return f"SquashLayer(axis={self.axis}, epsilon={self.epsilon}, name='{self.name}')"

# ---------------------------------------------------------------------
