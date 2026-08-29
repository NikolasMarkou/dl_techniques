"""LogitNorm: L2 normalization of logits with a fixed temperature.

``LogitNorm`` divides a logit vector by its own L2 norm and then by a fixed
temperature. It owns no weights, and the temperature is a hyperparameter rather
than a learned value. The published motivation is calibration: bounding the
logit magnitude stops the softmax growing arbitrarily confident.

Computation
-----------

For an input ``x`` reduced over ``axis``::

    norm = sqrt(max(sum(x ** 2), epsilon))
    output = x / (norm * temperature)

``sum(x ** 2)`` sums over ``axis``, so this is an L2 norm and not an RMS. The
output magnitude therefore depends on the length of the normalized axis.
``epsilon`` is a floor applied with ``maximum`` before the square root, not a
term added inside it.

``temperature`` DIVIDES the unit-norm logits. A larger value compresses the
output toward zero and a smaller value expands it. Measured on
``x = [1, 2, 3, 4]``: ``temperature = 1.0`` gives an output range of
``[0.183, 0.730]``, and ``temperature = 0.01`` gives ``[18.257, 73.030]``. The
default 0.04 multiplies the unit-norm logits by 25, which sharpens the softmax
that follows.

References
----------

[1] "Mitigating Neural Network Overconfidence with Logit Normalization"
"""

import keras
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms._masking import (
    normalizes_only_the_feature_axis,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.norms.logit_norm")
class LogitNorm(keras.layers.Layer):
    """L2-normalize logits, then divide by a fixed temperature.

    Reduces ``sum(x ** 2)`` over ``axis``, floors it at ``epsilon``, takes the
    square root, and divides the input by that norm times ``temperature``. The
    layer owns no weights and the output has the same shape as the input.

    ``temperature`` is a divisor, so a larger value compresses the output and a
    smaller value expands it. Measured on ``x = [1, 2, 3, 4]``:
    ``temperature = 1.0`` gives ``[0.183, 0.730]`` and ``temperature = 0.01``
    gives ``[18.257, 73.030]``.

    ``supports_masking`` is a promise about the AXIS, not about the class. It is
    ``True`` only while the normalized axis is the trailing (feature) axis. At
    ``axis=-1`` every position is normalized on its own, and the measured
    cross-position leak on a ``(3, 5, 8)`` input is exactly ``0.0``. Normalizing
    the token axis couples positions: the measured leak at ``axis=1`` on the same
    input is ``23.068``. The flag is ``False`` there, so Keras drops the mask and
    says so. ``__init__`` decides from the spelling alone, because only ``-1``
    names the trailing axis at every rank. ``build()`` then makes it exact.

    **Architecture Overview:**

    .. code-block:: text

             inputs: (..., C)
                     │
                     ├────────────────────────┐
                     │                        │
                     ▼                        │
        ┌─────────────────────────┐           │
        │ square, sum over axis,  │           │
        │ keepdims=True           │           │
        └────────────┬────────────┘           │
                     │ norm_squared: (..., 1) │
                     ▼                        │
        ┌─────────────────────────┐           │
        │ maximum(., epsilon),    │           │
        │ then sqrt               │           │
        └────────────┬────────────┘           │
                     │ norm: (..., 1)         │
                     ▼                        ▼
        ┌────────────────────────────────────────────────┐
        │ divide: inputs / (norm * temperature)          │
        └────────────────────────┬───────────────────────┘
                                 │
                                 ▼
                         output: (..., C)

    :param temperature: Divisor applied to the unit-norm logits. Higher values
        compress the output toward zero, lower values expand it. Must be
        positive. Defaults to 0.04, the value the original paper reports for
        CIFAR-10.
    :type temperature: float
    :param axis: Axis reduced by the L2 norm. Usually -1, the class dimension.
        Defaults to -1.
    :type axis: int
    :param epsilon: Floor applied to the squared norm before the square root.
        Must be positive. Defaults to 1e-7.
    :type epsilon: float

    :ivar temperature: The configured temperature.
    :vartype temperature: float
    :ivar axis: The configured normalization axis.
    :vartype axis: int
    :ivar epsilon: The configured numerical floor.
    :vartype epsilon: float

    :raises ValueError: If temperature is not positive.
    :raises ValueError: If epsilon is not positive.

    Example:

    .. code-block:: python

        import keras
        from dl_techniques.layers.norms import LogitNorm

        logits = keras.random.normal((4, 10))
        normalized = LogitNorm(temperature=0.04)(logits)
    """

    def __init__(
            self,
            temperature: float = 0.04,
            axis: int = -1,
            epsilon: float = 1e-7,
            **kwargs: Any
    ) -> None:
        """Initialize the layer.

        :param temperature: Divisor applied to the unit-norm logits. Must be
            positive.
        :type temperature: float
        :param axis: Axis reduced by the L2 norm.
        :type axis: int
        :param epsilon: Floor applied to the squared norm. Must be positive.
        :type epsilon: float
        :param kwargs: Additional keyword arguments for ``keras.layers.Layer``.
        :type kwargs: Any

        :raises ValueError: If temperature is not positive.
        :raises ValueError: If epsilon is not positive.
        """
        super().__init__(**kwargs)

        self._validate_inputs(temperature, epsilon)

        # Every constructor argument is stored, because get_config() must return
        # all of them.
        self.temperature = temperature
        self.axis = axis
        self.epsilon = epsilon

        # supports_masking is a promise about the AXIS, not about the class. It
        # holds only while the normalized axis is the trailing (feature) axis.
        # Only the spelling `-1` names that axis at every rank, so that is all
        # __init__ can decide. build() makes the answer exact.
        self.supports_masking = normalizes_only_the_feature_axis(axis)

        logger.debug(f"Initialized LogitNorm with temperature={temperature}, axis={axis}, epsilon={epsilon}")

    def _validate_inputs(self, temperature: float, epsilon: float) -> None:
        """Reject non-positive constructor arguments.

        :param temperature: Temperature to validate.
        :type temperature: float
        :param epsilon: Epsilon to validate.
        :type epsilon: float

        :raises ValueError: If temperature is not positive.
        :raises ValueError: If epsilon is not positive.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Decide ``supports_masking`` against the now-known input rank.

        The layer owns no weights. This override exists only to make the masking
        promise exact. ``axis`` may be spelled non-negatively, and whether such a
        spelling names the feature axis or the token axis depends on the rank.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        # Refine the __init__ estimate now that the rank is known. Keras reads
        # supports_masking inside __call__, which runs build() first, so this is
        # the value that decides whether the mask survives.
        self.supports_masking = normalizes_only_the_feature_axis(
            self.axis, rank=len(input_shape)
        )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply logit normalization.

        :param inputs: Input logits tensor of any shape.
        :type inputs: keras.KerasTensor
        :param training: Training-mode flag. Unused; the layer behaves the same
            in both modes and the argument is kept for API compatibility.
        :type training: Optional[bool]

        :return: Normalized logits, same shape as ``inputs``.
        :rtype: keras.KerasTensor
        """
        # maximum() floors the squared norm, so sqrt never sees a value below
        # epsilon.
        norm_squared = keras.ops.sum(keras.ops.square(inputs), axis=self.axis, keepdims=True)
        norm = keras.ops.sqrt(keras.ops.maximum(norm_squared, self.epsilon))

        # Dividing by temperature controls how sharp the following softmax is.
        return inputs / (norm * self.temperature)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Return the output shape, which equals the input shape.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]

        :return: The same shape tuple that was passed in.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration needed to rebuild this layer.

        :return: Dictionary holding every constructor argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "temperature": self.temperature,
            "axis": self.axis,
            "epsilon": self.epsilon,
        })
        return config

# ---------------------------------------------------------------------
