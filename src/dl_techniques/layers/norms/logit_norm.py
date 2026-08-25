"""
LogitNorm Layer for Classification Tasks
=======================================

This module implements LogitNorm, a normalization technique that applies L2 normalization
to logits with a fixed temperature hyperparameter (not learned). This helps stabilize
training and can improve model calibration by reducing overconfidence in predictions.

Mathematical Formulation:
------------------------
For input logits x with shape (..., d), LogitNorm computes:

    norm = sqrt(max(sum(x²), ε))
    output = x / (norm * τ)

Where:
- sum(x²) is computed over specified axes (typically the class dimension)
- ε is a small epsilon floor (via ``max``) for numerical stability
- τ is the temperature parameter (a fixed hyperparameter). It DIVIDES the
  unit-norm logits, so a LARGER τ compresses the output toward zero and a SMALLER τ
  expands it: measured on ``x = [1, 2, 3, 4]``, ``τ = 1.0`` gives an output range of
  ``[0.183, 0.730]`` while ``τ = 0.01`` gives ``[18.26, 73.03]``

Key Benefits:
- **Improved Calibration**: Reduces model overconfidence
- **Training Stability**: L2 normalization prevents logit explosion
- **Temperature Scaling**: Fixed hyperparameter; the output is the unit-norm logit
  vector divided by τ, so smaller τ means larger logits and a sharper softmax
- **Gradient Flow**: Maintains good gradient properties during backpropagation

References:
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

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class LogitNorm(keras.layers.Layer):
    """LogitNorm layer for classification tasks.

    Applies L2 normalization with a temperature parameter to logits, stabilizing
    training and improving model calibration by reducing overconfidence. The
    normalization is computed as:
    ``norm = sqrt(max(sum(logits²), ε))``, ``output = logits / (norm × τ)``,
    where τ is the (fixed) temperature. τ divides the unit-norm logits, so a
    LARGER τ compresses the output and a SMALLER τ expands it.

    ``supports_masking`` is decided from the RESOLVED normalization axis, not set
    unconditionally: it is ``True`` only while every normalized axis is the trailing
    (feature) axis of the input. At ``axis=-1`` each position is normalized independently of
    the others (measured cross-position leak exactly ``0.0`` on a ``(3, 5, 8)``
    input).
    Normalizing over the TOKEN axis instead couples positions - measured leak
    ``23.068`` at ``axis=1`` on the same input - and there the flag is ``False``, so
    Keras drops the mask and says so. The decision is made in ``__init__`` from the
    spelling (only ``-1`` is rank-independent) and made exact in ``build()``.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────┐
        │    Input Logits (x)     │
        │   shape: (..., C)       │
        └───────────┬─────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │   Square: x²            │
        └───────────┬─────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │  Sum, floor at ε        │
        │  norm² = max(Σ(x²), ε)  │
        └───────────┬─────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │  sqrt(norm²) → L2 norm  │
        └───────────┬─────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │  Divide: x / (norm × τ) │
        └───────────┬─────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │   Normalized Logits     │
        │   shape: (..., C)       │
        └─────────────────────────┘

    :param temperature: Temperature scaling parameter. It is the DIVISOR applied to
        the unit-norm logits (``output = logits / (norm × τ)``), so a HIGHER value
        compresses the output toward zero and a LOWER value expands it - measured on
        ``x = [1, 2, 3, 4]``: ``τ = 1.0`` gives an output range of ``[0.183, 0.730]``,
        ``τ = 0.01`` gives ``[18.26, 73.03]``. The default 0.04 therefore multiplies
        the unit-norm logits by 25, which sharpens the resulting softmax. Must be
        positive. Defaults to 0.04 (optimal for CIFAR-10 from original paper).
    :type temperature: float
    :param axis: Axis along which to perform normalization. Typically -1 for the
        class dimension. Defaults to -1.
    :type axis: int
    :param epsilon: Small constant for numerical stability. Must be positive.
        Defaults to 1e-7.
    :type epsilon: float

    :raises ValueError: If temperature is not positive.
    :raises ValueError: If epsilon is not positive.
    """

    def __init__(
            self,
            temperature: float = 0.04,  # Default from paper for CIFAR-10
            axis: int = -1,
            epsilon: float = 1e-7,
            **kwargs: Any
    ) -> None:
        """Initialize the LogitNorm layer.

        :param temperature: Temperature scaling parameter; the divisor applied to
            the unit-norm logits, so higher values compress the output and lower
            values expand it. Must be positive.
        :type temperature: float
        :param axis: Axis along which to perform normalization.
        :type axis: int
        :param epsilon: Small constant for numerical stability. Must be positive.
        :type epsilon: float
        :param kwargs: Additional keyword arguments for the Layer parent class.
        :type kwargs: Any

        :raises ValueError: If temperature is not positive.
        :raises ValueError: If epsilon is not positive.
        """
        super().__init__(**kwargs)

        # Validate inputs early
        self._validate_inputs(temperature, epsilon)

        # Store ALL configuration parameters - required for get_config()
        self.temperature = temperature
        self.axis = axis
        self.epsilon = epsilon

        # supports_masking is a promise about the AXIS, not about the class: it holds
        # only while the normalized axis is the trailing (feature) axis. Decided here
        # from the spelling alone - `-1` names the trailing axis at every rank - and
        # made exact in build(), where the input rank is finally known.
        self.supports_masking = normalizes_only_the_feature_axis(axis)

        logger.debug(f"Initialized LogitNorm with temperature={temperature}, axis={axis}, epsilon={epsilon}")

    def _validate_inputs(self, temperature: float, epsilon: float) -> None:
        """Validate initialization parameters.

        :param temperature: Temperature parameter to validate.
        :type temperature: float
        :param epsilon: Epsilon parameter to validate.
        :type epsilon: float

        :raises ValueError: If parameters are invalid.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Decide ``supports_masking`` against the now-known input rank.

        The layer owns no weights, so this override exists solely to make the
        masking promise exact: ``axis`` may be spelled non-negatively, and whether
        it names the feature axis or the token axis depends on the rank.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        # Refine the __init__ estimate now that the rank is known. Keras reads
        # supports_masking inside __call__, which runs build() first, so this is the
        # value that decides whether the mask actually survives.
        self.supports_masking = normalizes_only_the_feature_axis(
            self.axis, rank=len(input_shape)
        )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply logit normalization to inputs.

        Computes L2 normalization along the specified axis and scales by
        the temperature parameter.

        :param inputs: Input logits tensor of any shape.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating whether in training mode (unused,
            kept for API compatibility).
        :type training: Optional[bool]

        :return: Normalized logits tensor with the same shape as inputs.
        :rtype: keras.KerasTensor
        """
        # Compute L2 norm along specified axis with numerical stability
        # Use maximum to prevent sqrt of values smaller than epsilon
        norm_squared = keras.ops.sum(keras.ops.square(inputs), axis=self.axis, keepdims=True)
        norm = keras.ops.sqrt(keras.ops.maximum(norm_squared, self.epsilon))

        # Normalize logits and scale by temperature
        # Division by temperature controls the "sharpness" of the distribution
        return inputs / (norm * self.temperature)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]

        :return: Output shape tuple (same as input shape for normalization layers).
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all constructor arguments.
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
