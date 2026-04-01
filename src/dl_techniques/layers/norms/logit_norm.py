"""
LogitNorm Layer for Classification Tasks
=======================================

This module implements LogitNorm, a normalization technique that applies L2 normalization
to logits with a learned temperature parameter. This helps stabilize training and can
improve model calibration by reducing overconfidence in predictions.

Mathematical Formulation:
------------------------
For input logits x with shape (..., d), LogitNorm computes:

    norm = sqrt(sum(x²) + ε)
    output = x / (norm * τ)

Where:
- sum(x²) is computed over specified axes (typically the class dimension)
- ε is a small epsilon for numerical stability
- τ is the temperature parameter that controls the spread of normalized logits

Key Benefits:
- **Improved Calibration**: Reduces model overconfidence
- **Training Stability**: L2 normalization prevents logit explosion
- **Temperature Scaling**: Learnable parameter for optimal calibration
- **Gradient Flow**: Maintains good gradient properties during backpropagation

References:
[1] "Mitigating Neural Network Overconfidence with Logit Normalization"
"""

import keras
from keras import ops
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class LogitNorm(keras.layers.Layer):
    """LogitNorm layer for classification tasks.

    Applies L2 normalization with a temperature parameter to logits, stabilizing
    training and improving model calibration by reducing overconfidence. The
    normalization is computed as:
    ``norm = sqrt(sum(logits²) + ε)``, ``output = logits / (norm × τ)``,
    where τ is the temperature controlling distribution sharpness.

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
        │  Sum along axis + ε     │
        │  norm² = Σ(x²) + ε     │
        └───────────┬─────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │  sqrt(norm²) → L2 norm  │
        └───────────┬─────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │  Divide: x / (norm × τ)│
        └───────────┬─────────────┘
                    │
                    ▼
        ┌─────────────────────────┐
        │   Normalized Logits     │
        │   shape: (..., C)       │
        └─────────────────────────┘

    :param temperature: Temperature scaling parameter. Higher values produce more
        spread-out logits, while lower values make the distribution sharper. Must
        be positive. Defaults to 0.04 (optimal for CIFAR-10 from original paper).
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

        :param temperature: Temperature scaling parameter. Higher values produce more
            spread-out logits. Must be positive.
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
        norm_squared = ops.sum(ops.square(inputs), axis=self.axis, keepdims=True)
        norm = ops.sqrt(ops.maximum(norm_squared, self.epsilon))

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
