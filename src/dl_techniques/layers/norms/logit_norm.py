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
    """
    LogitNorm layer for classification tasks.

    This layer implements logit normalization by applying L2 normalization with a learned
    temperature parameter. This helps stabilize training and can improve model calibration
    by reducing overconfidence in predictions.

    The normalization is computed as:

    .. math::
        \\text{norm} = \\sqrt{\\text{sum}(\\text{logits}^2) + \\varepsilon}

    .. math::
        \\text{output} = \\frac{\\text{logits}}{\\text{norm} \\times \\text{temperature}}

    Args:
        temperature: Temperature scaling parameter. Higher values produce more spread-out
            logits, while lower values make the distribution sharper. Must be positive.
            Defaults to 0.04 (optimal for CIFAR-10 from original paper).
        axis: Axis along which to perform normalization. Typically -1 for the class
            dimension. Defaults to -1.
        epsilon: Small constant for numerical stability. Must be positive.
            Defaults to 1e-7.
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Input shape:
        N-D tensor with shape: ``(..., num_classes)``

        The layer can handle any dimensionality, with normalization applied along
        the specified axis.

    Output shape:
        Same shape as input: ``(..., num_classes)``

    Raises:
        ValueError: If temperature is not positive.
        ValueError: If epsilon is not positive.

    Example:
        Basic usage in classification model:

        .. code-block:: python

            import keras
            from dl_techniques.layers.norms.logit_norm import LogitNorm

            # Standard classification model with LogitNorm
            inputs = keras.Input(shape=(784,))
            x = keras.layers.Dense(256, activation='relu')(inputs)
            x = keras.layers.Dense(128, activation='relu')(x)
            logits = keras.layers.Dense(10)(x)  # Raw logits

            # Apply LogitNorm before softmax
            normalized_logits = LogitNorm(temperature=0.04)(logits)
            outputs = keras.layers.Softmax()(normalized_logits)

            model = keras.Model(inputs, outputs)

        Custom temperature for different datasets:

        .. code-block:: python

            # For ImageNet (larger dataset, may need different temperature)
            logit_norm = LogitNorm(temperature=0.1)

            # For small datasets (may need smaller temperature)
            logit_norm = LogitNorm(temperature=0.01)

        Integration with mixed precision:

        .. code-block:: python

            # LogitNorm works well with mixed precision training
            keras.mixed_precision.set_global_policy('mixed_float16')

            inputs = keras.Input(shape=(224, 224, 3), dtype='float16')
            # ... feature extraction layers ...
            logits = keras.layers.Dense(1000)(features)

            # LogitNorm helps with numerical stability in fp16
            normalized = LogitNorm(temperature=0.05, epsilon=1e-6)(logits)
            outputs = keras.layers.Softmax(dtype='float32')(normalized)

        Calibration-aware training:

        .. code-block:: python

            def create_calibrated_classifier(num_classes, temperature=0.04):
                inputs = keras.Input(shape=(input_dim,))

                # Feature extraction
                features = keras.layers.Dense(512, activation='relu')(inputs)
                features = keras.layers.Dropout(0.5)(features)
                features = keras.layers.Dense(256, activation='relu')(features)

                # Raw logits
                logits = keras.layers.Dense(num_classes)(features)

                # LogitNorm for calibration
                normalized_logits = LogitNorm(temperature=temperature)(logits)
                probabilities = keras.layers.Softmax()(normalized_logits)

                return keras.Model(inputs, probabilities)

        Multi-axis normalization:

        .. code-block:: python

            # For multi-label classification (normalize each label separately)
            inputs = keras.Input(shape=(sequence_length, feature_dim))
            # ... processing layers ...
            logits = keras.layers.Dense(num_labels)(processed)

            # Normalize along the feature dimension
            normalized = LogitNorm(axis=-1, temperature=0.1)(logits)

    Note:
        - This layer performs only computation on inputs without creating any weights
        - Temperature parameter is fixed at initialization (not learnable by default)
        - For learnable temperature, consider using a separate Dense layer with sigmoid activation
        - Works well with mixed precision training due to numerical stability improvements
    """

    def __init__(
            self,
            temperature: float = 0.04,  # Default from paper for CIFAR-10
            axis: int = -1,
            epsilon: float = 1e-7,
            **kwargs: Any
    ) -> None:
        """
        Initialize the LogitNorm layer.

        Args:
            temperature: Temperature scaling parameter. Higher values produce more
                spread-out logits. Must be positive.
            axis: Axis along which to perform normalization.
            epsilon: Small constant for numerical stability. Must be positive.
            **kwargs: Additional keyword arguments for the Layer parent class.

        Raises:
            ValueError: If temperature is not positive.
            ValueError: If epsilon is not positive.
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
        """
        Validate initialization parameters.

        Args:
            temperature: Temperature parameter to validate.
            epsilon: Epsilon parameter to validate.

        Raises:
            ValueError: If parameters are invalid.
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
        """
        Apply logit normalization to inputs.

        This method implements L2 normalization with temperature scaling:
        1. Compute L2 norm along specified axis
        2. Normalize inputs by the L2 norm
        3. Scale by temperature parameter

        Args:
            inputs: Input logits tensor of any shape.
            training: Boolean indicating whether in training mode (unused, kept for
                API compatibility).

        Returns:
            Normalized logits tensor with the same shape as inputs.
        """
        # Compute L2 norm along specified axis with numerical stability
        # Use maximum to prevent sqrt of values smaller than epsilon
        norm_squared = ops.sum(ops.square(inputs), axis=self.axis, keepdims=True)
        norm = ops.sqrt(ops.maximum(norm_squared, self.epsilon))

        # Normalize logits and scale by temperature
        # Division by temperature controls the "sharpness" of the distribution
        return inputs / (norm * self.temperature)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple (same as input shape for normalization layers).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Following modern Keras 3 patterns, this method returns ALL constructor
        arguments needed to recreate this layer instance.

        Returns:
            Dictionary containing all constructor arguments.
        """
        config = super().get_config()
        config.update({
            "temperature": self.temperature,
            "axis": self.axis,
            "epsilon": self.epsilon,
        })
        return config

# ---------------------------------------------------------------------
