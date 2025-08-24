"""
Adaptive Temperature Softmax Implementation

This implementation provides an enhanced version of the softmax function that dynamically
adjusts its temperature parameter based on the entropy of the input distribution. The
goal is to maintain sharpness in the output probabilities even as the input size grows,
addressing a fundamental limitation of standard softmax.

Standard Softmax Limitations:
- Dispersion Effect: As input size grows, probabilities tend towards 1/n
- Out-of-Distribution Behavior: Performance degrades on larger inputs than training
- Fixed Temperature: No adaptation to different input distributions

Key Features:
- Entropy-based temperature adaptation using polynomial mapping
- Selective application when entropy exceeds threshold
- Maintains computational efficiency for already-sharp distributions
- Backend-agnostic implementation using keras.ops

Performance improvements on max retrieval tasks:
- 512 items: 70.1% → 72.5% (+2.4%)
- 1024 items: 53.8% → 57.7% (+3.9%)
- 2048 items: 35.7% → 39.4% (+3.7%)

References:
- Original Paper: "Softmax is not enough (for sharp out-of-distribution), 2024"
"""

import keras
from keras import ops
from typing import Optional, Tuple, List, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class AdaptiveTemperatureSoftmax(keras.layers.Layer):
    """
    Adaptive Temperature Softmax layer with entropy-based temperature adaptation.

    This layer provides an enhanced version of the softmax function that dynamically
    adjusts its temperature parameter based on the entropy of the input distribution.
    The goal is to maintain sharpness in output probabilities even as input size grows,
    addressing fundamental limitations of standard softmax.

    The layer computes Shannon entropy of the initial softmax distribution and uses
    a polynomial mapping to determine appropriate temperature values. Temperature
    adaptation is only applied when entropy exceeds a specified threshold, maintaining
    efficiency and preserving sharp distributions.

    Mathematical formulation:
        1. p_initial = softmax(logits)
        2. H = -Σ p_i * log(p_i + ε)  # Shannon entropy
        3. T = polynomial(H) if H > threshold else 1.0
        4. p_output = softmax(logits / T)

    Args:
        min_temp: Float, minimum temperature value. Must be positive.
            Controls the sharpest possible output distribution. Defaults to 0.1.
        max_temp: Float, maximum temperature value. Must be positive and >= min_temp.
            Controls the smoothest possible output distribution. Defaults to 1.0.
        entropy_threshold: Float, entropy threshold for applying adaptation.
            Only applies temperature scaling when input entropy exceeds this value.
            Must be non-negative. Defaults to 0.5.
        eps: Float, small epsilon for numerical stability. If None, uses a small
            default value for safe logarithm computation. Defaults to None.
        polynomial_coeffs: Optional[List[float]], coefficients for polynomial
            temperature function ordered from highest to lowest degree.
            If None, uses empirically derived default coefficients.
            Defaults to None.
        **kwargs: Additional keyword arguments for Layer base class.

    Input shape:
        N-D tensor with shape (..., num_classes).
        The last dimension represents class logits.

    Output shape:
        Same shape as input. Values are probabilities that sum to 1.0 along last axis.

    Example:
        ```python
        # Basic usage
        layer = AdaptiveTemperatureSoftmax()
        logits = keras.random.normal((batch_size, num_classes))
        probabilities = layer(logits)

        # Custom configuration
        layer = AdaptiveTemperatureSoftmax(
            min_temp=0.05,
            max_temp=2.0,
            entropy_threshold=0.3
        )

        # In a model
        inputs = keras.Input(shape=(features,))
        logits = keras.layers.Dense(num_classes)(inputs)
        probabilities = AdaptiveTemperatureSoftmax()(logits)
        model = keras.Model(inputs, probabilities)
        ```

    Raises:
        ValueError: If min_temp <= 0, max_temp <= 0, min_temp > max_temp,
                   or entropy_threshold < 0.

    Note:
        This layer is stateless and does not contain trainable parameters.
        Temperature adaptation is computed dynamically based on input entropy.
        For efficiency, adaptation is only applied when needed (entropy > threshold).
    """

    def __init__(
        self,
        min_temp: float = 0.1,
        max_temp: float = 1.0,
        entropy_threshold: float = 0.5,
        eps: Optional[float] = None,
        polynomial_coeffs: Optional[List[float]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Input validation
        if min_temp <= 0.0:
            raise ValueError(f"min_temp must be positive, got {min_temp}")
        if max_temp <= 0.0:
            raise ValueError(f"max_temp must be positive, got {max_temp}")
        if min_temp > max_temp:
            raise ValueError(f"min_temp ({min_temp}) must be <= max_temp ({max_temp})")
        if entropy_threshold < 0.0:
            raise ValueError(f"entropy_threshold must be non-negative, got {entropy_threshold}")

        # Store configuration parameters
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.entropy_threshold = entropy_threshold
        self.eps = eps if eps is not None else 1e-7

        # Default polynomial coefficients for temperature adaptation
        # Coefficients ordered from highest to lowest degree: [x^4, x^3, x^2, x^1, x^0]
        self.polynomial_coeffs = polynomial_coeffs or [-1.791, 4.917, -2.3, 0.481, -0.037]

        logger.info(
            f"Initialized AdaptiveTemperatureSoftmax: min_temp={min_temp}, "
            f"max_temp={max_temp}, entropy_threshold={entropy_threshold}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer - validates input shape.

        Args:
            input_shape: Shape tuple of input tensor.
        """
        # Validate that we have at least 2 dimensions (batch + classes)
        if len(input_shape) < 2:
            raise ValueError(
                f"AdaptiveTemperatureSoftmax expects at least 2D input, "
                f"got shape {input_shape}"
            )

        # Validate that last dimension is defined (number of classes)
        if input_shape[-1] is None:
            raise ValueError(
                f"Last dimension (num_classes) must be defined, got shape {input_shape}"
            )

        super().build(input_shape)

    def _evaluate_polynomial(self, coeffs: List[float], x: keras.KerasTensor) -> keras.KerasTensor:
        """
        Evaluate polynomial using Horner's method for numerical stability.

        Args:
            coeffs: Polynomial coefficients from highest to lowest degree.
            x: Input tensor values.

        Returns:
            Polynomial evaluated at x.
        """
        if not coeffs:
            return ops.zeros_like(x)

        # Horner's method: efficient polynomial evaluation
        # P(x) = a_n * x^n + ... + a_1 * x + a_0
        # P(x) = (...((a_n * x + a_{n-1}) * x + a_{n-2}) * x + ... + a_1) * x + a_0
        result = ops.full_like(x, coeffs[0])

        for coeff in coeffs[1:]:
            result = result * x + coeff

        return result

    def _compute_entropy(self, probabilities: keras.KerasTensor) -> keras.KerasTensor:
        """
        Compute Shannon entropy: H = -Σ p_i * log(p_i + ε).

        Args:
            probabilities: Probability tensor with shape (..., num_classes).

        Returns:
            Entropy tensor with shape (..., 1) (keepdims=True).
        """
        # Clamp probabilities to avoid log(0)
        safe_probs = ops.clip(probabilities, self.eps, 1.0 - self.eps)

        # Compute entropy: H = -Σ p * log(p)
        log_probs = ops.log(safe_probs)
        entropy = -ops.sum(safe_probs * log_probs, axis=-1, keepdims=True)

        return entropy

    def _compute_adaptive_temperature(self, entropy: keras.KerasTensor) -> keras.KerasTensor:
        """
        Compute adaptive temperature using polynomial mapping.

        Temperature adaptation is only applied when entropy exceeds threshold.
        The polynomial maps entropy values to temperature range [min_temp, max_temp].

        Args:
            entropy: Entropy tensor with shape (..., 1).

        Returns:
            Temperature tensor with same shape as entropy.
        """
        # Determine which samples need adaptation
        needs_adaptation = entropy > self.entropy_threshold

        # Compute raw polynomial value
        poly_value = self._evaluate_polynomial(self.polynomial_coeffs, entropy)

        # Clamp polynomial output to [0, 1] range
        clamped_poly = ops.clip(poly_value, 0.0, 1.0)

        # Scale to [min_temp, max_temp] range
        temperature_range = self.max_temp - self.min_temp
        scaled_temp = self.min_temp + temperature_range * clamped_poly

        # Apply adaptation conditionally:
        # - If entropy > threshold: use adaptive temperature
        # - Otherwise: use standard temperature (1.0)
        temperature = ops.where(
            needs_adaptation,
            scaled_temp,
            ops.ones_like(scaled_temp)
        )

        return temperature

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply adaptive temperature softmax to input logits.

        Args:
            inputs: Input logits tensor with shape (..., num_classes).
            training: Boolean indicating training mode. Not used in this layer.

        Returns:
            Output probabilities tensor with same shape as input.
        """
        # Step 1: Compute initial probability distribution
        initial_probs = ops.nn.softmax(inputs, axis=-1)

        # Step 2: Compute Shannon entropy of initial distribution
        entropy = self._compute_entropy(initial_probs)

        # Step 3: Compute adaptive temperature based on entropy
        temperature = self._compute_adaptive_temperature(entropy)

        # Step 4: Scale logits with inverse temperature
        # Use safe division to prevent division by zero
        safe_temperature = ops.maximum(temperature, self.eps)
        scaled_logits = inputs / safe_temperature

        # Step 5: Apply final softmax to get adaptive probabilities
        output_probs = ops.nn.softmax(scaled_logits, axis=-1)

        return output_probs

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute output shape of the layer.

        Args:
            input_shape: Shape tuple of input tensor.

        Returns:
            Output shape tuple (same as input shape).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing all initialization parameters.
        """
        config = super().get_config()
        config.update({
            'min_temp': self.min_temp,
            'max_temp': self.max_temp,
            'entropy_threshold': self.entropy_threshold,
            'eps': self.eps,
            'polynomial_coeffs': self.polynomial_coeffs,
        })
        return config

# ---------------------------------------------------------------------
