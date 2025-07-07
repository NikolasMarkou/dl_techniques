"""
Adaptive Temperature Softmax Implementation
========================================

Overview:
---------
This implementation provides an enhanced version of the softmax function that dynamically
adjusts its temperature parameter based on the entropy of the input distribution. The
goal is to maintain sharpness in the output probabilities even as the input size grows,
addressing a fundamental limitation of standard softmax.

Standard Softmax Limitations:
---------------------------
1. Dispersion Effect:
   - As input size (n) grows, probabilities tend towards 1/n
   - Theoretical proof shows this is inevitable for any non-zero temperature
   - Makes it difficult to maintain sharp focus on specific inputs

2. Out-of-Distribution Behavior:
   - Performance degrades significantly on larger input sizes than seen during training
   - Standard softmax cannot adapt to varying input complexities
   - No mechanism to adjust sharpness based on input characteristics

3. Fixed Temperature:
   - Standard temperature parameter must be chosen at training time
   - No way to adapt to different input distributions
   - Trade-off between sharpness and stability cannot be dynamically balanced

Implementation Details:
---------------------
1. Entropy Computation:
   - First applies standard softmax to get initial probability distribution
   - Computes Shannon entropy: H = -Σ p_i * log(p_i)
   - Uses epsilon padding to avoid numerical instability with log(0)
   - Entropy provides measure of how dispersed current distribution is

2. Temperature Adaptation:
   - Uses polynomial fit based on empirical data
   - Coefficients: [-0.037, 0.481, -2.3, 4.917, -1.791]
   - Maps entropy to temperature value
   - Constrains temperature to [min_temp, max_temp] range
   - Never increases temperature above 1.0 to avoid additional dispersion

3. Selective Application:
   - Only adapts temperature when entropy exceeds threshold (default 0.5)
   - Maintains original behavior for already-sharp distributions
   - Smooth transition between adapted and non-adapted regions
   - Computationally efficient by avoiding unnecessary adaptations

4. Final Computation:
   - Scales logits by inverse temperature: logits * (1/T)
   - Applies final softmax to get output probabilities
   - Maintains proper probability distribution (sums to 1.0)

References:
----------
1. Original Paper: "Softmax is not enough (for sharp out-of-distribution), 2024"
"""

import keras
from keras import ops
from typing import Optional, Dict, Any, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class AdaptiveTemperatureSoftmax(keras.layers.Layer):
    """Adaptive Temperature Softmax layer.

    This layer provides an enhanced version of the softmax function that dynamically
    adjusts its temperature parameter based on the entropy of the input distribution.
    The goal is to maintain sharpness in the output probabilities even as the input
    size grows, addressing a fundamental limitation of standard softmax.

    The layer computes Shannon entropy of the initial softmax distribution and uses
    a polynomial mapping to determine an appropriate temperature value. Temperature
    adaptation is only applied when entropy exceeds a specified threshold, maintaining
    efficiency and preserving sharp distributions.

    Performance improvements on max retrieval tasks:
    - 512 items: 70.1% → 72.5% (+2.4%)
    - 1024 items: 53.8% → 57.7% (+3.9%)
    - 2048 items: 35.7% → 39.4% (+3.7%)

    Args:
        min_temp: Minimum temperature value. Must be positive. Default: 0.1
        max_temp: Maximum temperature value. Must be positive and >= min_temp. Default: 1.0
        entropy_threshold: Entropy threshold for applying adaptation. Default: 0.5
        eps: Small epsilon value for numerical stability. Default: keras.backend.epsilon()
        polynomial_coeffs: Optional list of coefficients for the polynomial temperature
            function. If None, uses empirically derived default coefficients.
        **kwargs: Additional layer arguments passed to the base Layer class.

    Raises:
        ValueError: If temperature values are invalid or min_temp > max_temp.

    Example:
        >>> # Basic usage
        >>> layer = AdaptiveTemperatureSoftmax()
        >>> logits = keras.random.normal((2, 10))
        >>> probs = layer(logits)
        >>> print(probs.shape)  # (2, 10)

        >>> # Custom parameters
        >>> layer = AdaptiveTemperatureSoftmax(
        ...     min_temp=0.05,
        ...     max_temp=2.0,
        ...     entropy_threshold=0.3
        ... )
    """

    def __init__(
        self,
        min_temp: float = 0.1,
        max_temp: float = 1.0,
        entropy_threshold: float = 0.5,
        eps: float = None,
        polynomial_coeffs: Optional[List[float]] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # Validation
        if min_temp <= 0 or max_temp <= 0:
            raise ValueError("Temperature values must be positive")
        if min_temp > max_temp:
            raise ValueError("min_temp must be less than or equal to max_temp")
        if entropy_threshold < 0:
            raise ValueError("entropy_threshold must be non-negative")

        # Store configuration parameters
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.entropy_threshold = entropy_threshold
        self.eps = eps if eps is not None else keras.backend.epsilon()

        # Default polynomial coefficients for temperature adaptation
        # Coefficients are ordered from highest to lowest degree
        self.polynomial_coeffs = polynomial_coeffs or [-1.791, 4.917, -2.3, 0.481, -0.037]

        # Store build input shape for serialization
        self._build_input_shape = None

        logger.info(
            f"Initialized AdaptiveTemperatureSoftmax with min_temp={min_temp}, "
            f"max_temp={max_temp}, entropy_threshold={entropy_threshold}"
        )

    def build(self, input_shape) -> None:
        """Build the layer based on input shape.

        Args:
            input_shape: Shape tuple indicating the input shape of the layer.
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        logger.debug(f"Building AdaptiveTemperatureSoftmax with input_shape: {input_shape}")

        super().build(input_shape)

    def _polyval(self, coeffs: List[float], x) -> Any:
        """Evaluate polynomial with given coefficients using Horner's method.

        This is a backend-agnostic implementation of polynomial evaluation
        since tf.math.polyval is not available in keras.ops.

        Args:
            coeffs: List of polynomial coefficients from highest to lowest degree.
            x: Input tensor values.

        Returns:
            Polynomial evaluated at x.
        """
        if not coeffs:
            return ops.zeros_like(x)

        # Start with the highest degree coefficient
        result = ops.full_like(x, coeffs[0])

        # Apply Horner's method: result = result * x + coeff
        for coeff in coeffs[1:]:
            result = result * x + coeff

        return result

    def compute_entropy(self, probs) -> Any:
        """Compute Shannon entropy across the last dimension of probabilities.

        Args:
            probs: Probability tensor with shape (..., num_classes).

        Returns:
            Entropy tensor with shape (..., 1) (keepdims=True).
        """
        # Compute -sum(p * log(p + eps)) along last axis
        log_probs = ops.log(probs + self.eps)
        entropy = -ops.sum(probs * log_probs, axis=-1, keepdims=True)
        return entropy

    def compute_temperature(self, entropy) -> Any:
        """Compute adaptive temperature using polynomial fit.

        The temperature is computed using a polynomial mapping from entropy values,
        then clamped to the specified range. Temperature adaptation is only applied
        when entropy exceeds the threshold.

        Args:
            entropy: Entropy tensor with shape (..., 1).

        Returns:
            Temperature tensor with same shape as entropy, values in [min_temp, max_temp].
        """
        # 1. Identify if adaptation is needed based on threshold
        should_adapt = ops.cast(entropy > self.entropy_threshold, entropy.dtype)

        # 2. Compute raw polynomial value using custom polyval
        raw_temp = self._polyval(self.polynomial_coeffs, entropy)

        # 3. Clamp polynomial to [0, 1]
        raw_temp = ops.clip(raw_temp, 0.0, 1.0)

        # 4. Scale up to [min_temp, max_temp]
        scaled_temp = self.min_temp + (self.max_temp - self.min_temp) * raw_temp

        # 5. Apply adaptation conditionally:
        # - If entropy > threshold: use scaled temperature
        # - Otherwise: use standard softmax (T=1.0)
        temperature = ops.where(should_adapt > 0, scaled_temp, ops.ones_like(scaled_temp))

        return temperature

    def call(self, logits, training=None):
        """Apply adaptive temperature softmax to input logits.

        Args:
            logits: Input logits tensor with shape (..., num_classes).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Not used in this layer.

        Returns:
            Output probabilities tensor with same shape as input.
        """
        # 1. Initial probability distribution using standard softmax
        probs = ops.nn.softmax(logits, axis=-1)
        probs = ops.clip(probs, self.eps, 1.0 - self.eps)

        # 2. Compute Shannon entropy
        entropy = self.compute_entropy(probs)

        # 3. Compute adaptive temperature
        temperature = self.compute_temperature(entropy)

        # 4. Scale logits with inverse temperature
        # Use safe division to avoid division by zero
        inv_temp = ops.divide(1.0, temperature + self.eps)
        scaled_logits = logits * inv_temp

        # 5. Apply final softmax to get output probabilities
        output_probs = ops.nn.softmax(scaled_logits, axis=-1)

        logger.debug(
            f"Applied adaptive temperature softmax: "
            f"entropy_mean={ops.mean(entropy):.4f}, "
            f"temp_mean={ops.mean(temperature):.4f}"
        )

        return output_probs

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Output shape (same as input shape).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
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

    def get_build_config(self) -> Dict[str, Any]:
        """Get the config needed to build the layer from a config.

        This method is needed for proper model saving and loading.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------

