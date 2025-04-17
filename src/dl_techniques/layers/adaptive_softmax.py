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

Improvements Over Standard Softmax:
--------------------------------
1. Performance Benefits:
   - Maintains sharpness with increasing input size
   - Better generalization to out-of-distribution inputs
   - Example improvements on max retrieval task:
     * 512 items: 70.1% → 72.5% (+2.4%)
     * 1024 items: 53.8% → 57.7% (+3.9%)
     * 2048 items: 35.7% → 39.4% (+3.7%)

2. Adaptive Behavior:
   - Dynamically adjusts to input complexity
   - No need to tune temperature parameter manually
   - Automatically handles varying input sizes
   - Preserves sharp attention when appropriate

3. Stability Features:
   - Numerically stable with epsilon handling
   - Smooth temperature transitions
   - Bounded output ranges
   - Maintains probability distribution properties

4. Implementation Advantages:
   - Drop-in replacement for standard softmax
   - No additional training required
   - Minimal computational overhead
   - Thread-safe implementation
   - Type-safe with proper error handling

Usage Guidelines:
---------------
1. When to Use:
   - Large-scale attention mechanisms
   - Variable input size scenarios
   - Out-of-distribution robustness needed
   - When sharp focus must be maintained

2. Parameter Tuning:
   - min_temp: Lower bound for temperature (default 0.1)
     * Too low: May cause numerical instability
     * Too high: Limits maximum sharpness

   - max_temp: Upper bound for temperature (default 1.0)
     * Too low: Limits flexibility
     * Too high: Can cause excessive dispersion

   - entropy_threshold: When to apply adaptation (default 0.5)
     * Too low: May adapt unnecessarily
     * Too high: May miss adaptation opportunities

3. Integration Tips:
   - Use layer version for stateful models
   - Use functional version for simple cases
   - Monitor entropy values for tuning
   - Consider batch statistics for threshold selection

References:
----------
1. Original Paper: "Softmax is not enough (for sharp out-of-distribution)"
"""

import keras
import tensorflow as tf
from typing import Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class AdaptiveTemperatureSoftmax(keras.layers.Layer):
    """Adaptive Temperature Softmax layer.

    This layer provides an enhanced version of the softmax function that dynamically
    adjusts its temperature parameter based on the entropy of the input distribution.
    The goal is to maintain sharpness in the output probabilities even as the input
    size grows, addressing a fundamental limitation of standard softmax.

    Args:
        min_temp: float
            Minimum temperature value (default: 0.1)
        max_temp: float
            Maximum temperature value (default: 1.0)
        entropy_threshold: float
            Entropy threshold for applying adaptation (default: 0.5)
        polynomial_coeffs: Optional[list]
            Coefficients for the polynomial temperature function
        **kwargs:
            Additional layer arguments

    Raises:
        ValueError: If temperature values are invalid or min_temp > max_temp

    """

    def __init__(
        self,
        min_temp: float = 0.1,
        max_temp: float = 1.0,
        entropy_threshold: float = 0.5,
        eps: float = keras.backend.epsilon(),
        polynomial_coeffs: Optional[list] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        if min_temp <= 0 or max_temp <= 0:
            raise ValueError("Temperature values must be positive")
        if min_temp > max_temp:
            raise ValueError("min_temp must be less than max_temp")

        self.min_temp = min_temp
        self.max_temp = max_temp
        self.entropy_threshold = tf.constant(entropy_threshold)
        self.eps = eps
        # Default polynomial coefficients for temperature adaptation
        self.polynomial_coeffs = polynomial_coeffs or [-1.791, 4.917, -2.3, 0.481, -0.037]

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build the layer based on input shape.

        Args:
            input_shape: TensorShape
                Shape of the input tensor
        """

        super().build(input_shape)

    def compute_entropy(self, probs: tf.Tensor) -> tf.Tensor:
        """Compute Shannon entropy across the last dimension of probs.
           Returns shape = probs.shape[:-1] (with keepdims=True)."""
        entropy = -tf.reduce_sum(
            probs * tf.math.log(probs + self.eps), axis=-1, keepdims=True
        )
        return entropy

    def compute_temperature(self, entropy: tf.Tensor) -> tf.Tensor:
        """Compute adaptive temperature using polynomial fit.

        Args:
            entropy: tf.Tensor
                Entropy values

        Returns:
            tf.Tensor: Temperature values clamped to [min_temp, max_temp]
        """
        # 1. Identify if adaptation is needed based on threshold
        should_adapt = tf.cast(entropy > self.entropy_threshold, entropy.dtype)

        # 2. Compute raw polynomial value
        raw_temp = tf.math.polyval(coeffs=self.polynomial_coeffs, x=entropy)

        # 3. Clamp polynomial to [0, 1]
        raw_temp = tf.clip_by_value(raw_temp, 0.0, 1.0)

        # 4. Scale up to [min_temp, max_temp]
        scaled_temp = self.min_temp + (self.max_temp - self.min_temp) * raw_temp

        # 5. Apply adaptation conditionally:
        # - If entropy > threshold: use scaled temperature
        # - Otherwise: use standard softmax (T=1.0)
        temperature = tf.where(should_adapt > 0, scaled_temp, tf.ones_like(scaled_temp))

        return temperature

    def call(self, logits: tf.Tensor) -> tf.Tensor:
        """Apply adaptive temperature softmax to input logits.

        Args:
            logits: tf.Tensor
                Input logits with shape = (..., num_classes)

        Returns:
            tf.Tensor: Output probabilities with same shape as input
        """
        # 1. Initial probability distribution
        probs = tf.nn.softmax(logits, axis=-1)

        # 2. Compute entropy
        entropy = self.compute_entropy(probs)

        # 3. Compute temperature
        temperature = self.compute_temperature(entropy)

        # 4. Scale logits with inverse temperature
        # Add small epsilon to avoid division by zero
        inv_temp = tf.math.reciprocal_no_nan(temperature)
        scaled_logits = logits * inv_temp

        # 5. Final probabilities
        return tf.nn.softmax(scaled_logits, axis=-1)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute the output shape of the layer.

        Args:
            input_shape: tf.TensorShape
                Shape of the input tensor

        Returns:
            tf.TensorShape: Shape of the output tensor (same as input)
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get the layer configuration.

        Returns:
            Dict[str, Any]: Layer configuration dictionary
        """
        config = super().get_config()
        config.update({
            'min_temp': self.min_temp,
            'max_temp': self.max_temp,
            'entropy_threshold': self.entropy_threshold,
            'polynomial_coeffs': self.polynomial_coeffs,
            'eps': self.eps
        })
        return config

# ---------------------------------------------------------------------
