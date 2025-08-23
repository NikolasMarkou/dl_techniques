"""ThreshMax activation layer with differentiable step function.

This module implements a sparse softmax variant that creates sparsity through
confidence thresholding while maintaining smooth gradients via a differentiable
step function. The layer helps create more confident and sparse probability
distributions compared to standard softmax.
"""

import keras
from keras import ops
from typing import Optional, Any, Tuple, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------


def differentiable_step(
        x: keras.KerasTensor,
        slope: float = 1.0,
        shift: float = 0.0
) -> keras.KerasTensor:
    """Approximates a Heaviside step function using a scaled and shifted tanh.

    This function is fully differentiable and provides control over the
    steepness and location of the step transition. It smoothly transitions
    from 0 to 1, making it ideal for soft thresholding operations.

    The formula is: f(x) = (tanh(slope * (x - shift)) + 1) / 2

    Args:
        x: Input tensor of any shape.
        slope: Controls the steepness of the transition. Higher values create
            a sharper, more step-like function. Lower values create smoother
            transitions with better gradient flow.
        shift: The center point where the step occurs (output equals 0.5).

    Returns:
        Tensor with values smoothly transitioning from 0 to 1.
    """
    scaled_shifted_x = slope * (x - shift)
    return (ops.tanh(scaled_shifted_x) + 1.0) / 2.0


def _compute_threshmax(
        x: keras.KerasTensor,
        axis: int,
        epsilon: float,
        slope: float = 10.0
) -> keras.KerasTensor:
    """Internal computation for ThreshMax activation using differentiable step function.

    This function contains the core ThreshMax logic that is shared between
    the layer implementation and functional interface. Uses a smooth differentiable
    step function instead of hard clipping for better gradient flow.

    Args:
        x: Input tensor containing logits.
        axis: The axis along which to apply softmax normalization.
        epsilon: Small value for numerical stability and degenerate case detection.
        slope: Controls the steepness of the differentiable step. Higher values
            create sharper transitions (more like hard clipping), lower values
            create smoother transitions (better gradient flow).

    Returns:
        Output tensor with sparse probability distributions, falling back to
        standard softmax in maximum entropy cases.
    """
    # Step 1: Compute standard softmax
    y_soft = keras.activations.softmax(x, axis=axis)

    # Step 2: Compute confidence difference from uniform probability
    num_classes = ops.shape(x)[axis]
    uniform_prob = 1.0 / ops.cast(num_classes, x.dtype)
    confidence_diff = y_soft - uniform_prob

    # Step 3: Apply differentiable step function (replaces hard clipping)
    # This creates smooth sparsification based on confidence threshold
    y_stepped = differentiable_step(confidence_diff, slope=slope, shift=0.0)

    # Step 4: Detect the degenerate (maximum entropy) case
    # This occurs when all confidence differences are near zero
    total_sum = ops.sum(y_stepped, axis=axis, keepdims=True)
    is_degenerate = ops.less(total_sum, epsilon)

    # Step 5: Conditionally choose the output
    # If degenerate case: fall back to standard softmax
    # Otherwise: perform ThreshMax renormalization
    normal_output = y_stepped / (total_sum + epsilon)
    final_output = ops.where(
        is_degenerate,
        y_soft,  # Fallback: standard softmax (degenerate case)
        normal_output  # Normal: renormalized sparse output
    )

    return final_output


# ---------------------------------------------------------------------
# Keras layer implementation
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ThreshMax(keras.layers.Layer):
    """ThreshMax activation layer with differentiable step function.

    This layer implements a sparse softmax variant that creates sparsity by
    subtracting a uniform probability (1/N) from standard softmax probabilities,
    applying a smooth differentiable step function, and renormalizing the result.
    This technique helps create more confident and sparse probability distributions
    while maintaining smooth gradients throughout.

    The key innovation is using a differentiable step function instead of hard
    clipping, providing smooth gradients everywhere while still achieving the
    desired sparsity effect.

    Mathematical formulation:
        1. y_soft = softmax(x)
        2. confidence_diff = y_soft - 1/N  (where N is number of classes)
        3. y_stepped = differentiable_step(confidence_diff, slope)
        4. y_final = y_stepped / sum(y_stepped)  (with degenerate case handling)

    The differentiable step function is: (tanh(slope * x) + 1) / 2

    Key features:
        - Creates sparse probability distributions
        - Maintains smooth gradients via differentiable step function
        - Handles degenerate (maximum entropy) cases gracefully
        - Tunable sparsity through slope parameter
        - Falls back to standard softmax when appropriate

    Args:
        axis: Integer, the axis along which the softmax normalization is applied.
            Defaults to -1 (last axis).
        slope: Float, controls the steepness of the differentiable step function.
            Higher values create sharper transitions (more sparse, closer to hard
            clipping). Lower values create smoother transitions (better gradient
            flow, less sparse). Must be positive. Defaults to 10.0.
        epsilon: Float, small value for numerical stability to prevent division
            by zero and for detecting the degenerate case. Must be positive.
            Defaults to 1e-12.
        **kwargs: Additional keyword arguments passed to the Layer base class,
            such as `name`, `dtype`, `trainable`, etc.

    Input shape:
        Arbitrary tensor shape. The softmax normalization is applied along
        the specified axis.

    Output shape:
        Same shape as input. Contains sparse probability distributions that
        sum to 1.0 along the specified axis.

    Attributes:
        axis: The axis along which softmax normalization is applied.
        slope: The steepness parameter for the differentiable step function.
        epsilon: The numerical stability parameter.

    Example:
        ```python
        # Basic usage with default parameters
        layer = ThreshMax()
        inputs = keras.Input(shape=(10,))
        outputs = layer(inputs)

        # Custom parameters for different behaviors
        gentle_layer = ThreshMax(slope=2.0)    # Smoother, better gradients
        sharp_layer = ThreshMax(slope=50.0)    # Sharper, more sparse

        # In a classification model
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(784,)),
            keras.layers.Dense(10),  # Logits layer
            ThreshMax(slope=15.0)    # Sparse probability layer
        ])

        # Demonstrating different cases
        import numpy as np

        # Case 1: Clear winner -> sparse output
        logits1 = np.array([[1.0, 3.0, 0.5, -1.0]])
        layer = ThreshMax(slope=10.0)
        sparse_output = layer(logits1)
        # Results in sparse distribution favoring the maximum

        # Case 2: Uniform input -> fallback to softmax
        logits2 = np.array([[2.0, 2.0, 2.0, 2.0]])
        uniform_output = layer(logits2)
        # Results in [0.25, 0.25, 0.25, 0.25]

        # Case 3: Slope comparison
        gentle = ThreshMax(slope=2.0)
        sharp = ThreshMax(slope=20.0)
        logits3 = np.array([[0.2, 0.8, 0.3, 0.1]])
        gentle_result = gentle(logits3)  # Less sparse
        sharp_result = sharp(logits3)    # More sparse
        ```

    References:
        - Based on confidence thresholding techniques for sparse attention mechanisms
        - Differentiable step functions for smooth optimization in neural networks
        - Related to entropy regularization methods and sparse softmax variants

    Raises:
        ValueError: If epsilon is not positive or slope is not positive.

    Note:
        The slope parameter provides fine-grained control over the sparsity-gradient
        trade-off:
        - Low slope (1-5): Gentle regularization, smooth gradients, low sparsity
        - Medium slope (5-20): Balanced sparsity and gradient flow
        - High slope (20+): Sharp sparsity, approaching hard clipping behavior

        The layer gracefully handles the degenerate case when all logits are
        identical (maximum entropy) by falling back to standard softmax output.
    """

    def __init__(
            self,
            axis: int = -1,
            slope: float = 10.0,
            epsilon: float = 1e-12,
            **kwargs: Any
    ) -> None:
        """Initialize the ThreshMax layer.

        Args:
            axis: The axis along which to apply softmax normalization.
            slope: Controls the steepness of the differentiable step function.
                Must be positive.
            epsilon: Small value for numerical stability. Must be positive.
            **kwargs: Additional keyword arguments for the Layer base class.

        Raises:
            ValueError: If epsilon or slope is not positive.
        """
        super().__init__(**kwargs)

        # Validate parameters
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if slope <= 0:
            raise ValueError(f"slope must be positive, got {slope}")

        # Store configuration parameters
        self.axis = axis
        self.slope = float(slope)
        self.epsilon = float(epsilon)

        logger.info(f"Initialized ThreshMax with axis={axis}, slope={slope}, epsilon={epsilon}")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply ThreshMax activation to inputs.

        Applies the ThreshMax computation with robust degenerate case handling,
        using the differentiable step function for smooth gradient flow.

        Args:
            inputs: Input tensor containing logits.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Not used in this activation
                layer but kept for API consistency.

        Returns:
            Tensor with same shape as inputs containing sparse probability
            distributions that sum to 1.0 along the specified axis.
        """
        return _compute_threshmax(inputs, self.axis, self.epsilon, self.slope)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        For activation layers, the output shape is identical to the input shape
        since no dimensional transformation occurs.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple, identical to input_shape.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get the layer configuration for serialization.

        Returns all parameters passed to __init__ so the layer can be
        properly reconstructed during model loading.

        Returns:
            Dictionary containing the layer configuration, including
            axis, slope, and epsilon parameters along with parent class configuration.
        """
        config = super().get_config()
        config.update({
            'axis': self.axis,
            'slope': self.slope,
            'epsilon': self.epsilon,
        })
        return config

    def __repr__(self) -> str:
        """Return string representation of the layer.

        Returns:
            String representation including the layer name and key parameters.
        """
        return f"ThreshMax(axis={self.axis}, slope={self.slope}, epsilon={self.epsilon}, name='{self.name}')"


# ---------------------------------------------------------------------
# Functional interface and utilities
# ---------------------------------------------------------------------


def thresh_max(
        x: keras.KerasTensor,
        axis: int = -1,
        slope: float = 10.0,
        epsilon: float = 1e-12
) -> keras.KerasTensor:
    """Functional interface for ThreshMax activation.

    This function applies ThreshMax activation to the input tensor with robust
    handling of degenerate cases. It implements the computation using a shared
    internal function for optimal maintainability.

    Args:
        x: Input tensor containing logits.
        axis: The axis along which the softmax normalization is applied.
        slope: Controls the steepness of the differentiable step function.
        epsilon: Small value for numerical stability and degenerate case detection.

    Returns:
        Output tensor with sparse probability distributions, falling back to
        standard softmax in maximum entropy cases.

    Raises:
        ValueError: If epsilon or slope is not positive.

    Example:
        ```python
        import keras
        from keras import ops

        # Using functional interface
        logits = ops.convert_to_tensor([[1.0, 2.0, 0.5, -1.0]])
        sparse_probs = thresh_max(logits, slope=10.0)

        # Degenerate case (uniform logits)
        uniform_logits = ops.convert_to_tensor([[2.0, 2.0, 2.0, 2.0]])
        fallback_probs = thresh_max(uniform_logits)

        # Compare with standard softmax
        standard_probs = keras.activations.softmax(logits)
        print("Standard softmax:", standard_probs)
        print("ThreshMax sparse:", sparse_probs)
        print("ThreshMax fallback:", fallback_probs)  # Should equal standard softmax
        ```
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if slope <= 0:
        raise ValueError(f"slope must be positive, got {slope}")

    return _compute_threshmax(x, axis, epsilon, slope)

# ---------------------------------------------------------------------