import keras
from keras import ops
from typing import Optional, Any, Tuple

from dl_techniques.utils.logger import logger


def differentiable_step(
        x: keras.KerasTensor,
        slope: float = 1.0,
        shift: float = 0.0
) -> keras.KerasTensor:
    """
    Approximates a Heaviside step function using a scaled and shifted tanh.

    This function is fully differentiable and provides control over the
    steepness and location of the step transition. It smoothly transitions
    from 0 to 1, making it ideal for soft thresholding operations.

    The formula is: f(x) = (tanh(slope * (x - shift)) + 1) / 2

    Args:
        x: Input tensor.
        slope: Controls the steepness of the transition. Higher values create
            a sharper, more step-like function. Lower values create smoother
            transitions with better gradient flow.
        shift: The center point where the step occurs (output equals 0.5).

    Returns:
        Tensor with values smoothly transitioning from 0 to 1.

    Example:
        ```python
        import keras.ops as ops

        # Gentle step (good for gradient flow)
        x = ops.linspace(-2.0, 2.0, 100)
        gentle = differentiable_step(x, slope=2.0)

        # Sharp step (closer to hard clipping)
        sharp = differentiable_step(x, slope=20.0)

        # Shifted step (threshold at x=1.0)
        shifted = differentiable_step(x, slope=10.0, shift=1.0)
        ```
    """
    scaled_shifted_x = slope * (x - shift)
    return (ops.tanh(scaled_shifted_x) + 1.0) / 2.0


def _compute_threshmax(
        x: keras.KerasTensor,
        axis: int,
        epsilon: float,
        slope: float = 10.0
) -> keras.KerasTensor:
    """
    Internal computation for ThreshMax activation using differentiable step function.

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


@keras.saving.register_keras_serializable()
class ThreshMax(keras.layers.Layer):
    """
    ThreshMax activation layer with differentiable step function.

    This layer implements a sparse softmax variant that creates sparsity by
    subtracting a uniform probability (1/N) from standard softmax probabilities,
    applying a smooth differentiable step function, and renormalizing the result.
    This technique helps create more confident and sparse probability distributions
    while maintaining smooth gradients throughout.

    The computation follows these steps:
    1. Compute standard softmax: softmax(x)
    2. Subtract uniform probability: softmax(x) - 1/N
    3. Apply differentiable step: smooth_step(result, slope)
    4. Renormalize: result / sum(result)

    **Key Innovation - Differentiable Step Function:**
    Instead of hard clipping (max(0, x)), this layer uses a smooth differentiable
    step function based on tanh: (tanh(slope * x) + 1) / 2. This provides:
    - Smooth gradients everywhere (better optimization)
    - Tunable sparsity via the slope parameter
    - Principled mathematical foundation

    **Special Handling for Maximum Entropy (Degenerate) Case:**
    If the input logits are all identical (resulting in maximum entropy), the
    subtraction and step application would produce very low values. To handle
    this degenerate case gracefully, the layer detects this condition and
    returns the standard softmax output directly.

    Args:
        axis: Integer, the axis along which the softmax normalization is applied.
            Defaults to -1 (last axis).
        slope: Float, controls the steepness of the differentiable step function.
            Higher values create sharper transitions (more sparse, closer to hard
            clipping). Lower values create smoother transitions (better gradient
            flow, less sparse). Defaults to 10.0.
        epsilon: Float, small value for numerical stability to prevent division
            by zero and for detecting the degenerate case. Defaults to 1e-12.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Arbitrary tensor shape. The softmax normalization is applied along
        the specified axis.

    Output shape:
        Same shape as input. Contains sparse probability distributions that
        sum to 1.0.

    Example:
        ```python
        # Basic usage with default slope
        layer = ThreshMax()

        # Custom parameters for different behaviors
        gentle_layer = ThreshMax(slope=2.0)    # Smoother, better gradients
        sharp_layer = ThreshMax(slope=50.0)    # Sharper, more sparse

        # In a model
        inputs = keras.Input(shape=(10,))
        logits = keras.layers.Dense(4)(inputs)
        sparse_probs = ThreshMax(slope=15.0)(logits)
        model = keras.Model(inputs, sparse_probs)

        # Example showing normal and degenerate cases
        import numpy as np

        # Case 1: Clear winner -> sparse output (controlled by slope)
        logits1 = np.array([[1.0, 3.0, 0.5, -1.0]])
        output1 = layer(logits1)
        print("Sparse output:", output1.numpy())

        # Case 2: Uniform input (maximum entropy) -> fallback to softmax
        logits2 = np.array([[2.0, 2.0, 2.0, 2.0]])
        output2 = layer(logits2)
        print("Fallback output:", output2.numpy())
        # Expected: [0.25, 0.25, 0.25, 0.25]

        # Case 3: Slope comparison
        gentle = ThreshMax(slope=2.0)
        sharp = ThreshMax(slope=20.0)
        logits3 = np.array([[0.2, 0.4, 0.3, 0.1]])
        print("Gentle sparsity:", gentle(logits3).numpy())
        print("Sharp sparsity:", sharp(logits3).numpy())
        ```

    References:
        - Based on confidence thresholding techniques for sparse attention
        - Differentiable step functions for smooth optimization
        - Related to entropy regularization methods in neural networks

    Note:
        The slope parameter provides fine-grained control over the sparsity-gradient
        trade-off:
        - Low slope (1-5): Gentle regularization, smooth gradients, low sparsity
        - Medium slope (5-20): Balanced sparsity and gradient flow
        - High slope (20+): Sharp sparsity, approaching hard clipping behavior
    """

    def __init__(
            self,
            axis: int = -1,
            slope: float = 10.0,
            epsilon: float = 1e-12,
            **kwargs: Any
    ) -> None:
        """
        Initialize the ThreshMax layer.

        Args:
            axis: The axis along which to apply softmax normalization.
            slope: Controls the steepness of the differentiable step function.
                Higher values create sharper transitions (more sparse), lower
                values create smoother transitions (better gradient flow).
            epsilon: Small value for numerical stability.
            **kwargs: Additional keyword arguments for the Layer base class.

        Raises:
            ValueError: If epsilon is not positive or slope is not positive.
        """
        super().__init__(**kwargs)

        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if slope <= 0:
            raise ValueError(f"slope must be positive, got {slope}")

        self.axis = axis
        self.slope = slope
        self.epsilon = epsilon

        logger.info(f"Initialized ThreshMax with axis={axis}, slope={slope}, epsilon={epsilon}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer (no trainable parameters needed).

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        super().build(input_shape)
        logger.debug(f"Built ThreshMax for input shape: {input_shape}")

    def call(self, inputs: keras.KerasTensor, **_: Any) -> keras.KerasTensor:
        """
        Apply ThreshMax activation to inputs with robust degenerate case handling.

        Args:
            inputs: Input tensor containing logits.
            **_: Additional keyword arguments (intentionally ignored).

        Returns:
            Tensor with same shape as inputs containing sparse probability
            distributions that sum to 1.0 along the specified axis.
        """
        return _compute_threshmax(inputs, self.axis, self.epsilon)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple (same as input shape).
        """
        return input_shape

    def get_config(self) -> dict[str, Any]:
        """
        Return the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            'axis': self.axis,
            'epsilon': self.epsilon,
        })
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'ThreshMax':
        """
        Create layer from configuration dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            ThreshMax layer instance.
        """
        return cls(**config)


# Convenience function for functional API usage
def thresh_max(
        x: keras.KerasTensor,
        axis: int = -1,
        epsilon: float = 1e-12
) -> keras.KerasTensor:
    """
    Functional interface for ThreshMax activation.

    This function applies ThreshMax activation to the input tensor with robust
    handling of degenerate cases. It implements the computation using a shared
    internal function for optimal maintainability.

    Args:
        x: Input tensor containing logits.
        axis: The axis along which the softmax normalization is applied.
        epsilon: Small value for numerical stability and degenerate case detection.

    Returns:
        Output tensor with sparse probability distributions, falling back to
        standard softmax in maximum entropy cases.

    Raises:
        ValueError: If epsilon is not positive.

    Example:
        ```python
        import keras
        from keras import ops

        # Using functional interface
        logits = ops.convert_to_tensor([[1.0, 2.0, 0.5, -1.0]])
        sparse_probs = thresh_max(logits)

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

    return _compute_threshmax(x, axis, epsilon)


# Factory function for creating different variants
def create_thresh_max(
        axis: int = -1,
        epsilon: float = 1e-12,
        name: Optional[str] = None
) -> ThreshMax:
    """
    Factory function to create ThreshMax layer.

    This function provides a convenient way to create the layer with
    explicit parameters and optional naming.

    Args:
        axis: The axis along which to apply softmax normalization.
        epsilon: Small value for numerical stability.
        name: Optional name for the layer.

    Returns:
        Configured ThreshMax layer.

    Example:
        ```python
        # Create layer with custom settings
        sparse_layer = create_thresh_max(
            axis=1,
            epsilon=1e-10,
            name='confidence_threshold'
        )

        # Use in model
        model = keras.Sequential([
            keras.layers.Dense(10),
            sparse_layer
        ])
        ```
    """
    return ThreshMax(
        axis=axis,
        epsilon=epsilon,
        name=name
    )