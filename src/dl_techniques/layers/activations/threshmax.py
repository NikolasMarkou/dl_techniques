"""
ThreshMax: A Keras-based Sparse Softmax Activation Module

This module provides a robust and flexible implementation of `ThreshMax`, a sparse
variant of the standard softmax activation function. It is designed to work
seamlessly with Keras 3 and is backend-agnostic (TensorFlow, PyTorch, JAX).

The core idea of ThreshMax is to create sparsity in probability distributions by
enforcing a confidence threshold. This is useful in classification and attention
mechanisms where encouraging confident predictions and filtering out low-confidence
"noise" is desirable.

The algorithm follows these steps:
1.  **Softmax**: Compute the standard softmax of the input logits.
2.  **Thresholding**: Subtract a uniform probability (1 / number_of_classes)
    from each probability. This acts as a confidence threshold.
3.  **Clipping**: Set any resulting negative values to zero. Elements that do not
    surpass the confidence threshold are pruned.
4.  **Renormalization**: Renormalize the clipped vector so that its elements sum
    to 1.0, forming a new, sparse probability distribution.

A key feature of this implementation is its robust handling of the **degenerate
case**: if all input logits are identical (maximum entropy), the thresholding
step would result in an all-zero vector. To prevent division by zero and produce
a meaningful output, the implementation detects this condition and gracefully
falls back to returning the standard softmax distribution.

Key Features:
    - **Robustness**: Handles the maximum entropy edge case and ensures numerical
      stability using a small epsilon.
    - **Multiple Interfaces**: Provides a `ThreshMax` Keras Layer for easy
      integration into `Sequential` and object-oriented models, as well as a
      functional `thresh_max` for use in the Keras functional API and custom loops.
    - **Keras 3 Native**: Built with `keras.ops` for full backend-agnostic
      compatibility (TensorFlow, PyTorch, JAX).
    - **Serialization**: The `ThreshMax` layer is fully serializable and can be
      saved and loaded as part of any Keras model.

Provided Components:
    - `ThreshMax (keras.layers.Layer)`: The primary class-based implementation.
      Ideal for use in `keras.Sequential` models or when a layer object is needed.
    - `thresh_max (function)`: A functional interface for applying the activation
      directly to a tensor. Perfect for the Keras functional API.
    - `create_thresh_max (function)`: A convenience factory function for creating
      `ThreshMax` layer instances with explicit naming.

Usage Examples
--------------

**1. Using the `ThreshMax` Layer in a Sequential Model:**

.. code-block:: python

    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from this_module import ThreshMax # Assuming saved as this_module.py

    model = Sequential([
        Dense(64, activation='relu', input_shape=(128,)),
        Dense(10),          # Output logits
        ThreshMax(axis=-1)  # Apply sparse activation
    ])

    model.summary()

**2. Using the `thresh_max` Functional Interface:**

.. code-block:: python

    import keras
    from keras.models import Model
    from keras.layers import Input, Dense
    from this_module import thresh_max

    inputs = Input(shape=(128,))
    x = Dense(64, activation='relu')(inputs)
    logits = Dense(10)(x)
    sparse_probabilities = thresh_max(logits, axis=-1)

    model = Model(inputs=inputs, outputs=sparse_probabilities)
    model.summary()

**3. Demonstrating Normal vs. Degenerate Case Behavior:**

.. code-block:: python

    import numpy as np
    from this_module import ThreshMax

    layer = ThreshMax()

    # Case 1: Clear winner -> sparse output
    logits1 = np.array([[1.0, 4.0, 0.5, -1.0]])
    output1 = layer(logits1)
    # Expected: A sparse distribution, e.g., [0. , 0.9, 0. , 0. ] (approx)

    # Case 2: Uniform logits (degenerate case) -> fallback to softmax
    logits2 = np.array([[2.0, 2.0, 2.0, 2.0]])
    output2 = layer(logits2)
    # Expected: [0.25, 0.25, 0.25, 0.25]

"""

import keras
from keras import ops
from typing import Optional, Any, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

def _compute_threshmax(
    x: keras.KerasTensor,
    axis: int,
    epsilon: float
) -> keras.KerasTensor:
    """
    Internal computation for ThreshMax activation.

    This function contains the core ThreshMax logic that is shared between
    the layer implementation and functional interface.

    Args:
        x: Input tensor containing logits.
        axis: The axis along which to apply softmax normalization.
        epsilon: Small value for numerical stability and degenerate case detection.

    Returns:
        Output tensor with sparse probability distributions, falling back to
        standard softmax in maximum entropy cases.
    """
    # Step 1: Compute standard softmax
    y_soft = keras.activations.softmax(x, axis=axis)

    # Step 2: Perform the core ThreshMax logic
    num_classes = ops.shape(x)[axis]
    uniform_prob = 1.0 / ops.cast(num_classes, x.dtype)
    y_shifted = y_soft - uniform_prob
    y_clipped = ops.maximum(0.0, y_shifted)

    # Step 3: Detect the degenerate (maximum entropy) case
    # This occurs when all values are clipped to zero
    total_sum = ops.sum(y_clipped, axis=axis, keepdims=True)
    is_degenerate = ops.less(total_sum, epsilon)

    # Step 4: Conditionally choose the output
    # If degenerate case: fall back to standard softmax
    # Otherwise: perform ThreshMax renormalization
    normal_output = y_clipped / (total_sum + epsilon)
    final_output = ops.where(
        is_degenerate,
        y_soft,          # Fallback: standard softmax (degenerate case)
        normal_output    # Normal: renormalized sparse output
    )

    return final_output

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ThreshMax(keras.layers.Layer):
    """
    ThreshMax activation layer.

    This layer implements a sparse softmax variant that creates sparsity by
    subtracting a uniform probability (1/N) from standard softmax probabilities,
    clipping negative values to zero, and renormalizing the result. This technique
    helps create more confident and sparse probability distributions.

    The computation follows these steps:
    1. Compute standard softmax: softmax(x)
    2. Subtract uniform probability: softmax(x) - 1/N
    3. Clip negative values: max(0, result)
    4. Renormalize: result / sum(result)

    **Special Handling for Maximum Entropy (Degenerate) Case:**
    If the input logits are all identical (resulting in maximum entropy), the
    subtraction and clipping steps would produce an all-zero vector. To handle
    this degenerate case gracefully, the layer detects this condition and
    returns the standard softmax output directly, ensuring a valid probability
    distribution.

    Args:
        axis: Integer, the axis along which the softmax normalization is applied.
            Defaults to -1 (last axis).
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
        # Basic usage
        layer = ThreshMax()

        # Custom axis and epsilon
        layer = ThreshMax(axis=1, epsilon=1e-10)

        # In a model
        inputs = keras.Input(shape=(10,))
        logits = keras.layers.Dense(4)(inputs)
        sparse_probs = ThreshMax()(logits)
        model = keras.Model(inputs, sparse_probs)

        # Example showing normal and degenerate cases
        import numpy as np

        # Case 1: Clear winner -> sparse output
        logits1 = np.array([[1.0, 3.0, 0.5, -1.0]])
        output1 = layer(logits1)
        print("Sparse output:", output1.numpy())
        # Expected: Sparse distribution with clear winner

        # Case 2: Uniform input (maximum entropy) -> fallback to softmax
        logits2 = np.array([[2.0, 2.0, 2.0, 2.0]])
        output2 = layer(logits2)
        print("Fallback output:", output2.numpy())
        # Expected: [0.25, 0.25, 0.25, 0.25]

        # Case 3: Low confidence -> sparse filtering
        logits3 = np.array([[0.1, 0.2, 0.1, 0.0]])
        output3 = layer(logits3)
        print("Filtered output:", output3.numpy())
        ```

    References:
        - Based on confidence thresholding techniques for sparse attention
        - Related to entropy regularization methods in neural networks

    Note:
        This activation is particularly useful for classification tasks where
        you want to encourage confident predictions and suppress uncertain ones.
        The uniform subtraction acts as a confidence threshold below which
        predictions are considered unreliable. The layer gracefully handles
        maximum entropy cases by falling back to standard softmax behavior.
    """

    def __init__(
        self,
        axis: int = -1,
        epsilon: float = 1e-12,
        **kwargs: Any
    ) -> None:
        """
        Initialize the ThreshMax layer.

        Args:
            axis: The axis along which to apply softmax normalization.
            epsilon: Small value for numerical stability.
            **kwargs: Additional keyword arguments for the Layer base class.

        Raises:
            ValueError: If epsilon is not positive.
        """
        super().__init__(**kwargs)

        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.axis = axis
        self.epsilon = epsilon

        logger.info(f"Initialized ThreshMax with axis={axis}, epsilon={epsilon}")

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

# ---------------------------------------------------------------------
# Convenience function for functional API usage
# ---------------------------------------------------------------------

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

# ---------------------------------------------------------------------
# Factory function for creating different variants
# ---------------------------------------------------------------------

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

# ---------------------------------------------------------------------
