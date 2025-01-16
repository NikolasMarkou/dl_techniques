"""
Blocks and builders for custom regularizers.

This module implements various regularization techniques for neural networks,
including custom soft orthogonal and orthonormal constraints based on the paper:
"Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?"
https://arxiv.org/abs/1810.09102

Key features:
- Soft orthogonality constraint regularization
- Soft orthonormality constraint regularization
- Support for both 2D and 4D tensors
- Configurable L1, L2 and lambda coefficients
- TensorFlow 2.x and Keras compatibility
"""
import keras
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, Union, List, Any, Optional

# Default hyperparameters for regularization
EPSILON = 1e-12
DEFAULT_SOFTORTHOGONAL_L1: float = 0.0
DEFAULT_SOFTORTHOGONAL_L2: float = 0.001
DEFAULT_SOFTORTHOGONAL_LAMBDA: float = 0.01
DEFAULT_SOFTORTHOGONAL_STDDEV: float = 0.02

# ---------------------------------------------------------------------

# @tf.function
# def reshape_to_2d(weights: tf.Tensor) -> tf.Tensor:
#     """Reshape 2D or 4D tensor to 2D matrix for regularization computations."""
#     weights_shape = tf.shape(weights)
#
#     if len(weights.shape) == 2:
#         return tf.transpose(weights)
#     elif len(weights.shape) == 4:
#         # More stable reshaping with explicit size computation
#         F = weights_shape[3]
#         spatial_dims = weights_shape[0] * weights_shape[1] * weights_shape[2]
#         w_t = tf.transpose(weights, perm=[3, 0, 1, 2])
#         return tf.reshape(w_t, [F, spatial_dims])
#
#     # Handle unexpected input shapes gracefully
#     tf.debugging.assert_rank_in(weights, [2, 4],
#                                 message="Input tensor must be 2D or 4D")
#     return weights

@tf.function
def reshape_to_2d(weights: tf.Tensor) -> tf.Tensor:
    """Reshape weight tensor to 2D matrix for regularization computations.

    Handles standard neural network weight tensor formats:
    - Dense: (in_features, out_features)
    - Conv2D: (h, w, in_c, out_c)
    - Conv3D: (d, h, w, in_c, out_c)
    - Conv1D: (w, in_c, out_c)

    Args:
        weights: Input weight tensor

    Returns:
        2D tensor where first dimension is output features/channels
    """
    ndims = len(weights.shape)

    # Assert supported number of dimensions
    tf.debugging.assert_equal(
        tf.reduce_any(tf.equal(ndims, [2, 3, 4, 5])),
        True,
        message=(
            "Tensor rank must be one of:\n"
            "2 (Dense: in_features, out_features)\n"
            "3 (Conv1D: width, in_channels, out_channels)\n"
            "4 (Conv2D: height, width, in_channels, out_channels)\n"
            "5 (Conv3D: depth, height, width, in_channels, out_channels)"
        )
    )

    # For any conv layer (1D/2D/3D), last dimension is always out_channels
    # Everything else gets flattened into the second dimension
    out_channels = tf.shape(weights)[-1]

    # Move out_channels to first dimension
    perm = tf.concat([
        [ndims - 1],  # Last dim (out_channels) goes first
        tf.range(ndims - 1)  # Other dims maintain relative order
    ], axis=0)
    w_t = tf.transpose(weights, perm)

    # Flatten rest into single dimension
    return tf.reshape(w_t, [out_channels, -1])


@tf.function
def gram_matrix(weights: tf.Tensor) -> tf.Tensor:
    """Compute W^T * W with improved numerical stability."""
    wt = reshape_to_2d(weights)
    return tf.matmul(wt, tf.transpose(wt))


@tf.function
def wt_x_w_normalize(weights: tf.Tensor) -> tf.Tensor:
    """Compute W^T * W with improved numerical stability."""
    wt = reshape_to_2d(weights)

    # Normalize the weights before multiplication for better conditioning
    norm = tf.maximum(tf.norm(wt, axis=1, keepdims=True), EPSILON)
    wt_normalized = wt / norm

    return tf.matmul(wt_normalized, tf.transpose(wt_normalized))


@tf.keras.utils.register_keras_serializable()
class SoftOrthogonalConstraintRegularizer(tf.keras.regularizers.Regularizer):
    """Implements soft orthogonality constraint regularization.

    This regularizer penalizes deviations from orthogonality in weight matrices
    by minimizing ||W^T * W - I||_F where I is zero for off-diagonal elements.
    """

    def __init__(
            self,
            lambda_coefficient: float = DEFAULT_SOFTORTHOGONAL_LAMBDA,
            l1_coefficient: float = DEFAULT_SOFTORTHOGONAL_L1,
            l2_coefficient: float = DEFAULT_SOFTORTHOGONAL_L2,
            **kwargs: Any
    ) -> None:
        """Initialize regularizer.

        Args:
            lambda_coefficient: Weight for Frobenius norm term
            l1_coefficient: Weight for L1 regularization
            l2_coefficient: Weight for L2 regularization
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)
        self._lambda_coefficient = lambda_coefficient
        self._l1_coefficient = l1_coefficient
        self._l2_coefficient = l2_coefficient

        # Cache function and flags for performance
        self._call_fn: Optional[tf.types.experimental.ConcreteFunction] = None
        self._use_lambda = self._lambda_coefficient > 0.0
        self._use_l1 = self._l1_coefficient > 0.0
        self._use_l2 = self._l2_coefficient > 0.0

        # Create L1/L2 regularizers once
        if self._use_l1:
            self._l1 = tf.keras.regularizers.L1(l1=self._l1_coefficient)
        if self._use_l2:
            self._l2 = tf.keras.regularizers.L2(l2=self._l2_coefficient)

    def generic_fn(self, x: tf.Tensor) -> tf.Tensor:
        """Compute regularization for given weights.

        Args:
            x: Weight tensor to regularize

        Returns:
            Scalar regularization loss value
        """
        # Compute W^T * W
        wt_w = gram_matrix(x)

        # Mask diagonal elements
        eye = tf.eye(tf.shape(wt_w)[0])
        wt_w_masked = tf.math.multiply(wt_w, 1.0 - eye)

        # Initialize result
        result = tf.constant(0.0, dtype=tf.float32)

        # Add Frobenius norm term if enabled
        if self._use_lambda:
            result += self._lambda_coefficient * tf.square(
                tf.norm(wt_w_masked, ord="fro", axis=(0, 1))
            )

        # Add L1 regularization if enabled
        if self._use_l1:
            result += self._l1(wt_w_masked)

        # Add L2 regularization if enabled
        if self._use_l2:
            result += self._l2(wt_w_masked)

        return result

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """Apply regularization to weights.

        Args:
            x: Weight tensor

        Returns:
            Regularization loss value
        """
        if self._call_fn is None:
            self._call_fn = tf.function(
                func=self.generic_fn,
                reduce_retracing=True
            ).get_concrete_function(x)
        return self._call_fn(x)

    def get_config(self) -> Dict[str, float]:
        """Get configuration for serialization.

        Returns:
            Dictionary containing configuration parameters
        """
        return {
            "lambda_coefficient": self._lambda_coefficient,
            "l1_coefficient": self._l1_coefficient,
            "l2_coefficient": self._l2_coefficient
        }


@tf.keras.utils.register_keras_serializable()
class SoftOrthonormalConstraintRegularizer(tf.keras.regularizers.Regularizer):
    """Implements soft orthonormality constraint regularization.

    This regularizer penalizes deviations from orthonormality in weight matrices
    by minimizing ||W^T * W - I||_F where I is the identity matrix.
    """

    def __init__(
            self,
            lambda_coefficient: float = DEFAULT_SOFTORTHOGONAL_LAMBDA,
            l1_coefficient: float = DEFAULT_SOFTORTHOGONAL_L1,
            l2_coefficient: float = DEFAULT_SOFTORTHOGONAL_L2,
            **kwargs: Any
    ) -> None:
        """Initialize regularizer.

        Args:
            lambda_coefficient: Weight for Frobenius norm term
            l1_coefficient: Weight for L1 regularization
            l2_coefficient: Weight for L2 regularization
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)
        self._lambda_coefficient = lambda_coefficient
        self._l1_coefficient = l1_coefficient
        self._l2_coefficient = l2_coefficient

        # Cache function and flags for performance
        self._call_fn: Optional[tf.types.experimental.ConcreteFunction] = None
        self._use_lambda = self._lambda_coefficient > 0.0
        self._use_l1 = self._l1_coefficient > 0.0
        self._use_l2 = self._l2_coefficient > 0.0

        # Create L1/L2 regularizers once
        if self._use_l1:
            self._l1 = tf.keras.regularizers.L1(l1=self._l1_coefficient)
        if self._use_l2:
            self._l2 = tf.keras.regularizers.L2(l2=self._l2_coefficient)

    def generic_fn(self, x: tf.Tensor) -> tf.Tensor:
        """Compute regularization for given weights.

        Args:
            x: Weight tensor to regularize

        Returns:
            Scalar regularization loss value
        """
        # Compute W^T * W
        wt_w = gram_matrix(x)
        eye = tf.eye(tf.shape(wt_w)[0])

        # Initialize result
        result = tf.constant(0.0, dtype=tf.float32)

        # Add Frobenius norm term if enabled
        if self._use_lambda:
            result += self._lambda_coefficient * tf.square(
                tf.norm(wt_w - eye, ord="fro", axis=(0, 1))
            )

        # Add L1 regularization if enabled
        if self._use_l1:
            result += self._l1(wt_w)

        # Add L2 regularization if enabled
        if self._use_l2:
            result += self._l2(wt_w)

        return result

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        """Apply regularization to weights.

        Args:
            x: Weight tensor

        Returns:
            Regularization loss value
        """
        if self._call_fn is None:
            self._call_fn = tf.function(
                func=self.generic_fn,
                reduce_retracing=True
            ).get_concrete_function(x)
        return self._call_fn(x)

    def get_config(self) -> Dict[str, float]:
        """Get configuration for serialization.

        Returns:
            Dictionary containing configuration parameters
        """
        return {
            "lambda_coefficient": self._lambda_coefficient,
            "l1_coefficient": self._l1_coefficient,
            "l2_coefficient": self._l2_coefficient
        }
