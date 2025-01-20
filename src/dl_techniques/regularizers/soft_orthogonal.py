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
# local imports
# ---------------------------------------------------------------------


from dl_techniques.utils.tensors import reshape_to_2d, gram_matrix, wt_x_w_normalize


# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class SoftOrthogonalConstraintRegularizer(keras.regularizers.Regularizer):
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
        super().__init__()
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
            self._l1 = keras.regularizers.L1(l1=self._l1_coefficient)
        if self._use_l2:
            self._l2 = keras.regularizers.L2(l2=self._l2_coefficient)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
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


@keras.utils.register_keras_serializable()
class SoftOrthonormalConstraintRegularizer(keras.regularizers.Regularizer):
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
        super().__init__()
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
            self._l1 = keras.regularizers.L1(l1=self._l1_coefficient)
        if self._use_l2:
            self._l2 = keras.regularizers.L2(l2=self._l2_coefficient)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
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
