"""
Theory and Implementation of Soft Orthogonality and Orthonormality Constraints
------------------------------------------------------------------------------

Background
----------
Neural networks often suffer from issues like vanishing/exploding gradients and covariate shift.
Orthogonal weight matrices have been shown to mitigate these issues by preserving gradient
magnitudes during backpropagation. The paper "Can We Gain More from Orthogonality Regularizations
in Training Deep CNNs?" (2018) demonstrates that soft orthogonality constraints can significantly
improve model performance.

Mathematical Foundation
----------------------
For a weight matrix W:
1. Orthogonality: W^T * W = I (identity matrix) for square matrices, or columns are orthogonal
   to each other for rectangular matrices
2. Orthonormality: W^T * W = I and each column of W has unit norm (L2 norm = 1)

Enforcing these properties exactly is challenging during optimization. Instead, we use "soft"
constraints implemented as regularization terms in the loss function.

Regularization Formulations
---------------------------
1. Soft Orthogonal Constraint:
   - Encourages off-diagonal elements of W^T*W to be zero
   - Regularization term: ||W^T*W - D||_F^2, where D is a diagonal matrix
     (implemented as masking off-diagonal elements)
   - Does not constrain the magnitudes of the weights

2. Soft Orthonormal Constraint:
   - Encourages W^T*W to approximate the identity matrix
   - Regularization term: ||W^T*W - I||_F^2
   - Encourages both orthogonality AND unit magnitude

Implementation Details
---------------------
Both regularizers compute the Gram matrix (W^T*W) and then penalize its deviation from
the target structure:

1. SoftOrthogonalConstraintRegularizer:
   - Calculates Gram matrix via gram_matrix(x)
   - Masks diagonal elements using (1.0 - eye)
   - Computes squared Frobenius norm of masked matrix
   - Optionally scales by matrix size for consistent effect across different layer sizes

2. SoftOrthonormalConstraintRegularizer:
   - Calculates Gram matrix via gram_matrix(x)
   - Computes squared Frobenius norm of (W^T*W - I)
   - Optionally scales by matrix size for consistent effect

Additional Features
------------------
1. Hybrid regularization:
   - Both regularizers support optional L1/L2 regularization terms
   - Lambda coefficient controls strength of orthogonality constraint

2. Numerical stability:
   - gram_matrix reshapes tensors to 2D for consistent processing
   - Epsilon values prevent division by zero
   - Matrix scaling improves consistency across different layer sizes

3. Performance optimizations:
   - Caches flags for enabled regularization terms (_use_lambda, _use_l1, _use_l2)
   - Creates L1/L2 regularizers only once during initialization

Practical Benefits
-----------------
1. Improved gradient flow during training
2. Better conditioning of the optimization landscape
3. Enhanced generalization performance
4. Reduced sensitivity to initialization
5. Faster convergence in deep networks

Usage Guidelines
---------------
- For very deep networks, use SoftOrthogonalConstraintRegularizer
- For convolutional layers, often SoftOrthonormalConstraintRegularizer works better
- lambda_coefficient should typically be in the range [1e-4, 1e-2]
- Matrix scaling is recommended for networks with varying layer sizes
"""
import keras
import tensorflow as tf
from typing import Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------
from dl_techniques.utils.tensors import gram_matrix

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

EPSILON = 1e-12
DEFAULT_SOFTORTHOGONAL_L1: float = 0.0
DEFAULT_SOFTORTHOGONAL_L2: float = 1e-4
DEFAULT_SOFTORTHOGONAL_LAMBDA: float = 1e-3
DEFAULT_SOFTORTHOGONAL_STDDEV: float = 0.02

# String constants
STR_FRO = "fro"
STR_L1_COEFFICIENT = "l1_coefficient"
STR_L2_COEFFICIENT = "l2_coefficient"
STR_LAMBDA_COEFFICIENT = "lambda_coefficient"
STR_USE_MATRIX_SCALING = "use_matrix_scaling"


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
            use_matrix_scaling: bool = False,
            **kwargs: Any
    ) -> None:
        """Initialize regularizer.

        Args:
            lambda_coefficient: Weight for Frobenius norm term
            l1_coefficient: Weight for L1 regularization
            l2_coefficient: Weight for L2 regularization
            use_matrix_scaling: Whether to scale regularization by matrix size
                                for consistent effect across different sized layers
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)
        self._lambda_coefficient = lambda_coefficient
        self._l1_coefficient = l1_coefficient
        self._l2_coefficient = l2_coefficient
        self._use_matrix_scaling = use_matrix_scaling

        # Cache function and flags for performance
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

        # Get shape information for scaling
        rank = tf.shape(wt_w)[0]

        # Mask diagonal elements
        eye = tf.eye(rank)
        wt_w_masked = tf.math.multiply(wt_w, 1.0 - eye)

        # Initialize result
        result = tf.constant(0.0, dtype=tf.float32)

        # Add Frobenius norm term if enabled
        if self._use_lambda:
            # Calculate Frobenius norm
            frob_norm = tf.square(tf.norm(wt_w_masked, ord=STR_FRO, axis=(0, 1)))

            # Apply matrix scaling if enabled
            if self._use_matrix_scaling:
                # For orthogonal regularizer (off-diagonal elements)
                # Total number of off-diagonal elements is rank^2 - rank
                off_diag_elements = tf.cast(rank * rank - rank, dtype=tf.float32)

                # Add epsilon to prevent division by zero
                scaling_factor = tf.maximum(off_diag_elements, EPSILON)

                # Scale by number of off-diagonal elements
                result += (self._lambda_coefficient * frob_norm) / scaling_factor
            else:
                # Original behavior (no scaling)
                result += self._lambda_coefficient * frob_norm

        # Add L1 regularization if enabled
        if self._use_l1:
            result += self._l1(wt_w_masked)

        # Add L2 regularization if enabled
        if self._use_l2:
            result += self._l2(wt_w_masked)

        return result

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization.

        Returns:
            Dictionary containing configuration parameters
        """
        return {
            STR_L1_COEFFICIENT: self._l1_coefficient,
            STR_L2_COEFFICIENT: self._l2_coefficient,
            STR_LAMBDA_COEFFICIENT: self._lambda_coefficient,
            STR_USE_MATRIX_SCALING: self._use_matrix_scaling
        }


# ---------------------------------------------------------------------


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
            use_matrix_scaling: bool = False,
            **kwargs: Any
    ) -> None:
        """Initialize regularizer.

        Args:
            lambda_coefficient: Weight for Frobenius norm term
            l1_coefficient: Weight for L1 regularization
            l2_coefficient: Weight for L2 regularization
            use_matrix_scaling: Whether to scale regularization by matrix size
                                for consistent effect across different sized layers
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(**kwargs)
        self._lambda_coefficient = lambda_coefficient
        self._l1_coefficient = l1_coefficient
        self._l2_coefficient = l2_coefficient
        self._use_matrix_scaling = use_matrix_scaling

        # Cache function and flags for performance
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

        # Get shape information for scaling
        rank = tf.shape(wt_w)[0]
        eye = tf.eye(rank)

        # Initialize result
        result = tf.constant(0.0, dtype=tf.float32)

        # Add Frobenius norm term if enabled
        if self._use_lambda:
            # Calculate Frobenius norm
            frob_norm = tf.square(tf.norm(wt_w - eye, ord=STR_FRO, axis=(0, 1)))

            # Apply matrix scaling if enabled
            if self._use_matrix_scaling:
                # For orthonormal regularizer (full matrix minus diagonal)
                # Total number of elements is rank^2 - rank
                total_elements = tf.cast(rank * rank - rank, dtype=tf.float32)

                # Add epsilon to prevent division by zero
                scaling_factor = tf.maximum(total_elements, EPSILON)

                # Scale by total number of elements
                result += (self._lambda_coefficient * frob_norm) / scaling_factor
            else:
                # Original behavior (no scaling)
                result += self._lambda_coefficient * frob_norm

        # Add L1 regularization if enabled
        if self._use_l1:
            result += self._l1(wt_w)

        # Add L2 regularization if enabled
        if self._use_l2:
            result += self._l2(wt_w)

        return result

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization.

        Returns:
            Dictionary containing configuration parameters
        """
        return {
            STR_L1_COEFFICIENT: self._l1_coefficient,
            STR_L2_COEFFICIENT: self._l2_coefficient,
            STR_LAMBDA_COEFFICIENT: self._lambda_coefficient,
            STR_USE_MATRIX_SCALING: self._use_matrix_scaling
        }

# ---------------------------------------------------------------------
