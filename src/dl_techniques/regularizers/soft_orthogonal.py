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
from keras import ops
from typing import Dict, Any, Optional, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.tensors import gram_matrix

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

EPSILON: float = 1e-12
DEFAULT_SOFTORTHOGONAL_L1: float = 0.0
DEFAULT_SOFTORTHOGONAL_L2: float = 1e-4
DEFAULT_SOFTORTHOGONAL_LAMBDA: float = 1e-3
DEFAULT_SOFTORTHOGONAL_STDDEV: float = 0.02

# String constants
STR_FRO: str = "fro"
STR_L1_COEFFICIENT: str = "l1_coefficient"
STR_L2_COEFFICIENT: str = "l2_coefficient"
STR_LAMBDA_COEFFICIENT: str = "lambda_coefficient"
STR_USE_MATRIX_SCALING: str = "use_matrix_scaling"


# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class SoftOrthogonalConstraintRegularizer(keras.regularizers.Regularizer):
    """Implements soft orthogonality constraint regularization.

    This regularizer penalizes deviations from orthogonality in weight matrices
    by minimizing ||W^T * W - I||_F where I is zero for off-diagonal elements.

    Parameters
    ----------
    lambda_coefficient : float, optional
        Weight for Frobenius norm term, by default DEFAULT_SOFTORTHOGONAL_LAMBDA
    l1_coefficient : float, optional
        Weight for L1 regularization, by default DEFAULT_SOFTORTHOGONAL_L1
    l2_coefficient : float, optional
        Weight for L2 regularization, by default DEFAULT_SOFTORTHOGONAL_L2
    use_matrix_scaling : bool, optional
        Whether to scale regularization by matrix size for consistent effect
        across different sized layers, by default False
    **kwargs : Any
        Additional arguments passed to parent regularizer

    Notes
    -----
    The regularizer computes the Gram matrix (W^T * W) and penalizes deviations
    from orthogonality by minimizing the off-diagonal elements.

    Examples
    --------
    >>> regularizer = SoftOrthogonalConstraintRegularizer(lambda_coefficient=1e-3)
    >>> dense_layer = keras.layers.Dense(64, kernel_regularizer=regularizer)
    """

    def __init__(
            self,
            lambda_coefficient: float = DEFAULT_SOFTORTHOGONAL_LAMBDA,
            l1_coefficient: float = DEFAULT_SOFTORTHOGONAL_L1,
            l2_coefficient: float = DEFAULT_SOFTORTHOGONAL_L2,
            use_matrix_scaling: bool = False,
            **kwargs: Any
    ) -> None:
        """Initialize the soft orthogonal constraint regularizer.

        Parameters
        ----------
        lambda_coefficient : float, optional
            Weight for Frobenius norm term, by default DEFAULT_SOFTORTHOGONAL_LAMBDA
        l1_coefficient : float, optional
            Weight for L1 regularization, by default DEFAULT_SOFTORTHOGONAL_L1
        l2_coefficient : float, optional
            Weight for L2 regularization, by default DEFAULT_SOFTORTHOGONAL_L2
        use_matrix_scaling : bool, optional
            Whether to scale regularization by matrix size, by default False
        **kwargs : Any
            Additional arguments passed to parent regularizer

        Raises
        ------
        ValueError
            If any coefficient is negative
        """
        super().__init__(**kwargs)

        # Validate input parameters
        if lambda_coefficient < 0.0:
            raise ValueError(f"lambda_coefficient must be non-negative, got {lambda_coefficient}")
        if l1_coefficient < 0.0:
            raise ValueError(f"l1_coefficient must be non-negative, got {l1_coefficient}")
        if l2_coefficient < 0.0:
            raise ValueError(f"l2_coefficient must be non-negative, got {l2_coefficient}")

        self._lambda_coefficient = lambda_coefficient
        self._l1_coefficient = l1_coefficient
        self._l2_coefficient = l2_coefficient
        self._use_matrix_scaling = use_matrix_scaling

        # Cache flags for performance optimization
        self._use_lambda = self._lambda_coefficient > 0.0
        self._use_l1 = self._l1_coefficient > 0.0
        self._use_l2 = self._l2_coefficient > 0.0

        # Initialize L1/L2 regularizers once for efficiency
        self._l1: Optional[keras.regularizers.L1] = None
        self._l2: Optional[keras.regularizers.L2] = None

        if self._use_l1:
            self._l1 = keras.regularizers.L1(l1=self._l1_coefficient)
        if self._use_l2:
            self._l2 = keras.regularizers.L2(l2=self._l2_coefficient)

        logger.debug(
            f"Initialized SoftOrthogonalConstraintRegularizer with "
            f"lambda={lambda_coefficient}, l1={l1_coefficient}, "
            f"l2={l2_coefficient}, scaling={use_matrix_scaling}"
        )

    def __call__(self, x: Union[keras.KerasTensor, Any], **kwargs) -> Union[keras.KerasTensor, Any]:
        """Compute regularization loss for given weights.

        Parameters
        ----------
        x : Union[keras.KerasTensor, Any]
            Weight tensor to regularize
        **kwargs : Any
            Additional keyword arguments (e.g., dtype) that Keras may pass

        Returns
        -------
        Union[keras.KerasTensor, Any]
            Scalar regularization loss value

        Notes
        -----
        The regularization loss is computed as:
        - Frobenius norm of off-diagonal elements of W^T * W
        - Optional L1/L2 regularization terms
        - Optional matrix scaling for size-invariant regularization
        """
        # Compute Gram matrix W^T * W
        wt_w = gram_matrix(x)

        # Get matrix rank for scaling calculations
        rank = ops.shape(wt_w)[0]

        # Create identity matrix and mask for off-diagonal elements
        eye = ops.eye(rank, dtype=wt_w.dtype)
        off_diagonal_mask = ops.subtract(ops.cast(1.0, dtype=wt_w.dtype), eye)
        wt_w_masked = ops.multiply(wt_w, off_diagonal_mask)

        # Initialize regularization loss
        result = ops.cast(0.0, dtype=x.dtype)

        # Add Frobenius norm term if enabled
        if self._use_lambda:
            # Calculate squared Frobenius norm of off-diagonal elements
            frob_norm = ops.square(ops.norm(wt_w_masked, ord=STR_FRO, axis=(0, 1)))

            if self._use_matrix_scaling:
                # Scale by number of off-diagonal elements for size invariance
                off_diag_elements = ops.cast(ops.subtract(ops.multiply(rank, rank), rank), dtype=x.dtype)
                scaling_factor = ops.maximum(off_diag_elements, ops.cast(EPSILON, dtype=x.dtype))
                scaled_loss = ops.divide(ops.multiply(self._lambda_coefficient, frob_norm), scaling_factor)
                result = ops.add(result, scaled_loss)
            else:
                # Original behavior without scaling
                lambda_loss = ops.multiply(self._lambda_coefficient, frob_norm)
                result = ops.add(result, lambda_loss)

        # Add L1 regularization if enabled
        if self._use_l1 and self._l1 is not None:
            l1_loss = self._l1(x)
            result = ops.add(result, l1_loss)

        # Add L2 regularization if enabled
        if self._use_l2 and self._l2 is not None:
            l2_loss = self._l2(x)
            result = ops.add(result, l2_loss)

        return result

    def get_config(self) -> Dict[str, Any]:
        """Get regularizer configuration for serialization.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing configuration parameters

        Notes
        -----
        This method is required for proper serialization and deserialization
        of models containing this regularizer.
        """
        config = {}
        config.update({
            STR_L1_COEFFICIENT: self._l1_coefficient,
            STR_L2_COEFFICIENT: self._l2_coefficient,
            STR_LAMBDA_COEFFICIENT: self._lambda_coefficient,
            STR_USE_MATRIX_SCALING: self._use_matrix_scaling
        })
        return config


# ---------------------------------------------------------------------


@keras.utils.register_keras_serializable()
class SoftOrthonormalConstraintRegularizer(keras.regularizers.Regularizer):
    """Implements soft orthonormality constraint regularization.

    This regularizer penalizes deviations from orthonormality in weight matrices
    by minimizing ||W^T * W - I||_F where I is the identity matrix.

    Parameters
    ----------
    lambda_coefficient : float, optional
        Weight for Frobenius norm term, by default DEFAULT_SOFTORTHOGONAL_LAMBDA
    l1_coefficient : float, optional
        Weight for L1 regularization, by default DEFAULT_SOFTORTHOGONAL_L1
    l2_coefficient : float, optional
        Weight for L2 regularization, by default DEFAULT_SOFTORTHOGONAL_L2
    use_matrix_scaling : bool, optional
        Whether to scale regularization by matrix size for consistent effect
        across different sized layers, by default False
    **kwargs : Any
        Additional arguments passed to parent regularizer

    Notes
    -----
    The regularizer computes the Gram matrix (W^T * W) and penalizes deviations
    from the identity matrix, encouraging both orthogonality and unit norm.

    Examples
    --------
    >>> regularizer = SoftOrthonormalConstraintRegularizer(lambda_coefficient=1e-3)
    >>> conv_layer = keras.layers.Conv2D(32, 3, kernel_regularizer=regularizer)
    """

    def __init__(
            self,
            lambda_coefficient: float = DEFAULT_SOFTORTHOGONAL_LAMBDA,
            l1_coefficient: float = DEFAULT_SOFTORTHOGONAL_L1,
            l2_coefficient: float = DEFAULT_SOFTORTHOGONAL_L2,
            use_matrix_scaling: bool = False,
            **kwargs: Any
    ) -> None:
        """Initialize the soft orthonormal constraint regularizer.

        Parameters
        ----------
        lambda_coefficient : float, optional
            Weight for Frobenius norm term, by default DEFAULT_SOFTORTHOGONAL_LAMBDA
        l1_coefficient : float, optional
            Weight for L1 regularization, by default DEFAULT_SOFTORTHOGONAL_L1
        l2_coefficient : float, optional
            Weight for L2 regularization, by default DEFAULT_SOFTORTHOGONAL_L2
        use_matrix_scaling : bool, optional
            Whether to scale regularization by matrix size, by default False
        **kwargs : Any
            Additional arguments passed to parent regularizer

        Raises
        ------
        ValueError
            If any coefficient is negative
        """
        super().__init__(**kwargs)

        # Validate input parameters
        if lambda_coefficient < 0.0:
            raise ValueError(f"lambda_coefficient must be non-negative, got {lambda_coefficient}")
        if l1_coefficient < 0.0:
            raise ValueError(f"l1_coefficient must be non-negative, got {l1_coefficient}")
        if l2_coefficient < 0.0:
            raise ValueError(f"l2_coefficient must be non-negative, got {l2_coefficient}")

        self._lambda_coefficient = lambda_coefficient
        self._l1_coefficient = l1_coefficient
        self._l2_coefficient = l2_coefficient
        self._use_matrix_scaling = use_matrix_scaling

        # Cache flags for performance optimization
        self._use_lambda = self._lambda_coefficient > 0.0
        self._use_l1 = self._l1_coefficient > 0.0
        self._use_l2 = self._l2_coefficient > 0.0

        # Initialize L1/L2 regularizers once for efficiency
        self._l1: Optional[keras.regularizers.L1] = None
        self._l2: Optional[keras.regularizers.L2] = None

        if self._use_l1:
            self._l1 = keras.regularizers.L1(l1=self._l1_coefficient)
        if self._use_l2:
            self._l2 = keras.regularizers.L2(l2=self._l2_coefficient)

        logger.debug(
            f"Initialized SoftOrthonormalConstraintRegularizer with "
            f"lambda={lambda_coefficient}, l1={l1_coefficient}, "
            f"l2={l2_coefficient}, scaling={use_matrix_scaling}"
        )

    def __call__(self, x: Union[keras.KerasTensor, Any], **kwargs) -> Union[keras.KerasTensor, Any]:
        """Compute regularization loss for given weights.

        Parameters
        ----------
        x : Union[keras.KerasTensor, Any]
            Weight tensor to regularize
        **kwargs : Any
            Additional keyword arguments (e.g., dtype) that Keras may pass

        Returns
        -------
        Union[keras.KerasTensor, Any]
            Scalar regularization loss value

        Notes
        -----
        The regularization loss is computed as:
        - Frobenius norm of (W^T * W - I)
        - Optional L1/L2 regularization terms
        - Optional matrix scaling for size-invariant regularization
        """
        # Compute Gram matrix W^T * W
        wt_w = gram_matrix(x)

        # Get matrix rank and create identity matrix
        rank = ops.shape(wt_w)[0]
        eye = ops.eye(rank, dtype=wt_w.dtype)

        # Initialize regularization loss
        result = ops.cast(0.0, dtype=x.dtype)

        # Add Frobenius norm term if enabled
        if self._use_lambda:
            # Calculate squared Frobenius norm of (W^T*W - I)
            deviation = ops.subtract(wt_w, eye)
            frob_norm = ops.square(ops.norm(deviation, ord=STR_FRO, axis=(0, 1)))

            if self._use_matrix_scaling:
                # Scale by total number of matrix elements for size invariance
                total_elements = ops.cast(ops.multiply(rank, rank), dtype=x.dtype)
                scaling_factor = ops.maximum(total_elements, ops.cast(EPSILON, dtype=x.dtype))
                scaled_loss = ops.divide(ops.multiply(self._lambda_coefficient, frob_norm), scaling_factor)
                result = ops.add(result, scaled_loss)
            else:
                # Original behavior without scaling
                lambda_loss = ops.multiply(self._lambda_coefficient, frob_norm)
                result = ops.add(result, lambda_loss)

        # Add L1 regularization if enabled
        if self._use_l1 and self._l1 is not None:
            l1_loss = self._l1(x)
            result = ops.add(result, l1_loss)

        # Add L2 regularization if enabled
        if self._use_l2 and self._l2 is not None:
            l2_loss = self._l2(x)
            result = ops.add(result, l2_loss)

        return result

    def get_config(self) -> Dict[str, Any]:
        """Get regularizer configuration for serialization.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing configuration parameters

        Notes
        -----
        This method is required for proper serialization and deserialization
        of models containing this regularizer.
        """
        config = {}
        config.update({
            STR_L1_COEFFICIENT: self._l1_coefficient,
            STR_L2_COEFFICIENT: self._l2_coefficient,
            STR_LAMBDA_COEFFICIENT: self._lambda_coefficient,
            STR_USE_MATRIX_SCALING: self._use_matrix_scaling
        })
        return config

# ---------------------------------------------------------------------