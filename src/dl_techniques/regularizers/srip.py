"""
This module provides a Keras implementation of the Spectral Restricted Isometry
Property (SRIP) regularizer, a powerful technique for improving the training of
deep neural networks, particularly Convolutional Neural Networks (CNNs).

The core idea behind this regularizer, as proposed by Bansal et al. in "Can We Gain
More from Orthogonality Regularizations in Training Deep CNNs?", is that encouraging
the weight matrices of a network to be "approximately orthogonal" can lead to
significant benefits. An orthogonal matrix preserves the norm of vectors, which helps
to mitigate the exploding and vanishing gradient problems during training. This leads
to more stable training, faster convergence, and often better generalization.

How it Works:

1.  **The Orthogonality Condition:** A matrix `W` is orthogonal if its transpose is
    also its inverse, which means `W^T * W = I` (where `I` is the identity matrix).
    This regularizer aims to enforce this condition.

2.  **The SRIP Loss Function:** Instead of a strict enforcement, SRIP encourages
    *near-orthogonality* by penalizing the deviation of `W^T * W` from the identity
    matrix `I`. The loss is calculated as:
    `Loss = lambda * ||W^T * W - I||_2`
    where `||...||_2` is the spectral norm (the largest singular value of the matrix).
    Minimizing this loss pushes the Gram matrix `W^T * W` to be close to `I`.

3.  **Spectral Norm Estimation:** Calculating the exact spectral norm is computationally
    expensive. This implementation uses the **power iteration method**, an efficient
    iterative algorithm, to approximate the largest singular value. The number of
    `power_iterations` controls the trade-off between accuracy and computational cost.

4.  **Support for Convolutional and Dense Layers:** The regularizer is designed to
    work with both `Dense` and `Conv2D` layers. For convolutional kernels, which are
    4D tensors, it automatically reshapes them into 2D matrices so that the `W^T * W`
    operation can be performed.

5.  **Lambda Scheduling:** The strength of the regularization is controlled by a
    parameter, `lambda`. This implementation includes a dynamic scheduling mechanism.
    The `update_lambda` method allows the regularization strength to be decayed over
    the course of training (e.g., via a Keras callback). This is often beneficial, as
    strong regularization is most needed in the early stages of training, and can be
    relaxed later to allow for fine-tuning.

By adding this regularizer to the kernel of `Dense` or `Conv2D` layers, one can
improve the conditioning of the network's weight matrices, leading to a more robust
and effective training process.
"""

import keras
from keras import ops
from typing import Optional, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SRIPRegularizer(keras.regularizers.Regularizer):
    """Spectral Restricted Isometry Property (SRIP) regularizer.

    Enforces near-orthogonality of weight matrices using spectral norm minimization.
    Supports both dense and convolutional layers by reshaping convolutional kernels
    into 2D matrices and computing the spectral norm of the Gram matrix W^T W - I.

    The regularizer includes a lambda scheduling mechanism that allows the regularization
    strength to be adjusted during training. The update_lambda method should be called
    externally (e.g., via a Keras callback) to update the regularization strength.

    Args:
        lambda_init: Initial regularization strength. Must be non-negative.
        power_iterations: Number of power iterations for spectral norm computation.
            Higher values give more accurate results but increase computation time.
        epsilon: Small constant for numerical stability. Must be positive.
        lambda_schedule: Optional dictionary mapping epochs to lambda values.
            Used for decay scheduling. Values must be non-negative.

    Attributes:
        lambda_init (float): The initial regularization strength.
        power_iterations (int): Number of power iteration steps.
        epsilon (float): Numerical stability constant.
        lambda_schedule (Dict[int, float]): Mapping of epochs to lambda values.
        current_lambda (float): Current regularization strength (read-only property).
    """

    def __init__(
        self,
        lambda_init: float = 0.1,
        power_iterations: int = 2,
        epsilon: float = 1e-7,
        lambda_schedule: Optional[Dict[int, float]] = None,
        **kwargs: Any
    ) -> None:
        """Initialize SRIP regularizer.

        Args:
            lambda_init: Initial regularization strength. Must be non-negative.
            power_iterations: Number of power iterations for spectral norm computation.
                Higher values give more accurate results but increase computation time.
            epsilon: Small constant for numerical stability. Must be positive.
            lambda_schedule: Optional dictionary mapping epochs to lambda values.
                Used for decay scheduling. Values must be non-negative.
            **kwargs: Additional arguments passed to parent class.

        Raises:
            ValueError: If any parameters have invalid values.
        """
        super().__init__(**kwargs)

        # Validate parameters
        self._validate_init_params(lambda_init, power_iterations, epsilon, lambda_schedule)

        self.lambda_init = float(lambda_init)
        self.power_iterations = int(power_iterations)
        self.epsilon = float(epsilon)
        self.lambda_schedule = lambda_schedule or {
            20: 1e-3,
            50: 1e-4,
            70: 1e-6,
            120: 0.0
        }

        # Current lambda value (will be updated via update_lambda method)
        self._current_lambda = self.lambda_init

        logger.info(
            f"Initialized SRIPRegularizer with lambda_init={self.lambda_init}, "
            f"power_iterations={self.power_iterations}, epsilon={self.epsilon}, "
            f"lambda_schedule={self.lambda_schedule}"
        )

    @property
    def current_lambda(self) -> float:
        """Current regularization strength."""
        return self._current_lambda

    def _validate_init_params(
        self,
        lambda_init: float,
        power_iterations: int,
        epsilon: float,
        lambda_schedule: Optional[Dict[int, float]]
    ) -> None:
        """Validate initialization parameters.

        Args:
            lambda_init: Initial regularization strength.
            power_iterations: Number of power iterations.
            epsilon: Numerical stability constant.
            lambda_schedule: Optional lambda decay schedule.

        Raises:
            ValueError: If any parameters are invalid.
        """
        if lambda_init < 0:
            raise ValueError(f"lambda_init must be non-negative, got {lambda_init}")
        if power_iterations < 1:
            raise ValueError(f"power_iterations must be positive, got {power_iterations}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        if lambda_schedule:
            if not all(isinstance(k, int) and k >= 0 for k in lambda_schedule.keys()):
                raise ValueError("Lambda schedule epochs must be non-negative integers")
            if not all(isinstance(v, (int, float)) and v >= 0 for v in lambda_schedule.values()):
                raise ValueError("Lambda schedule values must be non-negative")

    def _reshape_kernel(self, kernel) -> keras.KerasTensor:
        """Reshape kernel for gram matrix computation.

        Converts convolutional kernels from 4D (H, W, C_in, C_out) to 2D
        (H*W*C_in, C_out) format for matrix operations.

        Args:
            kernel: Input kernel tensor.

        Returns:
            Reshaped kernel tensor as 2D matrix.
        """
        kernel_shape = ops.shape(kernel)

        if len(kernel.shape) == 4:  # Conv2D kernel (H, W, C_in, C_out)
            # Flatten spatial and input channel dimensions
            flattened_size = kernel_shape[0] * kernel_shape[1] * kernel_shape[2]
            return ops.reshape(kernel, [flattened_size, kernel_shape[3]])
        elif len(kernel.shape) == 2:  # Dense kernel (input_dim, output_dim)
            return kernel
        else:
            # For other kernel shapes, flatten all but the last dimension
            flattened_size = ops.prod(kernel_shape[:-1])
            return ops.reshape(kernel, [flattened_size, kernel_shape[-1]])

    def _safe_normalize(self, vector) -> keras.KerasTensor:
        """Safely normalize a vector with numerical stability.

        Args:
            vector: Input tensor to normalize.

        Returns:
            Normalized tensor with unit L2 norm.
        """
        # Compute L2 norm with epsilon for stability
        squared_norm = ops.sum(ops.square(vector), axis=0, keepdims=True)
        safe_norm = ops.sqrt(squared_norm + self.epsilon)
        normalized = vector / safe_norm
        return normalized

    def _power_iteration(self, matrix) -> keras.KerasTensor:
        """Compute spectral norm using power iteration method.

        This implementation follows the original power iteration algorithm,
        using forward and backward multiplication in each iteration to find
        the largest singular value.

        Args:
            matrix: Input 2D matrix for which to compute spectral norm.

        Returns:
            Spectral norm (largest singular value) of the input matrix.

        Raises:
            ValueError: If input matrix is not 2-dimensional.
        """
        if len(matrix.shape) != 2:
            raise ValueError("Input matrix must be 2-dimensional")

        matrix_shape = ops.shape(matrix)

        # Initialize random vector
        # Use a deterministic initialization based on matrix shape for reproducibility
        init_seed = ops.sum(matrix_shape) % 2147483647  # Large prime for seed
        vector = keras.random.normal(
            shape=[matrix_shape[1], 1],
            seed=int(init_seed),
            dtype=matrix.dtype
        )

        # Normalize initial vector
        vector_norm = ops.sqrt(ops.sum(ops.square(vector)) + self.epsilon)
        vector = vector / vector_norm

        # Multiple iterations for convergence
        for _ in range(self.power_iterations):
            # Compute matrix-vector product (forward)
            product = ops.matmul(matrix, vector)
            product_norm = ops.sqrt(ops.sum(ops.square(product)) + self.epsilon)
            vector = product / product_norm

            # Compute transpose multiplication (backward)
            product = ops.matmul(ops.transpose(matrix), vector)
            product_norm = ops.sqrt(ops.sum(ops.square(product)) + self.epsilon)
            vector = product / product_norm

        # Final power iteration step
        product = ops.matmul(matrix, vector)

        # Compute spectral norm using the ratio of norms
        product_norm = ops.sqrt(ops.sum(ops.square(product)))
        vector_norm = ops.sqrt(ops.sum(ops.square(vector)) + self.epsilon)

        spectral_norm = product_norm / vector_norm
        return spectral_norm

    def __call__(self, weights) -> keras.KerasTensor:
        """Compute SRIP regularization term.

        The regularization loss is computed as:
        lambda_reg * ||W^T W - I||_2 (spectral norm)

        Args:
            weights: Weight tensor to regularize.

        Returns:
            Regularization loss value as a scalar tensor.

        Raises:
            ValueError: If weights tensor has invalid shape.
        """
        # Handle edge case of very small weights
        weights_abs_max = ops.max(ops.abs(weights))
        if weights_abs_max < self.epsilon:
            return ops.cast(self.epsilon, dtype=weights.dtype)

        # Numerical stability: normalize very large weights
        weights_norm = ops.sqrt(ops.sum(ops.square(weights)))
        if weights_norm > 1e8:  # Threshold for "large" weights
            weights = weights / weights_norm
            logger.debug(f"Normalized large weights with norm {weights_norm}")

        # Reshape weights if needed (handles both Dense and Conv layers)
        try:
            weights_2d = self._reshape_kernel(weights)
        except Exception as e:
            logger.error(f"Failed to reshape weights with shape {weights.shape}: {e}")
            raise ValueError(f"Cannot reshape weights with shape {weights.shape}")

        # Ensure we have a valid 2D matrix
        if len(weights_2d.shape) != 2:
            raise ValueError(f"Expected 2D matrix after reshaping, got shape {weights_2d.shape}")

        # Compute gram matrix (W^T W - I)
        gram = ops.matmul(ops.transpose(weights_2d), weights_2d)
        identity = ops.eye(ops.shape(gram)[0], dtype=weights.dtype)
        gram_centered = gram - identity

        # Compute spectral norm of the centered gram matrix
        spec_norm = self._power_iteration(gram_centered)

        # Apply regularization strength
        regularization_loss = self.current_lambda * spec_norm

        return regularization_loss

    def update_lambda(self, epoch: int) -> None:
        """Update lambda value based on current epoch.

        This method should be called externally (e.g., via a callback) to update
        the regularization strength according to the defined schedule.

        Args:
            epoch: Current training epoch.
        """
        current_lambda = self.lambda_init
        for e, lambda_val in sorted(self.lambda_schedule.items()):
            if epoch >= e:
                current_lambda = lambda_val

        if current_lambda != self._current_lambda:
            logger.info(f"Updated SRIP lambda from {self._current_lambda} to {current_lambda} at epoch {epoch}")
            self._current_lambda = current_lambda

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the regularizer.

        Returns:
            Configuration dictionary containing the regularizer parameters.
        """
        return {
            'lambda_init': self.lambda_init,
            'power_iterations': self.power_iterations,
            'epsilon': self.epsilon,
            'lambda_schedule': {int(k): float(v) for k, v in self.lambda_schedule.items()}
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SRIPRegularizer':
        """Create regularizer instance from configuration dictionary.

        Args:
            config: Dictionary containing configuration parameters.

        Returns:
            New SRIPRegularizer instance.
        """
        if 'lambda_schedule' in config:
            config['lambda_schedule'] = {int(k): float(v)
                                         for k, v in config['lambda_schedule'].items()}
        return cls(**config)

    def __repr__(self) -> str:
        """Return string representation of the regularizer.

        Returns:
            String representation including key parameters.
        """
        return (
            f"SRIPRegularizer("
            f"lambda_init={self.lambda_init}, "
            f"power_iterations={self.power_iterations}, "
            f"epsilon={self.epsilon}, "
            f"current_lambda={self.current_lambda})"
        )


def get_srip_regularizer(
    lambda_init: Optional[float] = 0.1,
    power_iterations: Optional[int] = 2,
    epsilon: Optional[float] = 1e-7,
    lambda_schedule: Optional[Dict[int, float]] = None
) -> SRIPRegularizer:
    """Factory function to create a SRIP regularizer instance.

    This function provides a convenient way to create SRIP regularizer instances
    with sensible defaults and parameter validation.

    Args:
        lambda_init: Initial regularization strength. Higher values enforce stronger
            orthogonality constraints. Defaults to 0.1.
        power_iterations: Number of power iterations for spectral norm computation.
            Higher values give more accurate spectral norms but increase computation.
            Defaults to 2.
        epsilon: Small constant for numerical stability. Defaults to 1e-7.
        lambda_schedule: Optional dictionary mapping epochs to lambda values for
            decay scheduling. If None, uses default schedule.

    Returns:
        An instance of the SRIPRegularizer.

    Raises:
        ValueError: If any parameters have invalid values.

    Example:
        >>> # Create SRIP regularizer with default parameters
        >>> regularizer = get_srip_regularizer()
        >>> conv_layer = keras.layers.Conv2D(64, 3, kernel_regularizer=regularizer)
        >>>
        >>> # Create SRIP regularizer with stronger initial orthogonality constraint
        >>> strong_regularizer = get_srip_regularizer(lambda_init=0.5)
        >>> dense_layer = keras.layers.Dense(128, kernel_regularizer=strong_regularizer)
        >>>
        >>> # Create SRIP regularizer with custom decay schedule
        >>> custom_schedule = {10: 0.05, 30: 0.01, 50: 0.001}
        >>> scheduled_regularizer = get_srip_regularizer(
        ...     lambda_init=0.1, lambda_schedule=custom_schedule
        ... )
        >>>
        >>> # Update lambda during training (e.g., in a callback)
        >>> # scheduled_regularizer.update_lambda(current_epoch)
    """
    return SRIPRegularizer(
        lambda_init=lambda_init,
        power_iterations=power_iterations,
        epsilon=epsilon,
        lambda_schedule=lambda_schedule
    )


# Alias for backward compatibility
srip_regularizer = get_srip_regularizer

# ---------------------------------------------------------------------