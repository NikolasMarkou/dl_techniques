"""
SRIP (Spectral Restricted Isometry Property) Regularizer Implementation
====================================================================

This implementation follows the paper:
"Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?"
by Bansal et al. (arXiv:1810.09102)

The SRIP regularizer enforces approximate orthogonality of weight matrices
by minimizing the spectral norm of W^T W - I.
"""

import keras
import tensorflow as tf
from typing import Optional, Dict, Any, Union, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.tensors import power_iteration

# ---------------------------------------------------------------------


@keras.utils.register_keras_serializable()
class SRIPRegularizer(keras.regularizers.Regularizer):
    """
    Spectral Restricted Isometry Property (SRIP) regularizer.

    Enforces near-orthogonality of weight matrices using spectral norm minimization.
    Supports both dense and convolutional layers.

    Attributes:
        lambda_init: Initial regularization strength
        power_iterations: Number of power iterations for spectral norm computation
        epsilon: Small constant for numerical stability
        lambda_schedule: Dictionary mapping epochs to lambda values
        current_lambda: Current regularization strength (updated during training)
    """

    def __init__(
            self,
            lambda_init: float = 0.1,
            power_iterations: int = 2,
            epsilon: float = 1e-7,
            lambda_schedule: Optional[Dict[int, float]] = None,
            **kwargs: Any
    ) -> None:
        """
        Initialize SRIP regularizer.

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

        # Validate and store parameters
        self._validate_init_params(lambda_init, power_iterations, epsilon, lambda_schedule)

        self.lambda_init = lambda_init
        self.power_iterations = power_iterations
        self.epsilon = epsilon
        self.lambda_schedule = lambda_schedule or {
            20: 1e-3,
            50: 1e-4,
            70: 1e-6,
            120: 0.0
        }

        # Initialize lambda variable for training
        self._current_lambda = tf.Variable(
            lambda_init,
            trainable=False,
            dtype=tf.float32,
            name='srip_lambda'
        )

    @property
    def current_lambda(self) -> tf.Tensor:
        """Current regularization strength."""
        return self._current_lambda

    def _validate_init_params(
            self,
            lambda_init: float,
            power_iterations: int,
            epsilon: float,
            lambda_schedule: Optional[Dict[int, float]]
    ) -> None:
        """
        Validate initialization parameters.

        Args:
            lambda_init: Initial regularization strength
            power_iterations: Number of power iterations
            epsilon: Numerical stability constant
            lambda_schedule: Optional lambda decay schedule

        Raises:
            ValueError: If any parameters are invalid
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

    def _reshape_kernel(self, kernel: tf.Tensor) -> tf.Tensor:
        """
        Reshape kernel for gram matrix computation.

        Args:
            kernel: Input kernel tensor

        Returns:
            Reshaped kernel tensor
        """
        if len(kernel.shape) == 4:  # Conv2D kernel
            kernel_shape = tf.shape(kernel)
            flattened_size = kernel_shape[0] * kernel_shape[1] * kernel_shape[2]
            return tf.reshape(kernel, [flattened_size, kernel_shape[3]])
        return kernel

    def _safe_normalize(self, vector: tf.Tensor) -> tf.Tensor:
        """
        Safely normalize a vector with numerical stability.

        Args:
            vector: Input tensor to normalize

        Returns:
            Normalized tensor with unit L2 norm
        """
        # Compute L2 norm with epsilon for stability
        squared_norm = tf.reduce_sum(tf.square(vector), axis=0)
        safe_norm = tf.sqrt(squared_norm + self.epsilon)
        normalized = vector / safe_norm

        return normalized

    def _power_iteration(self, matrix: tf.Tensor) -> tf.Tensor:
        """
        Compute spectral norm using power iteration.

        Args:
            matrix: Input matrix

        Returns:
            Spectral norm (largest singular value)
        """
        return power_iteration(matrix=matrix, iterations=self.power_iterations, epsilon=self.epsilon)

    @tf.function
    def __call__(self, weights: tf.Tensor) -> tf.Tensor:
        """
        Compute SRIP regularization term.

        Args:
            weights: Weight tensor to regularize

        Returns:
            Regularization loss value
        """
        if tf.reduce_all(tf.abs(weights) < self.epsilon):
            return self.epsilon

        # For numerical stability with large weights, normalize the input
        weights_norm = tf.norm(weights)
        if weights_norm > 1e8:  # Threshold for "large" weights
            weights = weights / weights_norm

        # Reshape weights if needed
        weights = self._reshape_kernel(weights)

        # Compute gram matrix (W^T W - I)
        gram = tf.matmul(weights, weights, transpose_a=True)
        identity = tf.eye(tf.shape(gram)[0], dtype=weights.dtype)
        gram_centered = gram - identity

        # Compute spectral norm
        spec_norm = self._power_iteration(gram_centered)
        return self.current_lambda * spec_norm

    def update_lambda(self, epoch: int) -> None:
        """
        Update lambda value based on current epoch.

        Args:
            epoch: Current training epoch
        """
        current_lambda = self.lambda_init
        for e, lambda_val in sorted(self.lambda_schedule.items()):
            if epoch >= e:
                current_lambda = lambda_val
        self._current_lambda.assign(current_lambda)

    def get_config(self):
        config = {}
        config.update({
            'lambda_init': self.lambda_init,
            'power_iterations': self.power_iterations,
            'epsilon': self.epsilon,
            'lambda_schedule': {int(k): float(v) for k, v in self.lambda_schedule.items()}
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SRIPRegularizer':
        """
        Create regularizer instance from configuration dictionary.

        Args:
            config: Dictionary containing configuration parameters

        Returns:
            New regularizer instance
        """
        if 'lambda_schedule' in config:
            config['lambda_schedule'] = {int(k): float(v)
                                         for k, v in config['lambda_schedule'].items()}
        return cls(**config)
