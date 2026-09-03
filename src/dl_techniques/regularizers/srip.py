"""Near-orthonormality via a spectral norm penalty (SRIP).

Provides :class:`SRIPRegularizer`, which penalizes the spectral norm of
``W^T W - I``, and :func:`create_srip_regularizer`, a factory for it.

Pushing a weight matrix toward an isometry, a transformation that preserves
Euclidean norm, keeps signal and gradient magnitudes from growing or shrinking
exponentially across layers. The regularizer works on dense (2D) and
convolutional (4D) kernels; a conv kernel is reshaped into an equivalent
matrix first.

The penalty
-----------
``W`` is a perfect isometry when its Gram matrix ``W^T W`` is the identity.
The penalty measures how far it is from that::

    Loss = lambda * ||W^T W - I||_2

``||.||_2`` is the spectral norm, the largest singular value. It bounds the
largest change in a vector's norm caused by the deviation, so minimizing it
drives every singular value of ``W^T W`` toward 1.

An exact spectral norm needs an SVD, which is expensive. This implementation
uses power iteration, which approximates the largest singular value without a
full decomposition.

Reference
---------
-   Bansal, N., Chen, X., & Wang, Z. (2018). "Can We Gain More from
    Orthogonality Regularizations in Training Deep CNNs?". *Advances in Neural
    Information Processing Systems (NeurIPS)*.
"""

import keras
from keras import ops
from typing import Optional, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.regularizers.srip")
class SRIPRegularizer(keras.regularizers.Regularizer):
    """Penalize the spectral norm of ``W^T W - I`` to enforce near-orthonormality.

    Works on dense and convolutional kernels: a conv kernel is reshaped to 2D
    before the Gram matrix is built. The regularization strength can be
    stepped down over training by calling :meth:`update_lambda` from a
    callback.

    **Penalty pipeline:**

    .. code-block:: text

        weights (any rank >= 2)
             |
             v
        ┌────────────────────────────────┐
        │ rescale if ||W|| > 1e8         │  overflow guard on the Gram
        └───────────────┬────────────────┘
                        v
        ┌────────────────────────────────┐
        │ _reshape_kernel                │
        │  4D (H,W,Cin,Cout) -> 2D       │
        │  2D passed through             │
        │  other -> flatten all but last │
        └───────────────┬────────────────┘
                        │ [fan_in, units]
                        v
        ┌────────────────────────────────┐
        │ G = W^T W ;  G - I             │
        └───────────────┬────────────────┘
                        │ [units, units]
                        v
        ┌────────────────────────────────┐
        │ _power_iteration               │
        │  power_iterations forward and  │
        │  backward products, then       │
        │  ||Mv|| / ||v||                │
        └───────────────┬────────────────┘
                        │ spectral norm
                        v
        ┌────────────────────────────────┐
        │ if max|W| < epsilon: epsilon   │  near-zero weights give a noisy
        └───────────────┬────────────────┘  norm, so return the floor
                        v
                * current_lambda
                        v
                     scalar

    **Default lambda schedule:**

    .. code-block:: text

        epoch >=    lambda
        --------    ------
        0           lambda_init
        20          1e-3
        50          1e-4
        70          1e-6
        120         0.0

    The schedule is inert until :meth:`update_lambda` is called with the
    current epoch; nothing steps it automatically.

    :param lambda_init: Initial regularization strength. Must be non-negative.
    :type lambda_init: float
    :param power_iterations: Number of power iterations. More iterations give a
        more accurate spectral norm at higher cost. Must be at least 1.
    :type power_iterations: int
    :param epsilon: Numerical stability constant, also the value returned for
        near-zero weights. Must be positive.
    :type epsilon: float
    :param lambda_schedule: Maps epoch to lambda value. ``None`` installs the
        default schedule above. Keys must be non-negative ints, values
        non-negative numbers.
    :type lambda_schedule: dict or None
    :param kwargs: Must be empty. ``keras.regularizers.Regularizer`` defines no
        ``__init__``, so any keyword forwarded here reaches ``object.__init__``
        and raises ``TypeError``.

    :ivar lambda_init: The initial regularization strength.
    :vartype lambda_init: float
    :ivar power_iterations: Number of power iteration steps.
    :vartype power_iterations: int
    :ivar epsilon: Numerical stability constant.
    :vartype epsilon: float
    :ivar lambda_schedule: Mapping of epoch to lambda value.
    :vartype lambda_schedule: dict

    :raises ValueError: If any parameter is outside its valid range.
    :raises TypeError: If any keyword argument is supplied.
    """

    def __init__(
        self,
        lambda_init: float = 0.1,
        power_iterations: int = 2,
        epsilon: float = 1e-7,
        lambda_schedule: Optional[Dict[int, float]] = None,
        **kwargs: Any
    ) -> None:
        """Validate the settings and install the lambda schedule.

        :param lambda_init: Non-negative initial regularization strength.
        :type lambda_init: float
        :param power_iterations: Number of power iterations, at least 1.
        :type power_iterations: int
        :param epsilon: Positive numerical stability constant.
        :type epsilon: float
        :param lambda_schedule: Epoch-to-lambda mapping, or ``None`` for the
            default schedule.
        :type lambda_schedule: dict or None
        :param kwargs: Must be empty; see the class docstring.
        :raises ValueError: If any parameter is outside its valid range.
        :raises TypeError: If any keyword argument is supplied.
        """
        super().__init__(**kwargs)

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

        self._current_lambda = self.lambda_init

        logger.debug(
            f"Initialized SRIPRegularizer with lambda_init={self.lambda_init}, "
            f"power_iterations={self.power_iterations}, epsilon={self.epsilon}, "
            f"lambda_schedule={self.lambda_schedule}"
        )

    @property
    def current_lambda(self) -> float:
        """Current regularization strength.

        :return: The lambda value in effect, as set by :meth:`update_lambda`.
        :rtype: float
        """
        return self._current_lambda

    def _validate_init_params(
        self,
        lambda_init: float,
        power_iterations: int,
        epsilon: float,
        lambda_schedule: Optional[Dict[int, float]]
    ) -> None:
        """Check the constructor arguments.

        :param lambda_init: Initial regularization strength.
        :type lambda_init: float
        :param power_iterations: Number of power iterations.
        :type power_iterations: int
        :param epsilon: Numerical stability constant.
        :type epsilon: float
        :param lambda_schedule: Optional epoch-to-lambda mapping.
        :type lambda_schedule: dict or None
        :return: Nothing.
        :rtype: None
        :raises ValueError: If ``lambda_init`` is negative,
            ``power_iterations`` is below 1, ``epsilon`` is not positive, or a
            schedule key is not a non-negative int or a value is negative.
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
        """Flatten a kernel to 2D for the Gram matrix computation.

        A 4D conv kernel ``(H, W, C_in, C_out)`` becomes
        ``(H*W*C_in, C_out)``. A 2D kernel passes through. Any other rank has
        every axis but the last flattened.

        :param kernel: The kernel tensor.
        :type kernel: tensor
        :return: The kernel as a 2D matrix.
        :rtype: tensor
        """
        kernel_shape = ops.shape(kernel)

        # Conv2D kernel (H, W, C_in, C_out): flatten spatial and input channels.
        if len(kernel.shape) == 4:
            flattened_size = kernel_shape[0] * kernel_shape[1] * kernel_shape[2]
            return ops.reshape(kernel, [flattened_size, kernel_shape[3]])
        # Dense kernel (input_dim, output_dim): already 2D.
        elif len(kernel.shape) == 2:
            return kernel
        else:
            flattened_size = ops.prod(kernel_shape[:-1])
            return ops.reshape(kernel, [flattened_size, kernel_shape[-1]])

    def _safe_normalize(self, vector) -> keras.KerasTensor:
        """Scale a vector to unit L2 norm, with epsilon in the denominator.

        :param vector: The tensor to normalize.
        :type vector: tensor
        :return: The normalized tensor.
        :rtype: tensor
        """
        squared_norm = ops.sum(ops.square(vector), axis=0, keepdims=True)
        safe_norm = ops.sqrt(squared_norm + self.epsilon)
        normalized = vector / safe_norm
        return normalized

    def _power_iteration(self, matrix) -> keras.KerasTensor:
        """Approximate a matrix's spectral norm by power iteration.

        Each iteration does a forward and a backward multiplication, then the
        largest singular value is read off as ``||Mv|| / ||v||``.

        The starting vector is seeded from the matrix shape, so every matrix of
        the same shape starts from the same vector. Power iteration converges
        regardless of the start, so this is harmless.

        :param matrix: A 2D matrix.
        :type matrix: tensor
        :return: The largest singular value.
        :rtype: tensor
        :raises ValueError: If the input is not 2-dimensional.
        """
        if len(matrix.shape) != 2:
            raise ValueError("Input matrix must be 2-dimensional")

        matrix_shape = ops.shape(matrix)

        init_seed = ops.sum(matrix_shape) % 2147483647
        vector = keras.random.normal(
            shape=[matrix_shape[1], 1],
            seed=int(init_seed),
            dtype=matrix.dtype
        )

        vector_norm = ops.sqrt(ops.sum(ops.square(vector)) + self.epsilon)
        vector = vector / vector_norm

        for _ in range(self.power_iterations):
            # Forward: matrix-vector product.
            product = ops.matmul(matrix, vector)
            product_norm = ops.sqrt(ops.sum(ops.square(product)) + self.epsilon)
            vector = product / product_norm

            # Backward: transpose multiplication.
            product = ops.matmul(ops.transpose(matrix), vector)
            product_norm = ops.sqrt(ops.sum(ops.square(product)) + self.epsilon)
            vector = product / product_norm

        product = ops.matmul(matrix, vector)

        product_norm = ops.sqrt(ops.sum(ops.square(product)))
        vector_norm = ops.sqrt(ops.sum(ops.square(vector)) + self.epsilon)

        spectral_norm = product_norm / vector_norm
        return spectral_norm

    def __call__(self, weights) -> keras.KerasTensor:
        """Compute the SRIP penalty ``current_lambda * ||W^T W - I||_2``.

        :param weights: Weight tensor to regularize.
        :type weights: tensor
        :return: The scalar penalty.
        :rtype: tensor
        :raises ValueError: If the reshaped kernel is not 2-dimensional.
        """
        # Rescale very large weights so the Gram matmul cannot overflow.
        # ops.where keeps the branch backend-agnostic.
        weights_norm = ops.sqrt(ops.sum(ops.square(weights)) + self.epsilon)
        large_threshold = ops.cast(1e8, dtype=weights.dtype)
        weights = ops.where(
            weights_norm > large_threshold,
            weights / weights_norm,
            weights,
        )

        weights_2d = self._reshape_kernel(weights)

        gram = ops.matmul(ops.transpose(weights_2d), weights_2d)
        identity = ops.eye(ops.shape(gram)[0], dtype=weights.dtype)
        gram_centered = gram - identity

        spec_norm = self._power_iteration(gram_centered)

        # Near-zero weights give a Gram matrix whose spectral norm is mostly
        # noise, so return the epsilon floor instead.
        weights_abs_max = ops.max(ops.abs(weights))
        epsilon_tensor = ops.cast(self.epsilon, dtype=weights.dtype)
        spec_norm = ops.where(
            weights_abs_max < epsilon_tensor,
            epsilon_tensor,
            spec_norm,
        )

        regularization_loss = ops.cast(self.current_lambda, dtype=weights.dtype) * spec_norm

        return regularization_loss

    def update_lambda(self, epoch: int) -> None:
        """Set the regularization strength from the schedule for ``epoch``.

        Call this from a callback; nothing advances the schedule on its own.
        The value taken is the one for the largest scheduled epoch at or below
        ``epoch``, falling back to ``lambda_init``.

        :param epoch: Current training epoch.
        :type epoch: int
        :return: Nothing.
        :rtype: None
        """
        current_lambda = self.lambda_init
        for e, lambda_val in sorted(self.lambda_schedule.items()):
            if epoch >= e:
                current_lambda = lambda_val

        if current_lambda != self._current_lambda:
            logger.info(f"Updated SRIP lambda from {self._current_lambda} to {current_lambda} at epoch {epoch}")
            self._current_lambda = current_lambda

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments for serialization.

        :return: A dict holding ``lambda_init``, ``power_iterations``,
            ``epsilon`` and ``lambda_schedule``.
        :rtype: dict
        """
        return {
            'lambda_init': self.lambda_init,
            'power_iterations': self.power_iterations,
            'epsilon': self.epsilon,
            'lambda_schedule': {int(k): float(v) for k, v in self.lambda_schedule.items()},
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SRIPRegularizer':
        """Rebuild a regularizer from a config dict.

        Schedule keys arrive as strings after a JSON round trip, so they are
        coerced back to ints here.

        :param config: Configuration dictionary from :meth:`get_config`.
        :type config: dict
        :return: A new regularizer.
        :rtype: SRIPRegularizer
        """
        if 'lambda_schedule' in config:
            config['lambda_schedule'] = {int(k): float(v)
                                         for k, v in config['lambda_schedule'].items()}
        return cls(**config)

    def __repr__(self) -> str:
        """Return the constructor-like representation.

        :return: A string naming the settings and the current lambda.
        :rtype: str
        """
        return (
            f"SRIPRegularizer("
            f"lambda_init={self.lambda_init}, "
            f"power_iterations={self.power_iterations}, "
            f"epsilon={self.epsilon}, "
            f"current_lambda={self.current_lambda})"
        )


def create_srip_regularizer(
    lambda_init: Optional[float] = 0.1,
    power_iterations: Optional[int] = 2,
    epsilon: Optional[float] = 1e-7,
    lambda_schedule: Optional[Dict[int, float]] = None
) -> SRIPRegularizer:
    """Build a :class:`SRIPRegularizer`.

    All validation lives in the constructor; this adds nothing but the call.

    :param lambda_init: Initial regularization strength. Larger values enforce
        orthogonality more strongly.
    :type lambda_init: float or None
    :param power_iterations: Number of power iterations for the spectral norm.
        More iterations are more accurate and more expensive.
    :type power_iterations: int or None
    :param epsilon: Numerical stability constant.
    :type epsilon: float or None
    :param lambda_schedule: Epoch-to-lambda mapping for decay scheduling.
        ``None`` uses the default schedule.
    :type lambda_schedule: dict or None
    :return: The configured regularizer.
    :rtype: SRIPRegularizer
    :raises ValueError: If any parameter is outside its valid range.

    Example:
        >>> # Default parameters
        >>> regularizer = create_srip_regularizer()
        >>> conv_layer = keras.layers.Conv2D(64, 3, kernel_regularizer=regularizer)
        >>>
        >>> # Stronger initial orthogonality constraint
        >>> strong_regularizer = create_srip_regularizer(lambda_init=0.5)
        >>> dense_layer = keras.layers.Dense(128, kernel_regularizer=strong_regularizer)
        >>>
        >>> # Custom decay schedule
        >>> custom_schedule = {10: 0.05, 30: 0.01, 50: 0.001}
        >>> scheduled_regularizer = create_srip_regularizer(
        ...     lambda_init=0.1, lambda_schedule=custom_schedule
        ... )
        >>>
        >>> # Step the schedule from a callback
        >>> # scheduled_regularizer.update_lambda(current_epoch)
    """
    return SRIPRegularizer(
        lambda_init=lambda_init,
        power_iterations=power_iterations,
        epsilon=epsilon,
        lambda_schedule=lambda_schedule
    )

# ---------------------------------------------------------------------
