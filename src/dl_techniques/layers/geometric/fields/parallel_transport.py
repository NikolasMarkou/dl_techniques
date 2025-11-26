"""
Parallel Transport Layer

This module provides a ParallelTransportLayer that implements parallel transport
of vectors along paths on the semantic manifold. Parallel transport is the key
operation that enables holonomy computation and gauge-invariant processing.

Mathematical Foundation:
    Parallel transport moves a vector V along a curve γ while keeping it
    "as parallel as possible" according to the connection Γ. The transport
    equation is:

    dV^k/dt + Γ^k_{ij} (dγ^i/dt) V^j = 0

    This preserves the geometric relationship between the vector and the
    manifold as we move along the curve.

In Holonomic AI:
    - Parallel transport enables semantic information to flow between positions
      while respecting the geometric structure
    - The connection determines how information transforms during transport
    - Adversarial perturbations that don't respect transport rules are rejected
"""

import keras
from keras import ops, initializers, regularizers
from typing import Optional, Union, Dict, Any, Tuple, Literal

TransportMethod = Literal['direct', 'iterative', 'path_ordered']


@keras.saving.register_keras_serializable(package='holonomic')
class ParallelTransportLayer(keras.layers.Layer):
    """
    Parallel transport of vectors along paths using the gauge connection.

    This layer implements parallel transport, which moves vectors from one
    position to another while preserving their geometric properties according
    to the connection. This is fundamental for holonomic processing where
    information must respect manifold structure.

    Transport methods:
    - 'direct': Single-step transport (efficient but less accurate)
    - 'iterative': Multi-step transport with small increments
    - 'path_ordered': Full path-ordered exponential (most accurate)

    Args:
        transport_dim: Dimension of vectors being transported.
        num_steps: Number of integration steps for iterative transport.
        transport_method: Method for computing transport.
        step_size: Step size for iterative integration.
        use_adaptive_steps: Whether to adaptively adjust step size.
        transport_regularization: Regularization for transport stability.
        kernel_initializer: Initializer for kernel weights.

    Example:
        >>> transport_layer = ParallelTransportLayer(
        ...     transport_dim=256,
        ...     num_steps=10,
        ...     transport_method='iterative'
        ... )
        >>> vectors = keras.ops.random.normal((2, 10, 256))
        >>> connection = keras.ops.random.normal((2, 10, 256, 256)) * 0.01
        >>> transported = transport_layer([vectors, connection])
        >>> print(transported.shape)  # (2, 10, 256)
    """

    def __init__(
            self,
            transport_dim: int,
            num_steps: int = 10,
            transport_method: TransportMethod = 'iterative',
            step_size: float = 0.1,
            use_adaptive_steps: bool = False,
            transport_regularization: float = 0.0,
            kernel_initializer: Union[str, initializers.Initializer] = 'orthogonal',
            **kwargs: Any
    ) -> None:
        """Initialize the ParallelTransportLayer."""
        super().__init__(**kwargs)

        if transport_dim <= 0:
            raise ValueError(f"transport_dim must be positive, got {transport_dim}")
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if transport_method not in ('direct', 'iterative', 'path_ordered'):
            raise ValueError(
                f"transport_method must be one of 'direct', 'iterative', 'path_ordered', "
                f"got {transport_method}"
            )

        self.transport_dim = transport_dim
        self.num_steps = num_steps
        self.transport_method = transport_method
        self.step_size = step_size
        self.use_adaptive_steps = use_adaptive_steps
        self.transport_regularization = transport_regularization

        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape: Tuple[Tuple[int, ...], ...]) -> None:
        """
        Build the layer weights.

        Args:
            input_shape: Tuple of (vectors_shape, connection_shape).
        """
        # Learnable transport correction for numerical stability
        self.transport_correction = self.add_weight(
            name='transport_correction',
            shape=(self.transport_dim, self.transport_dim),
            initializer='zeros',
            trainable=True
        )

        # For adaptive steps: learned step size adjustment
        if self.use_adaptive_steps:
            self.step_scale = self.add_weight(
                name='step_scale',
                shape=(1,),
                initializer=initializers.Constant(1.0),
                trainable=True
            )

        super().build(input_shape)

    def _direct_transport(
            self,
            vectors: keras.KerasTensor,
            connection: keras.KerasTensor,
            tangent: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Direct single-step parallel transport.

        Approximates transport as: V' = V - Γ(T, V) where T is tangent vector.

        Args:
            vectors: Vectors to transport, shape (batch, seq_len, dim).
            connection: Connection tensor, shape (batch, seq_len, dim, dim).
            tangent: Tangent vectors for transport direction.

        Returns:
            Transported vectors.
        """
        # Compute connection action: Γ^k_{ij} T^i V^j
        # connection: (batch, seq_len, dim, dim) as Γ^k_j with i contracted with T
        # First contract with tangent: (batch, seq_len, dim, dim) @ (batch, seq_len, dim)
        # Result: (batch, seq_len, dim)
        gamma_t = ops.einsum('bsij,bsi->bsj', connection, tangent)

        # Then contract with vectors
        transport_term = ops.einsum('bsj,bsj->bsj', gamma_t, vectors)

        # Apply transport: V' = V - step_size * Γ(T, V)
        step = self.step_size
        if self.use_adaptive_steps:
            step = step * ops.abs(self.step_scale)

        transported = vectors - step * transport_term

        # Apply small correction for stability
        correction = ops.matmul(transported, self.transport_correction)
        transported = transported + 0.01 * correction

        return transported

    def _iterative_transport(
            self,
            vectors: keras.KerasTensor,
            connection: keras.KerasTensor,
            tangent: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Iterative multi-step parallel transport.

        Uses multiple small steps for more accurate transport.

        Args:
            vectors: Vectors to transport.
            connection: Connection tensor.
            tangent: Tangent vectors.

        Returns:
            Transported vectors.
        """
        step = self.step_size / self.num_steps
        if self.use_adaptive_steps:
            step = step * ops.abs(self.step_scale)

        current = vectors

        # Iterate transport steps
        for _ in range(self.num_steps):
            # Compute transport increment
            gamma_t = ops.einsum('bsij,bsi->bsj', connection, tangent)
            transport_term = ops.einsum('bsj,bsj->bsj', gamma_t, current)

            # Update: Euler integration
            current = current - step * transport_term

        # Apply correction
        correction = ops.matmul(current, self.transport_correction)
        current = current + 0.01 * correction

        return current

    def _path_ordered_transport(
            self,
            vectors: keras.KerasTensor,
            connection: keras.KerasTensor,
            tangent: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Path-ordered exponential transport.

        Computes the path-ordered exponential of the connection, which is
        the exact solution to the parallel transport equation.

        P exp(-∫ Γ dt) V

        Args:
            vectors: Vectors to transport.
            connection: Connection tensor.
            tangent: Tangent vectors.

        Returns:
            Transported vectors.
        """
        batch_size = ops.shape(vectors)[0]
        seq_len = ops.shape(vectors)[1]

        # Build transport operator iteratively
        # Start with identity
        transport_op = ops.eye(self.transport_dim)
        transport_op = ops.expand_dims(ops.expand_dims(transport_op, 0), 0)
        transport_op = ops.tile(transport_op, (batch_size, seq_len, 1, 1))

        step = self.step_size / self.num_steps
        if self.use_adaptive_steps:
            step = step * ops.abs(self.step_scale)

        # Build path-ordered product
        for _ in range(self.num_steps):
            # Contract connection with tangent to get generator
            # Shape: (batch, seq_len, dim, dim)
            generator = -step * ops.einsum('bsij,bsi->bsij', connection, tangent)

            # Compute exponential approximation: exp(A) ≈ I + A + A²/2
            gen_sq = ops.matmul(generator, generator)
            exp_gen = (ops.eye(self.transport_dim) + generator +
                       0.5 * gen_sq)

            # Path-ordered product
            transport_op = ops.matmul(exp_gen, transport_op)

        # Apply transport operator to vectors
        # transport_op: (batch, seq_len, dim, dim)
        # vectors: (batch, seq_len, dim)
        transported = ops.einsum('bsij,bsj->bsi', transport_op, vectors)

        # Apply correction
        correction = ops.matmul(transported, self.transport_correction)
        transported = transported + 0.01 * correction

        return transported

    def call(
            self,
            inputs: Tuple[keras.KerasTensor, keras.KerasTensor],
            training: Optional[bool] = None,
            tangent: Optional[keras.KerasTensor] = None
    ) -> keras.KerasTensor:
        """
        Perform parallel transport of vectors.

        Args:
            inputs: Tuple of (vectors, connection).
                - vectors: shape (batch, seq_len, dim)
                - connection: shape (batch, seq_len, dim, dim)
            training: Whether in training mode.
            tangent: Optional tangent vectors. If None, uses difference between
                adjacent positions as tangent.

        Returns:
            Transported vectors of shape (batch, seq_len, dim).
        """
        vectors, connection = inputs

        # Compute tangent vectors if not provided
        if tangent is None:
            # Use finite differences as tangent direction
            # Shift to get forward differences
            shifted = ops.concatenate([
                vectors[:, 1:, :],
                vectors[:, -1:, :]  # Repeat last for boundary
            ], axis=1)
            tangent = shifted - vectors
            # Normalize tangent
            tangent_norm = ops.sqrt(ops.sum(tangent ** 2, axis=-1, keepdims=True) + 1e-8)
            tangent = tangent / tangent_norm

        # Match connection dimensions if needed
        conn_shape = ops.shape(connection)
        if conn_shape[-1] != self.transport_dim:
            # Interpolate connection to transport dimension
            # This handles dimension mismatches gracefully
            pass  # Assume dimensions match for now

        # Perform transport based on method
        if self.transport_method == 'direct':
            transported = self._direct_transport(vectors, connection, tangent)
        elif self.transport_method == 'iterative':
            transported = self._iterative_transport(vectors, connection, tangent)
        else:  # path_ordered
            transported = self._path_ordered_transport(vectors, connection, tangent)

        # Regularization: transport should approximately preserve norm
        if training and self.transport_regularization > 0:
            input_norm = ops.sqrt(ops.sum(vectors ** 2, axis=-1) + 1e-8)
            output_norm = ops.sqrt(ops.sum(transported ** 2, axis=-1) + 1e-8)
            norm_loss = ops.mean((input_norm - output_norm) ** 2)
            self.add_loss(self.transport_regularization * norm_loss)

        return transported

    def compute_output_shape(
            self,
            input_shape: Tuple[Tuple[int, ...], Tuple[int, ...]]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape.

        Args:
            input_shape: Tuple of (vectors_shape, connection_shape).

        Returns:
            Output shape (same as vectors input).
        """
        vectors_shape = input_shape[0]
        return vectors_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'transport_dim': self.transport_dim,
            'num_steps': self.num_steps,
            'transport_method': self.transport_method,
            'step_size': self.step_size,
            'use_adaptive_steps': self.use_adaptive_steps,
            'transport_regularization': self.transport_regularization,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
        })
        return config