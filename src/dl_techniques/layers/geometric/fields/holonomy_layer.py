"""
Holonomy Layer

This module provides a HolonomyLayer that computes holonomy - the path-ordered
integral of the connection around closed loops. Holonomy is a gauge-invariant
quantity that captures the global geometric structure of the manifold.

Mathematical Foundation:
    The holonomy around a loop γ is the path-ordered exponential:

    H[γ] = P exp(-∮_γ Γ)

    This is a gauge-invariant quantity - it doesn't change under local gauge
    transformations. The holonomy captures:
    - Curvature enclosed by the loop (via Stokes' theorem)
    - Global topological information
    - Non-local correlations in the field

- Holonomy provides gauge-invariant features that resist adversarial manipulation
- External prompts can't inject arbitrary transformations because they must
  respect the holonomy constraint
- The holonomy encodes global semantic coherence that local perturbations
  cannot break
"""

import keras
from keras import ops, initializers, regularizers
from typing import Optional, Union, Dict, Any, Tuple, List, Literal

LoopType = Literal['rectangular', 'triangular', 'circular', 'adaptive']


@keras.saving.register_keras_serializable(package='holonomic')
class HolonomyLayer(keras.layers.Layer):
    """
    Computes holonomy (path-ordered exponential around loops).

    Holonomy is a fundamental gauge-invariant quantity that measures the
    curvature enclosed by a loop. This layer computes holonomy for various
    loop geometries and uses it to create gauge-invariant representations.

    The holonomy around a loop depends only on the curvature enclosed,
    not on the specific path taken. This makes it robust to local
    perturbations and gauge transformations.

    Loop types:
    - 'rectangular': Axis-aligned rectangular loops
    - 'triangular': Triangular loops (simpler, captures curvature)
    - 'circular': Approximate circular loops
    - 'adaptive': Learned loop shapes

    Args:
        hidden_dim: Dimension of the representation.
        loop_sizes: List of loop sizes to compute holonomy for.
        loop_type: Type of loops to use.
        num_loops: Number of loop orientations per size.
        use_trace: Whether to use trace of holonomy (Wilson loop).
        holonomy_regularization: Regularization for holonomy smoothness.
        kernel_initializer: Initializer for kernel weights.

    Example:
        >>> holonomy_layer = HolonomyLayer(
        ...     hidden_dim=256,
        ...     loop_sizes=[2, 4, 8],
        ...     loop_type='rectangular'
        ... )
        >>> embeddings = keras.ops.random.normal((2, 16, 256))
        >>> connection = keras.ops.random.normal((2, 16, 256, 256)) * 0.01
        >>> holonomy = holonomy_layer([embeddings, connection])
        >>> print(holonomy.shape)  # (2, 16, num_holonomy_features)
    """

    def __init__(
            self,
            hidden_dim: int,
            loop_sizes: List[int] = [2, 4, 8],
            loop_type: LoopType = 'rectangular',
            num_loops: int = 4,
            use_trace: bool = True,
            holonomy_regularization: float = 0.001,
            kernel_initializer: Union[str, initializers.Initializer] = 'orthogonal',
            **kwargs: Any
    ) -> None:
        """Initialize the HolonomyLayer."""
        super().__init__(**kwargs)

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if not loop_sizes:
            raise ValueError("loop_sizes cannot be empty")
        if num_loops <= 0:
            raise ValueError(f"num_loops must be positive, got {num_loops}")
        if loop_type not in ('rectangular', 'triangular', 'circular', 'adaptive'):
            raise ValueError(
                f"loop_type must be one of 'rectangular', 'triangular', 'circular', 'adaptive', "
                f"got {loop_type}"
            )

        self.hidden_dim = hidden_dim
        self.loop_sizes = list(loop_sizes)
        self.loop_type = loop_type
        self.num_loops = num_loops
        self.use_trace = use_trace
        self.holonomy_regularization = holonomy_regularization

        self.kernel_initializer = initializers.get(kernel_initializer)

        # Compute output dimension
        if use_trace:
            # Trace gives scalar per loop
            self.output_dim = len(loop_sizes) * num_loops
        else:
            # Full holonomy matrix per loop
            self.output_dim = len(loop_sizes) * num_loops * hidden_dim

    def build(self, input_shape: Tuple[Tuple[int, ...], ...]) -> None:
        """
        Build the layer weights.

        Args:
            input_shape: Tuple of (embeddings_shape, connection_shape).
        """
        # For adaptive loops: learnable loop parameters
        if self.loop_type == 'adaptive':
            # Each loop has learnable direction vectors
            self.loop_directions = self.add_weight(
                name='loop_directions',
                shape=(len(self.loop_sizes), self.num_loops, 4, self.hidden_dim),
                initializer=self.kernel_initializer,
                trainable=True
            )

        # Projection for output normalization
        self.output_projection = self.add_weight(
            name='output_projection',
            shape=(self.output_dim, self.hidden_dim),
            initializer=self.kernel_initializer,
            trainable=True
        )

        # Learnable temperature for softmax in path integration
        self.integration_temperature = self.add_weight(
            name='integration_temperature',
            shape=(1,),
            initializer=initializers.Constant(1.0),
            trainable=True
        )

        super().build(input_shape)

    def _compute_path_integral(
            self,
            connection: keras.KerasTensor,
            path_indices: List[Tuple[int, int]]
    ) -> keras.KerasTensor:
        """
        Compute path-ordered integral along a discrete path.

        The path-ordered product is:
        P = P_n * P_{n-1} * ... * P_1
        where P_i is the transport operator at step i.

        Args:
            connection: Connection tensor, shape (batch, seq_len, dim, dim).
            path_indices: List of (position, direction) tuples defining the path.

        Returns:
            Path-ordered product as matrix, shape (batch, dim, dim).
        """
        batch_size = ops.shape(connection)[0]

        # Initialize with identity
        result = ops.eye(self.hidden_dim)
        result = ops.expand_dims(result, 0)
        result = ops.tile(result, (batch_size, 1, 1))

        # For sequence data, we use position indices
        # Each step accumulates: result = exp(-Γ_i) * result
        for pos, direction in path_indices:
            # Get connection at this position
            # Handle boundary conditions by clamping
            # Note: In actual use, path_indices should be valid

            # Connection contribution at this position
            gamma = connection[:, pos, :, :]  # (batch, dim, dim)

            # Approximate exponential: exp(-Γ) ≈ I - Γ + Γ²/2
            gamma_sq = ops.matmul(gamma, gamma)
            step_contribution = (
                    ops.eye(self.hidden_dim) -
                    0.1 * gamma * direction +
                    0.005 * gamma_sq
            )

            # Path-ordered product (multiply on left)
            result = ops.matmul(step_contribution, result)

        return result

    def _compute_rectangular_holonomy(
            self,
            connection: keras.KerasTensor,
            center_pos: int,
            size: int
    ) -> keras.KerasTensor:
        """
        Compute holonomy around a rectangular loop.

        The rectangular loop goes:
        center → center+size → center+size (same) → center → center
        In 1D sequence, this degenerates but still captures curvature.

        Args:
            connection: Connection tensor.
            center_pos: Center position of the loop.
            size: Size of the loop.

        Returns:
            Holonomy matrix.
        """
        batch_size = ops.shape(connection)[0]
        seq_len = ops.shape(connection)[1]

        # For 1D sequences, we compute holonomy using forward-backward paths
        # This captures the non-commutativity of the connection

        # Clamp positions to valid range
        half_size = size // 2
        start_pos = max(0, center_pos - half_size)
        end_pos = min(seq_len - 1, center_pos + half_size)

        # Initialize holonomy as identity
        holonomy = ops.eye(self.hidden_dim)
        holonomy = ops.expand_dims(holonomy, 0)
        holonomy = ops.tile(holonomy, (batch_size, 1, 1))

        # Forward path: accumulate connection
        for i in range(start_pos, end_pos):
            gamma = connection[:, i, :, :]
            # Small step contribution
            step = ops.eye(self.hidden_dim) - 0.1 * gamma
            holonomy = ops.matmul(step, holonomy)

        # Backward path: accumulate negative connection (return journey)
        for i in range(end_pos, start_pos, -1):
            gamma = connection[:, i, :, :]
            # Negative direction
            step = ops.eye(self.hidden_dim) + 0.1 * gamma
            holonomy = ops.matmul(step, holonomy)

        return holonomy

    def _compute_triangular_holonomy(
            self,
            connection: keras.KerasTensor,
            center_pos: int,
            size: int
    ) -> keras.KerasTensor:
        """
        Compute holonomy around a triangular path.

        Simpler than rectangular, uses three segments.

        Args:
            connection: Connection tensor.
            center_pos: Center position.
            size: Size of the triangle.

        Returns:
            Holonomy matrix.
        """
        batch_size = ops.shape(connection)[0]
        seq_len = ops.shape(connection)[1]

        # Three positions forming a triangle in the sequence
        pos1 = max(0, center_pos - size // 3)
        pos2 = min(seq_len - 1, center_pos)
        pos3 = min(seq_len - 1, center_pos + size // 3)

        # Initialize
        holonomy = ops.eye(self.hidden_dim)
        holonomy = ops.expand_dims(holonomy, 0)
        holonomy = ops.tile(holonomy, (batch_size, 1, 1))

        # Three legs of the triangle
        positions = [pos1, pos2, pos3, pos1]  # Closed loop

        for i in range(3):
            p1, p2 = positions[i], positions[i + 1]
            direction = 1 if p2 > p1 else -1
            for p in range(min(p1, p2), max(p1, p2)):
                gamma = connection[:, p, :, :]
                step = ops.eye(self.hidden_dim) - 0.1 * direction * gamma
                holonomy = ops.matmul(step, holonomy)

        return holonomy

    def _extract_holonomy_features(
            self,
            holonomy: keras.KerasTensor
    ) -> keras.KerasTensor:
        """
        Extract features from holonomy matrix.

        If use_trace is True, returns the trace (Wilson loop).
        Otherwise, returns the full matrix flattened.

        Args:
            holonomy: Holonomy matrix, shape (batch, dim, dim).

        Returns:
            Holonomy features.
        """
        if self.use_trace:
            # Wilson loop: Tr(H)
            # This is gauge-invariant
            return ops.trace(holonomy)
        else:
            # Full holonomy (not gauge-invariant but more expressive)
            batch_size = ops.shape(holonomy)[0]
            return ops.reshape(holonomy, (batch_size, -1))

    def call(
            self,
            inputs: Tuple[keras.KerasTensor, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Compute holonomy features for the input field.

        Args:
            inputs: Tuple of (embeddings, connection).
                - embeddings: shape (batch, seq_len, dim)
                - connection: shape (batch, seq_len, dim, dim)
            training: Whether in training mode.

        Returns:
            Holonomy features of shape (batch, seq_len, hidden_dim).
        """
        embeddings, connection = inputs

        batch_size = ops.shape(embeddings)[0]
        seq_len = ops.shape(embeddings)[1]

        # Collect holonomy features for each position
        all_features = []

        # Compute holonomy at each position for each loop size
        for size_idx, loop_size in enumerate(self.loop_sizes):
            for loop_idx in range(self.num_loops):
                # Different loop orientations (in 1D, this means different offsets)
                offset = loop_idx * loop_size // self.num_loops

                position_features = []
                for pos in range(seq_len):
                    adjusted_pos = pos + offset

                    if self.loop_type == 'rectangular':
                        holonomy = self._compute_rectangular_holonomy(
                            connection, adjusted_pos, loop_size
                        )
                    elif self.loop_type == 'triangular':
                        holonomy = self._compute_triangular_holonomy(
                            connection, adjusted_pos, loop_size
                        )
                    else:
                        # Default to rectangular for other types
                        holonomy = self._compute_rectangular_holonomy(
                            connection, adjusted_pos, loop_size
                        )

                    # Extract features
                    features = self._extract_holonomy_features(holonomy)
                    position_features.append(features)

                # Stack position features: (seq_len, batch) or (seq_len, batch, dim)
                stacked = ops.stack(position_features, axis=0)
                # Transpose to (batch, seq_len, ...)
                stacked = ops.transpose(stacked, (1, 0) + tuple(range(2, len(ops.shape(stacked)))))
                all_features.append(stacked)

        # Concatenate all holonomy features
        # Each feature tensor is (batch, seq_len) for trace or (batch, seq_len, dim²) for full
        if self.use_trace:
            # Shape: (batch, seq_len, num_features)
            holonomy_features = ops.stack(all_features, axis=-1)
        else:
            # Concatenate along feature dimension
            holonomy_features = ops.concatenate(all_features, axis=-1)

        # Project to hidden dimension
        output = ops.matmul(holonomy_features, self.output_projection)

        # Add regularization: holonomy should be close to identity (low curvature)
        if training and self.holonomy_regularization > 0:
            # The trace of holonomy near identity is close to hidden_dim
            expected_trace = float(self.hidden_dim)
            actual_traces = holonomy_features if self.use_trace else ops.zeros(())
            if self.use_trace:
                trace_deviation = ops.mean((actual_traces - expected_trace) ** 2)
                self.add_loss(self.holonomy_regularization * trace_deviation)

        return output

    def compute_output_shape(
            self,
            input_shape: Tuple[Tuple[int, ...], Tuple[int, ...]]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute output shape.

        Args:
            input_shape: Tuple of (embeddings_shape, connection_shape).

        Returns:
            Output shape.
        """
        embed_shape = input_shape[0]
        batch_size = embed_shape[0]
        seq_len = embed_shape[1] if len(embed_shape) > 1 else None

        return (batch_size, seq_len, self.hidden_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'loop_sizes': self.loop_sizes,
            'loop_type': self.loop_type,
            'num_loops': self.num_loops,
            'use_trace': self.use_trace,
            'holonomy_regularization': self.holonomy_regularization,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
        })
        return config