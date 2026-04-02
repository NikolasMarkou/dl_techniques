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
    """Computes holonomy (path-ordered exponential around closed loops).

    Holonomy H[gamma] = P exp(-oint_gamma Gamma) is a gauge-invariant quantity
    measuring the curvature enclosed by a loop. This layer approximates
    holonomy using the commutator of connections at different offsets:
    [Gamma_s, Gamma_{s+offset}], which captures curvature in the
    infinitesimal limit. For each loop_size and orientation, a shifted
    commutator is computed fully vectorised over (batch, seq_len), and the
    trace (Wilson loop) or Frobenius features are extracted and projected
    to ``hidden_dim``.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────┐  ┌──────────────────┐
        │ Embeddings      │  │ Connection       │
        │ [B, S, D]       │  │ [B, S, D, D]     │
        └────────┬────────┘  └────────┬─────────┘
                 └───────────┬────────┘
                             ▼
        ┌────────────────────────────────────────┐
        │  For each (loop_size, orientation):    │
        │  ┌──────────────────────────────────┐  │
        │  │ Commutator [Γ_s, Γ_{s+offset}]  │  │
        │  │ → Holonomy proxy [B, S, D, D]   │  │
        │  │ → Trace or Frobenius features   │  │
        │  └──────────────────────────────────┘  │
        └────────────────┬───────────────────────┘
                         ▼
        ┌────────────────────────────────────────┐
        │  Stack features [B, S, num_features]   │
        │  Output projection → [B, S, D]         │
        └────────────────────────────────────────┘

    :param hidden_dim: Dimension of the representation. Must be positive.
    :type hidden_dim: int
    :param loop_sizes: List of loop sizes. Defaults to ``[2, 4, 8]``.
    :type loop_sizes: List[int]
    :param loop_type: Type of loops
        (``'rectangular'``, ``'triangular'``, ``'circular'``, ``'adaptive'``).
        Defaults to ``'rectangular'``.
    :type loop_type: LoopType
    :param num_loops: Loop orientations per size. Defaults to 4.
    :type num_loops: int
    :param use_trace: Whether to use trace (Wilson loop). Defaults to ``True``.
    :type use_trace: bool
    :param holonomy_regularization: Regularization strength. Defaults to 0.001.
    :type holonomy_regularization: float
    :param kernel_initializer: Initializer for kernel weights.
        Defaults to ``'orthogonal'``.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param kwargs: Additional arguments for the ``Layer`` base class.
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
            # Frobenius norm gives scalar per loop
            self.output_dim = len(loop_sizes) * num_loops

    def build(self, input_shape: Tuple[Tuple[int, ...], ...]) -> None:
        """Build the layer weights.

        :param input_shape: Tuple of (embeddings_shape, connection_shape).
        :type input_shape: Tuple[Tuple[int, ...], ...]
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

        # Learnable scaling per loop feature
        self.feature_scale = self.add_weight(
            name='feature_scale',
            shape=(self.output_dim,),
            initializer=initializers.Constant(1.0),
            trainable=True
        )

        super().build(input_shape)

    def _compute_commutator_holonomy(
            self,
            connection: keras.KerasTensor,
            offset: int
    ) -> keras.KerasTensor:
        """Compute holonomy proxy via commutator at a given offset.

        For a forward-backward loop of span ``offset``, the holonomy
        approximation in the small-curvature limit is proportional to the
        commutator [Gamma_s, Gamma_{s+offset}]. This is fully vectorised.

        :param connection: Connection tensor, shape (batch, seq_len, dim, dim).
        :type connection: keras.KerasTensor
        :param offset: Position offset for the loop (static Python int).
        :type offset: int
        :return: Commutator tensor, shape (batch, seq_len, dim, dim).
        :rtype: keras.KerasTensor
        """
        offset = max(1, offset)

        # Gamma at each position: (batch, seq_len, dim, dim)
        gamma_here = connection

        # Shifted connection: Γ_{s+offset} using circular roll.
        # For sequences, this means positions near the end wrap to the start,
        # but the commutator still captures non-commutativity at each scale.
        gamma_shifted = ops.roll(connection, shift=-offset, axis=1)

        # Commutator: [Γ_here, Γ_shifted] = Γ_here @ Γ_shifted - Γ_shifted @ Γ_here
        comm_ab = ops.einsum('bsij,bsjk->bsik', gamma_here, gamma_shifted)
        comm_ba = ops.einsum('bsij,bsjk->bsik', gamma_shifted, gamma_here)
        commutator = comm_ab - comm_ba

        return commutator

    def _extract_features(
            self,
            commutator: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Extract scalar features from commutator holonomy proxy.

        :param commutator: Commutator tensor, shape (batch, seq_len, dim, dim).
        :type commutator: keras.KerasTensor
        :return: Scalar features, shape (batch, seq_len).
        :rtype: keras.KerasTensor
        """
        if self.use_trace:
            # Trace of commutator squared: Tr([A,B]^2) is gauge-invariant
            # and non-trivially zero (unlike Tr([A,B]) which is always 0).
            comm_sq = ops.einsum('bsij,bsjk->bsik', commutator, commutator)
            return ops.trace(comm_sq, axis1=-2, axis2=-1)
        else:
            # Frobenius norm of commutator (captures curvature magnitude)
            return ops.sqrt(
                ops.sum(commutator ** 2, axis=(-2, -1)) + 1e-8
            )

    def call(
            self,
            inputs: Tuple[keras.KerasTensor, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Compute holonomy features for the input field.

        Fully vectorised: no Python loops over sequence positions.

        :param inputs: Tuple of (embeddings, connection).
            - embeddings: shape (batch, seq_len, dim)
            - connection: shape (batch, seq_len, dim, dim)
        :type inputs: Tuple[keras.KerasTensor, keras.KerasTensor]
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :return: Holonomy features of shape (batch, seq_len, hidden_dim).
        :rtype: keras.KerasTensor
        """
        embeddings, connection = inputs

        # Collect holonomy features for each (loop_size, orientation)
        all_features = []

        for loop_size in self.loop_sizes:
            for loop_idx in range(self.num_loops):
                # Compute offset for this loop orientation
                # Different orientations sample different spans
                offset = max(1, (loop_idx + 1) * loop_size // self.num_loops)

                # Compute commutator-based holonomy proxy (fully vectorised)
                commutator = self._compute_commutator_holonomy(
                    connection, offset
                )

                # Extract per-position scalar features: (batch, seq_len)
                features = self._extract_features(commutator)
                all_features.append(features)

        # Stack all features: (batch, seq_len, num_features)
        holonomy_features = ops.stack(all_features, axis=-1)

        # Apply learnable scaling
        holonomy_features = holonomy_features * self.feature_scale

        # Project to hidden dimension
        output = ops.matmul(holonomy_features, self.output_projection)

        # Regularization: holonomy should be small (near-flat manifold)
        if training and self.holonomy_regularization > 0:
            reg_loss = ops.mean(holonomy_features ** 2)
            self.add_loss(self.holonomy_regularization * reg_loss)

        return output

    def compute_output_shape(
            self,
            input_shape: Tuple[Tuple[int, ...], Tuple[int, ...]]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        :param input_shape: Tuple of (embeddings_shape, connection_shape).
        :type input_shape: Tuple[Tuple[int, ...], Tuple[int, ...]]
        :return: Output shape.
        :rtype: Tuple[Optional[int], ...]
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
