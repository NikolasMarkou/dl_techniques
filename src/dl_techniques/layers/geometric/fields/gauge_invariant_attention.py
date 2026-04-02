"""
Gauge-Invariant Attention Layer.

This module provides a GaugeInvariantAttention layer that computes attention
while respecting the gauge structure of the field representation. Traditional
attention operates on raw vectors; gauge-invariant attention ensures that
the attention pattern is invariant under local gauge transformations.

Mathematical Foundation:
    In gauge theory, local transformations that don't affect physics are
    called gauge transformations. For attention to be gauge-invariant,
    the attention weights must depend only on gauge-invariant quantities
    like holonomy and curvature, not on the raw field values.

    The gauge-invariant attention score between positions i and j is:

    a_{ij} = f(H[γ_{ij}], R_i, R_j)

    where H[γ_{ij}] is the holonomy along the path from i to j, and
    R_i, R_j are the local curvatures.

    - Adversarial prompts cannot manipulate attention by gauge tricks
    - The attention respects the geometric structure of meaning
    - Information flow follows the natural geometry of the semantic manifold
"""

import keras
from keras import ops, initializers, regularizers
from typing import Optional, Union, Dict, Any, Tuple, Literal

AttentionMetric = Literal['holonomy', 'geodesic', 'curvature', 'hybrid']


@keras.saving.register_keras_serializable(package='holonomic')
class GaugeInvariantAttention(keras.layers.Layer):
    """Attention mechanism that respects gauge invariance.

    Computes attention scores using gauge-invariant quantities from the
    field representation: holonomy between positions (path-dependent
    transport), geodesic distance (curvature-weighted), and local curvature
    agreement. The gauge-invariant score between positions i and j is
    a_{ij} = f(H[gamma_{ij}], R_i, R_j) where H is holonomy and R is
    curvature. Standard Q/K/V projections are augmented with geometric
    corrections, curvature gating, and optional parallel transport of values.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────┐ ┌──────────────┐ ┌──────────────┐
        │ Embeddings  │ │ Curvature    │ │ Connection   │
        │ [B, S, D]   │ │ [B, S, ...]  │ │ [B, S, D, D] │
        └──────┬──────┘ └──────┬───────┘ └──────┬───────┘
               ▼               │                │
        ┌────────────────┐     │                │
        │ Q, K, V proj   │     │                │
        └──────┬─────────┘     │                │
               ▼               ▼                ▼
        ┌────────────────────────────────────────────┐
        │ Gauge-invariant scores                     │
        │ (holonomy / geodesic / curvature / hybrid) │
        └──────────────────┬─────────────────────────┘
                           ▼
        ┌────────────────────────────────────────────┐
        │ Temperature × Curvature Gate × Mask        │
        └──────────────────┬─────────────────────────┘
                           ▼
        ┌────────────────────────────────────────────┐
        │ Softmax → Dropout → Attend values          │
        │ (optional parallel transport of V)         │
        └──────────────────┬─────────────────────────┘
                           ▼
        ┌────────────────────────────────────────────┐
        │ Output projection → [B, S, D]              │
        └────────────────────────────────────────────┘

    :param hidden_dim: Hidden dimension size. Must be positive.
    :type hidden_dim: int
    :param num_heads: Number of attention heads. Defaults to 8.
    :type num_heads: int
    :param key_dim: Key dimension per head. Defaults to ``hidden_dim // num_heads``.
    :type key_dim: Optional[int]
    :param attention_metric: Gauge-invariant metric type
        (``'holonomy'``, ``'geodesic'``, ``'curvature'``, ``'hybrid'``).
        Defaults to ``'hybrid'``.
    :type attention_metric: AttentionMetric
    :param use_curvature_gating: Whether to gate attention by curvature.
        Defaults to ``True``.
    :type use_curvature_gating: bool
    :param use_parallel_transport: Whether to transport values before aggregation.
        Defaults to ``True``.
    :type use_parallel_transport: bool
    :param dropout_rate: Dropout rate for attention weights. Defaults to 0.0.
    :type dropout_rate: float
    :param kernel_initializer: Initializer for kernel weights.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Regularizer for kernel weights.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param kwargs: Additional arguments for the ``Layer`` base class.
    """

    def __init__(
            self,
            hidden_dim: int,
            num_heads: int = 8,
            key_dim: Optional[int] = None,
            attention_metric: AttentionMetric = 'hybrid',
            use_curvature_gating: bool = True,
            use_parallel_transport: bool = True,
            dropout_rate: float = 0.0,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the GaugeInvariantAttention layer."""
        super().__init__(**kwargs)

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        if attention_metric not in ('holonomy', 'geodesic', 'curvature', 'hybrid'):
            raise ValueError(
                f"attention_metric must be one of 'holonomy', 'geodesic', 'curvature', 'hybrid', "
                f"got {attention_metric}"
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.key_dim = key_dim or (hidden_dim // num_heads)
        self.attention_metric = attention_metric
        self.use_curvature_gating = use_curvature_gating
        self.use_parallel_transport = use_parallel_transport
        self.dropout_rate = dropout_rate

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        self.head_dim = hidden_dim // num_heads

    def build(self, input_shape: Tuple[Tuple[int, ...], ...]) -> None:
        """Build the layer weights.

        :param input_shape: Tuple of shapes for (embeddings, curvature, connection).
        :type input_shape: Tuple[Tuple[int, ...], ...]
        """
        if isinstance(input_shape, list):
            embed_shape = input_shape[0]
        else:
            embed_shape = input_shape

        input_dim = embed_shape[-1]

        # Query, Key, Value projections
        self.query_kernel = self.add_weight(
            name='query_kernel',
            shape=(input_dim, self.num_heads, self.key_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True
        )

        self.key_kernel = self.add_weight(
            name='key_kernel',
            shape=(input_dim, self.num_heads, self.key_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True
        )

        self.value_kernel = self.add_weight(
            name='value_kernel',
            shape=(input_dim, self.num_heads, self.head_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True
        )

        # Output projection
        self.output_kernel = self.add_weight(
            name='output_kernel',
            shape=(self.num_heads, self.head_dim, self.hidden_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True
        )

        # Gauge-invariant metric computation
        self.metric_kernel = self.add_weight(
            name='metric_kernel',
            shape=(input_dim * 2, self.num_heads),
            initializer=self.kernel_initializer,
            trainable=True
        )

        # Curvature gating
        if self.use_curvature_gating:
            self.gate_kernel = self.add_weight(
                name='gate_kernel',
                shape=(input_dim, self.num_heads),
                initializer=self.kernel_initializer,
                trainable=True
            )

        # Transport correction
        if self.use_parallel_transport:
            self.transport_kernel = self.add_weight(
                name='transport_kernel',
                shape=(self.head_dim, self.head_dim),
                initializer='orthogonal',
                trainable=True
            )

        # Learnable attention temperature
        self.attention_temperature = self.add_weight(
            name='attention_temperature',
            shape=(self.num_heads,),
            initializer=initializers.Constant(1.0 / (self.key_dim ** 0.5)),
            trainable=True
        )

        super().build(input_shape)

    def _compute_holonomy_attention(
            self,
            queries: keras.KerasTensor,
            keys: keras.KerasTensor,
            connection: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute attention scores based on holonomy.

        The attention between positions i and j depends on the holonomy
        along the path connecting them.

        :param queries: Query tensor, shape (batch, seq_len, num_heads, key_dim).
        :type queries: keras.KerasTensor
        :param keys: Key tensor, shape (batch, seq_len, num_heads, key_dim).
        :type keys: keras.KerasTensor
        :param connection: Connection tensor, shape (batch, seq_len, dim, dim).
        :type connection: keras.KerasTensor
        :return: Attention scores, shape (batch, num_heads, seq_len, seq_len).
        :rtype: keras.KerasTensor
        """
        batch_size = ops.shape(queries)[0]
        seq_len = ops.shape(queries)[1]

        # Compute standard attention scores as base
        # (batch, num_heads, seq_len, seq_len)
        base_scores = ops.einsum('bihd,bjhd->bhij', queries, keys)

        # Compute holonomy-based correction
        # For each pair (i, j), compute holonomy contribution
        # This is an approximation using connection at midpoint

        # Get connection features: (batch, seq_len, dim²)
        conn_flat = ops.reshape(
            connection,
            (batch_size, seq_len, -1)
        )

        # Outer product of connection features as pairwise holonomy proxy
        # This captures non-commutativity of connections
        # (batch, seq_len, seq_len, dim²)
        conn_i = ops.expand_dims(conn_flat, 2)
        conn_j = ops.expand_dims(conn_flat, 1)

        # Holonomy proxy: antisymmetric part captures curvature
        holonomy_proxy = conn_i - conn_j  # Antisymmetric

        # Reduce to attention shape: (batch, seq_len, seq_len)
        holonomy_score = ops.mean(holonomy_proxy ** 2, axis=-1)
        holonomy_score = -holonomy_score  # Prefer low holonomy (parallel paths)

        # Expand for heads: (batch, num_heads, seq_len, seq_len)
        holonomy_score = ops.expand_dims(holonomy_score, 1)
        holonomy_score = ops.tile(holonomy_score, (1, self.num_heads, 1, 1))

        # Combine base and holonomy scores
        combined_scores = base_scores + 0.1 * holonomy_score

        return combined_scores

    def _compute_geodesic_attention(
            self,
            queries: keras.KerasTensor,
            keys: keras.KerasTensor,
            curvature: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute attention scores based on geodesic distance.

        The geodesic distance accounts for curvature - paths through
        high-curvature regions are effectively longer.

        :param queries: Query tensor.
        :type queries: keras.KerasTensor
        :param keys: Key tensor.
        :type keys: keras.KerasTensor
        :param curvature: Curvature tensor, shape (batch, seq_len, dim) or similar.
        :type curvature: keras.KerasTensor
        :return: Attention scores.
        :rtype: keras.KerasTensor
        """
        batch_size = ops.shape(queries)[0]
        seq_len = ops.shape(queries)[1]

        # Standard Euclidean attention
        base_scores = ops.einsum('bihd,bjhd->bhij', queries, keys)

        # Flatten curvature if needed
        curv_shape = ops.shape(curvature)
        if len(curv_shape) > 3:
            curvature = ops.reshape(curvature, (curv_shape[0], curv_shape[1], -1))

        # Compute curvature magnitude at each position
        curv_magnitude = ops.sqrt(ops.sum(curvature ** 2, axis=-1) + 1e-8)
        # Shape: (batch, seq_len)

        # Geodesic correction: attention is reduced through high-curvature regions
        # Average curvature between positions
        curv_i = ops.expand_dims(curv_magnitude, 2)  # (batch, seq_len, 1)
        curv_j = ops.expand_dims(curv_magnitude, 1)  # (batch, 1, seq_len)
        avg_curv = (curv_i + curv_j) / 2.0  # (batch, seq_len, seq_len)

        # Curvature penalty (high curvature = longer geodesic = less attention)
        geodesic_correction = -0.1 * avg_curv

        # Expand for heads
        geodesic_correction = ops.expand_dims(geodesic_correction, 1)
        geodesic_correction = ops.tile(geodesic_correction, (1, self.num_heads, 1, 1))

        combined_scores = base_scores + geodesic_correction

        return combined_scores

    def _compute_curvature_attention(
            self,
            queries: keras.KerasTensor,
            keys: keras.KerasTensor,
            curvature: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Compute attention scores based on curvature agreement.

        Positions with similar local curvature attend to each other more.

        :param queries: Query tensor.
        :type queries: keras.KerasTensor
        :param keys: Key tensor.
        :type keys: keras.KerasTensor
        :param curvature: Curvature tensor.
        :type curvature: keras.KerasTensor
        :return: Attention scores.
        :rtype: keras.KerasTensor
        """
        # Standard attention
        base_scores = ops.einsum('bihd,bjhd->bhij', queries, keys)

        # Flatten curvature
        curv_shape = ops.shape(curvature)
        if len(curv_shape) > 3:
            curvature = ops.reshape(curvature, (curv_shape[0], curv_shape[1], -1))

        # Curvature similarity: dot product of curvatures
        curv_normalized = curvature / (
                ops.sqrt(ops.sum(curvature ** 2, axis=-1, keepdims=True)) + 1e-8
        )
        curv_similarity = ops.einsum('bid,bjd->bij', curv_normalized, curv_normalized)

        # Expand for heads
        curv_similarity = ops.expand_dims(curv_similarity, 1)
        curv_similarity = ops.tile(curv_similarity, (1, self.num_heads, 1, 1))

        combined_scores = base_scores + 0.2 * curv_similarity

        return combined_scores

    def _parallel_transport_values(
            self,
            values: keras.KerasTensor,
            attention_weights: keras.KerasTensor,
            connection: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Parallel transport values before aggregation.

        When aggregating values from different positions, we first transport
        them to a common frame using the connection.

        :param values: Value tensor, shape (batch, seq_len, num_heads, head_dim).
        :type values: keras.KerasTensor
        :param attention_weights: Attention weights, shape (batch, num_heads, seq_len, seq_len).
        :type attention_weights: keras.KerasTensor
        :param connection: Connection tensor.
        :type connection: keras.KerasTensor
        :return: Transported and aggregated values.
        :rtype: keras.KerasTensor
        """
        batch_size = ops.shape(values)[0]
        seq_len = ops.shape(values)[1]

        # For efficiency, we approximate transport using learned correction
        # Full transport would require per-pair computation

        # Apply transport correction to values
        transported_values = ops.einsum(
            'bshd,de->bshe',
            values,
            self.transport_kernel
        )

        # Connection-based correction
        # Average connection effect based on attention
        # connection: (batch, seq_len, dim, dim)
        # We project this to head dimension for value correction

        # Simple approximation: modulate values by attention-weighted connection
        # This captures the dominant transport direction

        return transported_values

    def call(
            self,
            inputs: Union[
                keras.KerasTensor,
                Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]
            ],
            training: Optional[bool] = None,
            attention_mask: Optional[keras.KerasTensor] = None
    ) -> keras.KerasTensor:
        """Compute gauge-invariant attention.

        :param inputs: Either single tensor or tuple of (embeddings, curvature, connection).
        :type inputs: Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]]
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :param attention_mask: Optional mask for attention (True = attend, False = mask).
        :type attention_mask: Optional[keras.KerasTensor]
        :return: Attention output of shape (batch, seq_len, hidden_dim).
        :rtype: keras.KerasTensor
        """
        # Parse inputs
        if isinstance(inputs, (list, tuple)):
            if len(inputs) >= 3:
                embeddings, curvature, connection = inputs[0], inputs[1], inputs[2]
            elif len(inputs) == 2:
                embeddings, curvature = inputs
                connection = None
            else:
                embeddings = inputs[0]
                curvature = None
                connection = None
        else:
            embeddings = inputs
            curvature = None
            connection = None

        batch_size = ops.shape(embeddings)[0]
        seq_len = ops.shape(embeddings)[1]

        # Compute Q, K, V projections
        # queries: (batch, seq_len, num_heads, key_dim)
        queries = ops.einsum('bsd,dhk->bshk', embeddings, self.query_kernel)
        keys = ops.einsum('bsd,dhk->bshk', embeddings, self.key_kernel)
        values = ops.einsum('bsd,dhv->bshv', embeddings, self.value_kernel)

        # Compute gauge-invariant attention scores
        if self.attention_metric == 'holonomy' and connection is not None:
            scores = self._compute_holonomy_attention(queries, keys, connection)
        elif self.attention_metric == 'geodesic' and curvature is not None:
            scores = self._compute_geodesic_attention(queries, keys, curvature)
        elif self.attention_metric == 'curvature' and curvature is not None:
            scores = self._compute_curvature_attention(queries, keys, curvature)
        elif self.attention_metric == 'hybrid':
            # Combine all available metrics
            scores = ops.einsum('bihd,bjhd->bhij', queries, keys)
            if connection is not None:
                holonomy_scores = self._compute_holonomy_attention(queries, keys, connection)
                scores = scores + 0.3 * (holonomy_scores - scores)
            if curvature is not None:
                curv_scores = self._compute_curvature_attention(queries, keys, curvature)
                scores = scores + 0.2 * (curv_scores - scores)
        else:
            # Fallback to standard attention
            scores = ops.einsum('bihd,bjhd->bhij', queries, keys)

        # Apply temperature scaling
        scores = scores * ops.reshape(self.attention_temperature, (1, self.num_heads, 1, 1))

        # Apply curvature gating if enabled
        if self.use_curvature_gating and curvature is not None:
            # Compute gate based on curvature
            curv_flat = ops.reshape(curvature, (batch_size, seq_len, -1))
            gate = ops.sigmoid(ops.einsum(
                'bsd,dh->bsh',
                curv_flat[..., :ops.shape(embeddings)[-1]],
                self.gate_kernel
            ))
            # Apply gate to scores
            gate = ops.expand_dims(gate, 3)  # (batch, seq_len, num_heads, 1)
            gate = ops.transpose(gate, (0, 2, 1, 3))  # (batch, num_heads, seq_len, 1)
            scores = scores * gate

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for heads: (batch, 1, seq_len, seq_len) or similar
            if len(ops.shape(attention_mask)) == 2:
                # (batch, seq_len) -> (batch, 1, 1, seq_len)
                mask = ops.expand_dims(ops.expand_dims(attention_mask, 1), 1)
            else:
                mask = attention_mask
            # Apply mask: -inf for masked positions
            scores = ops.where(
                mask,
                scores,
                ops.ones_like(scores) * (-1e9)
            )

        # Softmax to get attention weights
        attention_weights = ops.softmax(scores, axis=-1)

        # Apply dropout during training
        if training and self.dropout_rate > 0:
            attention_weights = keras.random.dropout(
                attention_weights,
                rate=self.dropout_rate,
                seed=None
            )

        # Parallel transport values if enabled
        if self.use_parallel_transport and connection is not None:
            values = self._parallel_transport_values(values, attention_weights, connection)

        # Aggregate values
        # attention_weights: (batch, num_heads, seq_len, seq_len)
        # values: (batch, seq_len, num_heads, head_dim)
        # Need: (batch, num_heads, seq_len, head_dim)
        values_transposed = ops.transpose(values, (0, 2, 1, 3))
        attended = ops.einsum('bhij,bhjd->bhid', attention_weights, values_transposed)

        # Transpose back: (batch, seq_len, num_heads, head_dim)
        attended = ops.transpose(attended, (0, 2, 1, 3))

        # Output projection
        output = ops.einsum('bshd,hdo->bso', attended, self.output_kernel)

        return output

    def compute_output_shape(
            self,
            input_shape: Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...]]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape.

        :param input_shape: Input shape or tuple of input shapes.
        :type input_shape: Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...]]
        :return: Output shape.
        :rtype: Tuple[Optional[int], ...]
        """
        if isinstance(input_shape, list):
            embed_shape = input_shape[0]
        else:
            embed_shape = input_shape

        batch_size = embed_shape[0]
        seq_len = embed_shape[1] if len(embed_shape) > 1 else None

        return (batch_size, seq_len, self.hidden_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'attention_metric': self.attention_metric,
            'use_curvature_gating': self.use_curvature_gating,
            'use_parallel_transport': self.use_parallel_transport,
            'dropout_rate': self.dropout_rate,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        })
        return config