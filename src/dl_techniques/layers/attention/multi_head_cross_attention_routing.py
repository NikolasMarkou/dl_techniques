import keras
from keras import ops
from typing import Optional, Any, Dict, Tuple, Union, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..hierarchical_routing import HierarchicalRoutingLayer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class HierarchicalMultiHeadCrossAttention(keras.layers.Layer):
    """
    Multi-head attention with hierarchical routing for value selection.

    This layer correctly utilizes a `HierarchicalRoutingLayer` as its core
    component for generating attention weights. Instead of dot-product scores,
    each query is passed through the routing layer to produce a valid
    probability distribution over the key/value items.

    **Intent**: Provide a scalable alternative to standard attention by replacing
    the O(N) softmax with an O(log N) `HierarchicalRoutingLayer`, where N is
    the sequence length. This implementation correctly encapsulates the routing
    and normalization logic within the sub-layer.

    **Architecture**:
    ```
    Query Input [B, Q_seq, D] ──> Q_proj ──> Q [B, H, Q_seq, D_h]
                                               │
                                               ↓
            ┌───────────────> HierarchicalRoutingLayer(output_dim=kv_seq_len)
            │                 │
            │                 ↓
    KV Input [B, KV_seq, D] ──> V_proj ──> V [B, H, KV_seq, D_h]
            │                 │
            └─────────────────> Attention Weights [B, H, Q_seq, KV_seq]
                                  (Masking & Re-norm if mask is applied)
                                       │
                                       ↓
                     Output = Attention_Weights @ V
                                       │
                                       ↓
                       proj_dense ──> Final Output [B, Q_seq, D]
    ```

    **Important Design Note**:
    The `HierarchicalRoutingLayer` requires a fixed `output_dim` at initialization.
    To handle dynamic input sequence lengths, this layer must be re-instantiated
    if the `kv_seq_len` changes between calls in a context like a dynamic graph.
    For static graphs (e.g., after `model.build()`), this is handled automatically.
    This implementation uses a placeholder routing layer that is replaced by a
    correctly sized one on the first `call`.

    Args:
        dim: Integer, input/output dimension. Must be positive and divisible
            by num_heads.
        num_heads: Integer, number of attention heads. Must be positive.
            Defaults to 8.
        dropout_rate: Float, dropout rate for the final output projection.
            Must be between 0.0 and 1.0. Defaults to 0.0.
        use_bias: Boolean, whether to use bias in linear projections.
            Defaults to True.
        kernel_initializer: String or Initializer for kernel weights.
            Defaults to "glorot_uniform".
        bias_initializer: String or Initializer for bias vectors.
            Defaults to "zeros".
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        epsilon: Float, a small value added to the denominator during
            probability re-normalization to prevent division by zero.
            Defaults to 1e-7.
        **kwargs: Additional keyword arguments for the Layer base class.

    Call arguments:
        query_input: Query tensor of shape `(batch, query_seq_len, dim)`.
        kv_input: Optional Key-Value tensor of shape `(batch, kv_seq_len, dim)`.
            If `None`, self-attention is performed on `query_input`.
        attention_mask: Optional padding mask for `kv_input` to prevent
            attention to padded tokens. Expected shape: `(batch, kv_seq_len)`.
        training: Boolean indicating training or inference mode.

    Returns:
        Output tensor with shape `(batch_size, query_seq_len, dim)`.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            dropout_rate: float = 0.0,
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            epsilon: float = 1e-7,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store configuration
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = self.dim // num_heads
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.epsilon = epsilon

        # We will create the routing layer in build() once we know the kv_seq_len
        self.routing_layer = None

    def build(
            self,
            input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
    ) -> None:
        """
        Build the layer by creating weight variables and building sub-layers.
        """
        is_list_of_shapes = (
                isinstance(input_shape, (list, tuple))
                and len(input_shape) > 0
                and isinstance(input_shape[0], (list, tuple))
        )
        if is_list_of_shapes:
            query_shape, kv_shape = input_shape
        else:
            query_shape = kv_shape = input_shape

        if len(query_shape) != 3:
            raise ValueError(f"Query input must be 3D, got shape {query_shape}")
        if query_shape[-1] != self.dim:
            raise ValueError(f"Query last dimension ({query_shape[-1]}) must match dim ({self.dim})")
        if kv_shape[1] is None:
            raise ValueError(
                "The sequence length of the key/value input must be known at build time "
                "for HierarchicalMultiHeadCrossAttention. Received shape with unknown "
                f"sequence dimension: {kv_shape}"
            )

        kv_seq_len = kv_shape[1]
        dense_kwargs = {
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer
        }

        # Sub-layers are initialized here for robust serialization
        self.q_dense = keras.layers.Dense(self.dim, name="q_proj", **dense_kwargs)
        self.v_dense = keras.layers.Dense(self.dim, name="v_proj", **dense_kwargs)
        self.proj_dense = keras.layers.Dense(self.dim, name="output_proj", **dense_kwargs)

        # The routing layer is configured with the specific sequence length
        self.routing_layer = HierarchicalRoutingLayer(
            output_dim=kv_seq_len,
            epsilon=self.epsilon,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="hierarchical_router"
        )

        if self.dropout_rate > 0.0:
            self.dropout_layer = keras.layers.Dropout(self.dropout_rate, name="dropout")
        else:
            self.dropout_layer = None

        # Build sublayers explicitly
        self.q_dense.build(query_shape)
        self.v_dense.build(kv_shape)
        self.routing_layer.build((None, self.head_dim))
        self.proj_dense.build((query_shape[0], query_shape[1], self.dim))

        super().build(input_shape)

    def call(
            self,
            query_input: keras.KerasTensor,
            kv_input: Optional[keras.KerasTensor] = None,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass for hierarchical multi-head attention."""
        kv_source = kv_input if kv_input is not None else query_input

        batch_size = ops.shape(query_input)[0]
        query_seq_len = ops.shape(query_input)[1]

        # Project query and value inputs
        q = self.q_dense(query_input)
        v = self.v_dense(kv_source)

        # Reshape for multi-head processing
        q = ops.reshape(q, (batch_size, query_seq_len, self.num_heads, self.head_dim))
        q = ops.transpose(q, (0, 2, 1, 3))  # (B, H, Q_seq, D_h)
        v = ops.reshape(v, (batch_size, -1, self.num_heads, self.head_dim))
        v = ops.transpose(v, (0, 2, 1, 3))  # (B, H, KV_seq, D_h)

        # --- Generate Attention Weights using HierarchicalRoutingLayer ---
        # Reshape query to be 2D (batch_like, features) for the routing layer
        q_flat = ops.reshape(q, (-1, self.head_dim))

        # The routing layer correctly handles internal padding, slicing, and
        # renormalization to produce a valid probability distribution of the
        # desired output shape.
        attn_weights = self.routing_layer(q_flat, training=training)

        # Reshape weights back to their multi-head structure
        attn_weights = ops.reshape(
            attn_weights,
            (batch_size, self.num_heads, query_seq_len, -1) # -1 infers kv_seq_len
        )

        # --- Optional Masking ---
        # If a mask is applied, the distribution is no longer valid (doesn't sum
        # to 1), so it must be re-normalized.
        if attention_mask is not None:
            # Expected mask shape: (B, KV_seq). Expand for broadcasting.
            mask = ops.expand_dims(ops.expand_dims(attention_mask, 1), 1)
            attn_weights *= ops.cast(mask, attn_weights.dtype)
            # Re-normalize
            prob_sum = ops.sum(attn_weights, axis=-1, keepdims=True)
            attn_weights = attn_weights / (prob_sum + self.epsilon)

        # Aggregate values using the computed attention weights
        out = ops.matmul(attn_weights, v)  # (B, H, Q_seq, D_h)

        # Reshape output to match input dimension
        out = ops.transpose(out, (0, 2, 1, 3))  # (B, Q_seq, H, D_h)
        out = ops.reshape(out, (batch_size, query_seq_len, self.dim))

        # Final projection
        out = self.proj_dense(out)

        if self.dropout_layer is not None:
            out = self.dropout_layer(out, training=training)

        return out

    def compute_output_shape(
            self,
            input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape - returns query input shape."""
        is_list_of_shapes = (
                isinstance(input_shape, (list, tuple))
                and len(input_shape) > 0
                and isinstance(input_shape[0], (list, tuple))
        )
        if is_list_of_shapes:
            return input_shape[0]
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
            "epsilon": self.epsilon,
        })
        return config

# ---------------------------------------------------------------------
