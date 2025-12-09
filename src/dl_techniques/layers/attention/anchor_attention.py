"""
A hierarchical, memory-efficient anchor-based attention layer.

This layer provides a scalable alternative to standard self-attention by
creating an information bottleneck through a small, fixed set of "anchor"
tokens. It is designed to reduce the quadratic complexity of attention for
long sequences while preserving the model's ability to access global context.

Architecture Overview::

    Standard Self-Attention (O(N²)):
    ┌─────────────────────────────────────┐
    │  Token₁ ←→ Token₂ ←→ Token₃ ←→ ...  │  All tokens attend to all others
    │    ↑↓       ↑↓       ↑↓             │  (quadratic in sequence length)
    │  Token₄ ←→ Token₅ ←→ Token₆ ←→ ...  │
    └─────────────────────────────────────┘

    Anchor Attention (O(K² + (N-K)·K)):
    ┌─────────────────────────────────────────────────────────────┐
    │                    ANCHOR TOKENS (K)                        │
    │              ┌───────────────────────┐                      │
    │              │  A₁ ←→ A₂ ←→ ... ←→ Aₖ │  Full self-attention│
    │              └───────────────────────┘  among anchors       │
    │                  ↑   ↑   ↑   ↑   ↑                          │
    │    ┌─────────────┼───┼───┼───┼───┼──────────────────┐       │
    │    │             │   │   │   │   │                  │       │
    │    Q₁           Q₂  Q₃  Q₄  ...  Qₘ                 │       │
    │    QUERY TOKENS (N-K): Cross-attend to anchors only │       │
    └─────────────────────────────────────────────────────────────┘

The architecture transforms the standard all-to-all attention graph
into a two-tier, hub-and-spoke model:

1.  **Anchor Tokens**: A small subset of tokens (e.g., the first K
    tokens) that perform full, quadratic self-attention among themselves.
    These tokens aggregate information and form a compressed global summary.

2.  **Query Tokens**: The remaining tokens. To save computation, these
    tokens do not attend to each other. Instead, they perform cross-attention
    only to the anchor tokens, reading from the global summary.

References:
    - Beltagy, I., et al. (2020). "Longformer: The Long-Document Transformer".
    - Lee, J., et al. (2019). "Set Transformer".
"""

import keras
import numpy as np
from keras import ops, layers, initializers, regularizers
from typing import Optional, Any, Dict, Tuple, Union

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from ..activations import ProbabilityOutput

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class AnchorAttention(keras.layers.Layer):
    """
    Hierarchical attention mechanism with anchor-based information bottleneck.

    This layer implements a memory-efficient attention mechanism that reduces
    computational complexity for long sequences while maintaining representational
    power through a hierarchical two-tier structure.

    **Intent**: Provide a scalable attention alternative that reduces the O(N²)
    complexity of standard self-attention to approximately O(K² + N·K) where K
    is a small fixed number of anchor tokens, enabling efficient processing of
    long sequences.

    **Architecture**::

        ┌──────────────────────────────────────────────────────────────────────┐
        │                         STANDARD MODE                                │
        │                    (num_anchor_tokens=None)                          │
        │                                                                      │
        │    Input(batch, seq, dim)                                            │
        │           │                                                          │
        │           ├──────────────┬──────────────┐                            │
        │           ↓              ↓              ↓                            │
        │      ┌────────┐    ┌────────┐    ┌────────┐                          │
        │      │ Q Proj │    │ K Proj │    │ V Proj │                          │
        │      └────┬───┘    └────┬───┘    └────┬───┘                          │
        │           │             │             │                              │
        │           ↓             ↓             ↓                              │
        │    (batch, heads, seq, head_dim)     (batch, heads, seq, head_dim)   │
        │           │             │             │                              │
        │           └──────┬──────┘             │                              │
        │                  ↓                    │                              │
        │         ┌────────────────┐            │                              │
        │         │  Scaled Dot-   │            │                              │
        │         │  Product Attn  │            │                              │
        │         │  Q @ K^T / √d  │            │                              │
        │         └───────┬────────┘            │                              │
        │                 ↓                     │                              │
        │         ┌────────────────┐            │                              │
        │         │  Probability   │            │                              │
        │         │  Activation    │◄───────────┤                              │
        │         └───────┬────────┘            │                              │
        │                 ↓                     │                              │
        │         ┌────────────────┐            │                              │
        │         │    Dropout     │            │                              │
        │         └───────┬────────┘            │                              │
        │                 ↓                     ↓                              │
        │         ┌────────────────────────────────┐                           │
        │         │      Attn @ V                  │                           │
        │         └───────────────┬────────────────┘                           │
        │                         ↓                                            │
        │                  ┌────────────┐                                      │
        │                  │ Output Proj│                                      │
        │                  └─────┬──────┘                                      │
        │                        ↓                                             │
        │               Output(batch, seq, dim)                                │
        └──────────────────────────────────────────────────────────────────────┘

        ┌──────────────────────────────────────────────────────────────────────┐
        │                       HIERARCHICAL MODE                              │
        │                   (num_anchor_tokens=K > 0)                          │
        │                                                                      │
        │    Input(batch, seq, dim)                                            │
        │           │                                                          │
        │           ├─────────────────────────────┐                            │
        │           ↓                             ↓                            │
        │    Anchor Tokens[:K]              Query Tokens[K:]                   │
        │           │                             │                            │
        │     ┌─────┴─────┐                       │                            │
        │     ↓     ↓     ↓                       ↓                            │
        │   Q_anc K_anc V_anc               Q_query (separate proj)            │
        │     │     │     │                       │                            │
        │     │     │     │                       │                            │
        │     ↓     ↓     │                       │                            │
        │  ┌──────────┐   │                       │                            │
        │  │Reshape to│   │                       │                            │
        │  │multi-head│   │                       │                            │
        │  └────┬─────┘   │                       │                            │
        │       │         │                       │                            │
        │       ↓         ↓                       ↓                            │
        │    Combined Q = [Q_anc; Q_query]                                     │
        │    (batch, heads, seq, head_dim)                                     │
        │           │                                                          │
        │           ↓                                                          │
        │    ┌─────────────────────────────────────┐                           │
        │    │  All tokens attend ONLY to anchors  │                           │
        │    │     scores = Q_all @ K_anc^T / √d   │                           │
        │    │     (batch, heads, seq, K)          │                           │
        │    └────────────────┬────────────────────┘                           │
        │                     ↓                                                │
        │           Probability Activation                                     │
        │                     ↓                                                │
        │                  Dropout                                             │
        │                     ↓                                                │
        │             ┌───────────────┐                                        │
        │             │ Attn @ V_anc  │                                        │
        │             └───────┬───────┘                                        │
        │                     ↓                                                │
        │              Output Projection                                       │
        │                     ↓                                                │
        │            Output(batch, seq, dim)                                   │
        └──────────────────────────────────────────────────────────────────────┘

    **Mathematical Operations**:

    Standard Mode::

        Q = X @ W_q,  K = X @ W_k,  V = X @ W_v
        Attention(Q, K, V) = Probability(Q @ K^T / √d_k) @ V
        Output = Attention @ W_o

    Hierarchical Mode::

        X_anchor = X[:, :K, :],  X_query = X[:, K:, :]
        Q_anchor = X_anchor @ W_q,  K_anchor = X_anchor @ W_k,  V_anchor = X_anchor @ W_v
        Q_query = X_query @ W_q_token
        Q_combined = Concat([Q_anchor, Q_query], axis=seq)
        scores = Q_combined @ K_anchor^T / √d_k       # (batch, heads, N, K)
        Attn = Probability(scores) @ V_anchor
        Output = Attn @ W_o

    **Complexity Analysis**::

        Standard Self-Attention: O(N² · d)
        Anchor Attention:        O(K² · d + (N-K) · K · d) ≈ O(N · K · d)  when K << N

        For K=32 and N=4096: ~128x reduction in attention computation

    Args:
        dim: Integer, input/output dimension of the attention layer.
            Must be positive and divisible by num_heads.
        num_heads: Integer, number of attention heads.
            Must be positive and divide dim evenly.
        head_dim: Optional integer, dimension of each attention head.
            If None, computed as dim // num_heads. Defaults to None.
        dropout_rate: Float, dropout rate applied to attention weights.
            Must be in range [0.0, 1.0]. Defaults to 0.0.
        use_bias: Boolean, whether to use bias in linear projections.
            Defaults to True.
        probability_type: String, type of probability function for attention
            scores (e.g., 'softmax', 'sparsemax', 'adaptive').
            Defaults to 'softmax'.
        probability_config: Optional dictionary containing configuration for
            the probability layer. Defaults to None.
        kernel_initializer: String or Initializer instance for kernel weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: String or Initializer instance for bias vectors.
            Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for kernel weights.
            Defaults to None.
        bias_regularizer: Optional regularizer for bias weights.
            Defaults to None.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        3D tensor with shape: ``(batch_size, sequence_length, dim)``.

    Output shape:
        3D tensor with shape: ``(batch_size, sequence_length, dim)``.
        Shape is preserved through the attention operation.

    Attributes:
        dim: Output dimensionality.
        num_heads: Number of attention heads.
        head_dim: Dimension per attention head.
        scale: Scaling factor (1/√head_dim) for attention scores.
        query_proj: Dense projection for queries (anchor tokens and standard mode).
        key_proj: Dense projection for keys.
        value_proj: Dense projection for values.
        query_token_proj: Separate Dense projection for query tokens in
            hierarchical mode.
        output_proj: Dense projection for output.
        score_activation: ProbabilityOutput layer for attention normalization.
        dropout_layer: Optional Dropout layer for attention weights.

    Example:
        >>> # Standard self-attention
        >>> layer = AnchorAttention(dim=256, num_heads=8)
        >>> x = keras.random.normal((2, 100, 256))
        >>> output = layer(x)  # (2, 100, 256)
        >>>
        >>> # Hierarchical anchor attention with 16 anchors
        >>> output = layer(x, num_anchor_tokens=16)  # (2, 100, 256)
        >>>
        >>> # With custom probability function and regularization
        >>> layer = AnchorAttention(
        ...     dim=512,
        ...     num_heads=8,
        ...     dropout_rate=0.1,
        ...     probability_type='sparsemax',
        ...     kernel_regularizer=keras.regularizers.L2(1e-4)
        ... )

    Note:
        - When ``num_anchor_tokens`` is None or >= sequence_length, standard
          self-attention is applied.
        - Query tokens use a separate projection (``query_token_proj``) in
          hierarchical mode to allow differentiated representations.
        - The layer is designed for scenarios where global context is important
          but quadratic attention is prohibitive (e.g., long documents, genomics).

    References:
        - Beltagy et al. (2020). "Longformer: The Long-Document Transformer"
        - Lee et al. (2019). "Set Transformer"
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            head_dim: Optional[int] = None,
            dropout_rate: float = 0.0,
            use_bias: bool = True,
            probability_type: str = "softmax",
            probability_config: Optional[Dict[str, Any]] = None,
            kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        """
        Initialize the AnchorAttention layer.

        Creates all sub-layers and validates configuration parameters.
        """
        super().__init__(**kwargs)

        # ---------------------------------------------------------------------
        # Parameter validation
        # ---------------------------------------------------------------------
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(
                f"dropout_rate must be between 0 and 1, got {dropout_rate}"
            )

        # ---------------------------------------------------------------------
        # Store configuration
        # ---------------------------------------------------------------------
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.probability_type = probability_type
        self.probability_config = probability_config
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Scaling factor: 1/√d_k
        self.scale = 1.0 / np.sqrt(float(self.head_dim))

        # ---------------------------------------------------------------------
        # Create sub-layers
        # ---------------------------------------------------------------------
        common_kwargs = {
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
        }

        # Projections for anchor tokens (and standard mode)
        self.query_proj = keras.layers.Dense(
            self.num_heads * self.head_dim,
            name="query_proj",
            **common_kwargs
        )
        self.key_proj = keras.layers.Dense(
            self.num_heads * self.head_dim,
            name="key_proj",
            **common_kwargs
        )
        self.value_proj = keras.layers.Dense(
            self.num_heads * self.head_dim,
            name="value_proj",
            **common_kwargs
        )

        # Separate query projection for query tokens (hierarchical mode)
        self.query_token_proj = keras.layers.Dense(
            self.dim,
            name="query_token_proj",
            **common_kwargs
        )

        # Output projection
        self.output_proj = keras.layers.Dense(
            self.dim,
            name="output_proj",
            **common_kwargs
        )

        # Probability activation (softmax/sparsemax/etc.)
        self.score_activation = ProbabilityOutput(
            probability_type=self.probability_type,
            type_config=self.probability_config,
            name="score_activation"
        )

        # Dropout layer (optional)
        if self.dropout_rate > 0.0:
            self.dropout_layer = keras.layers.Dropout(
                self.dropout_rate,
                name="dropout"
            )
        else:
            self.dropout_layer = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all sub-layers.

        Validates input shape and explicitly builds all sub-layers to ensure
        proper weight initialization for serialization.

        Args:
            input_shape: Shape tuple of the input tensor.
                Expected format: ``(batch_size, sequence_length, dim)``.

        Raises:
            ValueError: If input is not 3D or last dimension doesn't match dim.
        """
        if len(input_shape) != 3:
            raise ValueError(f"Input must be 3D, got shape {input_shape}")

        if input_shape[-1] is not None and input_shape[-1] != self.dim:
            raise ValueError(
                f"Last dimension of input ({input_shape[-1]}) "
                f"must match dim ({self.dim})"
            )

        # Build projection layers
        self.query_proj.build(input_shape)
        self.key_proj.build(input_shape)
        self.value_proj.build(input_shape)
        self.query_token_proj.build(input_shape)
        self.output_proj.build(input_shape)

        # Build probability layer with representative attention score shape
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        score_shape = (batch_size, self.num_heads, seq_len, seq_len)
        self.score_activation.build(score_shape)

        # Build dropout layer if present
        if self.dropout_layer is not None:
            self.dropout_layer.build(score_shape)

        super().build(input_shape)

    def call(
            self,
            x: keras.KerasTensor,
            num_anchor_tokens: Optional[int] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply anchor-based attention to input tensor.

        Routes to either standard self-attention or hierarchical anchor attention
        based on the ``num_anchor_tokens`` parameter.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, dim)``.
            num_anchor_tokens: Optional integer specifying number of anchor tokens
                from the beginning of the sequence. If None, applies standard
                self-attention to all tokens. Defaults to None.
            training: Boolean indicating training mode for dropout.
                Defaults to None.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, dim)``.
        """
        if num_anchor_tokens is None:
            return self._standard_attention(x, training)
        else:
            return self._hierarchical_attention(x, num_anchor_tokens, training)

    def _standard_attention(
            self,
            x: keras.KerasTensor,
            training: Optional[bool]
    ) -> keras.KerasTensor:
        """
        Apply standard multi-head self-attention to all tokens.

        All tokens attend to all other tokens with O(N²) complexity.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, dim)``.
            training: Boolean indicating training mode.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, dim)``.
        """
        batch_size = ops.shape(x)[0]
        seq_len = ops.shape(x)[1]

        # Linear projections: (batch, seq, num_heads * head_dim)
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        # Reshape to multi-head format: (batch, seq, num_heads, head_dim)
        q = ops.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = ops.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = ops.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))

        # Transpose to: (batch, num_heads, seq, head_dim)
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        # Scaled dot-product attention scores: (batch, heads, seq, seq)
        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale

        # Apply probability function (softmax/sparsemax/etc.)
        attn_weights = self.score_activation(scores)

        # Apply dropout during training
        if self.dropout_layer is not None:
            attn_weights = self.dropout_layer(attn_weights, training=training)

        # Compute attention output: (batch, heads, seq, head_dim)
        out = ops.matmul(attn_weights, v)

        # Reshape back: (batch, seq, heads, head_dim) -> (batch, seq, dim)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (batch_size, seq_len, self.dim))

        return self.output_proj(out)

    def _hierarchical_attention(
            self,
            x: keras.KerasTensor,
            num_anchor_tokens: int,
            training: Optional[bool]
    ) -> keras.KerasTensor:
        """
        Apply hierarchical anchor-query attention pattern.

        Anchor tokens perform full self-attention among themselves, while
        query tokens only cross-attend to the anchors.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, dim)``.
            num_anchor_tokens: Number of anchor tokens (first K tokens).
            training: Boolean indicating training mode.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, dim)``.
        """
        batch_size = ops.shape(x)[0]
        seq_len = ops.shape(x)[1]

        # Fallback to standard attention if all tokens are anchors
        if num_anchor_tokens >= seq_len:
            return self._standard_attention(x, training)

        # Split input into anchor and query tokens
        anchor_tokens = x[:, :num_anchor_tokens, :]
        query_tokens = x[:, num_anchor_tokens:, :]
        num_query_tokens = seq_len - num_anchor_tokens

        # -----------------------------------------------------------------
        # Process anchor tokens (full Q, K, V projections)
        # -----------------------------------------------------------------
        anchor_q = self.query_proj(anchor_tokens)
        anchor_k = self.key_proj(anchor_tokens)
        anchor_v = self.value_proj(anchor_tokens)

        # Reshape anchors to multi-head format
        anchor_q = ops.reshape(
            anchor_q,
            (batch_size, num_anchor_tokens, self.num_heads, self.head_dim)
        )
        anchor_k = ops.reshape(
            anchor_k,
            (batch_size, num_anchor_tokens, self.num_heads, self.head_dim)
        )
        anchor_v = ops.reshape(
            anchor_v,
            (batch_size, num_anchor_tokens, self.num_heads, self.head_dim)
        )

        # Transpose to: (batch, heads, num_anchors, head_dim)
        anchor_q = ops.transpose(anchor_q, (0, 2, 1, 3))
        anchor_k = ops.transpose(anchor_k, (0, 2, 1, 3))
        anchor_v = ops.transpose(anchor_v, (0, 2, 1, 3))

        # -----------------------------------------------------------------
        # Process query tokens (only Q projection, using separate weights)
        # -----------------------------------------------------------------
        query_q = self.query_token_proj(query_tokens)
        query_q = ops.reshape(
            query_q,
            (batch_size, num_query_tokens, self.num_heads, self.head_dim)
        )
        query_q = ops.transpose(query_q, (0, 2, 1, 3))

        # -----------------------------------------------------------------
        # Combine queries: [anchor queries; query queries]
        # Shape: (batch, heads, seq_len, head_dim)
        # -----------------------------------------------------------------
        combined_q = ops.concatenate([anchor_q, query_q], axis=2)

        # -----------------------------------------------------------------
        # Attention: ALL tokens attend ONLY to anchor tokens
        # scores shape: (batch, heads, seq_len, num_anchors)
        # -----------------------------------------------------------------
        scores = ops.matmul(
            combined_q,
            ops.transpose(anchor_k, (0, 1, 3, 2))
        ) * self.scale

        # Apply probability function
        attn_weights = self.score_activation(scores)

        # Apply dropout during training
        if self.dropout_layer is not None:
            attn_weights = self.dropout_layer(attn_weights, training=training)

        # Compute attention output using anchor values only
        # attn_weights: (batch, heads, seq_len, num_anchors)
        # anchor_v: (batch, heads, num_anchors, head_dim)
        # out: (batch, heads, seq_len, head_dim)
        out = ops.matmul(attn_weights, anchor_v)

        # Reshape back: (batch, seq, heads, head_dim) -> (batch, seq, dim)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (batch_size, seq_len, self.dim))

        return self.output_proj(out)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Shape tuple identical to input_shape (shape-preserving layer).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of the layer for serialization.

        Returns:
            Dictionary containing all constructor arguments needed to
            recreate the layer.
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "probability_type": self.probability_type,
            "probability_config": self.probability_config,
            "kernel_initializer": keras.initializers.serialize(
                self.kernel_initializer
            ),
            "bias_initializer": keras.initializers.serialize(
                self.bias_initializer
            ),
            "kernel_regularizer": keras.regularizers.serialize(
                self.kernel_regularizer
            ),
            "bias_regularizer": keras.regularizers.serialize(
                self.bias_regularizer
            ),
        })
        return config

# ---------------------------------------------------------------------
