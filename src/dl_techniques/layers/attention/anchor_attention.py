"""
A hierarchical, memory-efficient anchor-based attention layer.

This layer provides a scalable alternative to standard self-attention by
creating an information bottleneck through a small, fixed set of "anchor"
tokens. It is designed to reduce the quadratic complexity of attention for
long sequences while preserving the model's ability to access global context.

The architecture transforms the standard all-to-all attention graph
into a two-tier, hub-and-spoke model:

1.  **Anchor Tokens**: A small subset of tokens (e.g., the first K
    tokens) that perform full, quadratic self-attention among themselves.
    These tokens aggregate information and form a compressed global summary.

2.  **Query Tokens**: The remaining tokens. To save computation, these
    tokens do not attend to each other. Instead, they perform cross-attention
    only to the anchor tokens, reading from the global summary.

The complexity reduction is: standard self-attention is ``O(N^2 * d)``,
while anchor attention is ``O(K^2 * d + (N-K) * K * d) ~ O(N * K * d)``
when ``K << N``. For K=32 and N=4096, this yields approximately 128x
reduction in attention computation.

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
    power through a hierarchical two-tier structure. It operates in two modes:

    - **Standard Mode** (``num_anchor_tokens=None``): Full self-attention over all
      tokens with ``O(N^2)`` complexity, using configurable probability activation
      (softmax, sparsemax, etc.).
    - **Hierarchical Mode** (``num_anchor_tokens=K > 0``): Anchor tokens perform
      full self-attention among themselves, while query tokens cross-attend only to
      the anchors, achieving ``O(K^2 + N*K)`` complexity.

    The mathematical operations are:

    Standard: ``Q = X @ W_q, K = X @ W_k, V = X @ W_v;
    Output = Probability(Q @ K^T / sqrt(d_k)) @ V @ W_o``

    Hierarchical: ``Q_combined = [Q_anchor; Q_query];
    scores = Q_combined @ K_anchor^T / sqrt(d_k);
    Output = Probability(scores) @ V_anchor @ W_o``

    **Architecture Overview:**

    .. code-block:: text

        Standard Mode (num_anchor_tokens=None):

        ┌─────────────────────────────────────────────────────────┐
        │  Input [B, seq, dim]                                    │
        │         │                                               │
        │         ├──────────────┬──────────────┐                 │
        │         ▼              ▼              ▼                 │
        │    ┌────────┐    ┌────────┐    ┌────────┐               │
        │    │ Q Proj │    │ K Proj │    │ V Proj │               │
        │    └───┬────┘    └───┬────┘    └───┬────┘               │
        │        │             │             │                    │
        │        ▼             ▼             │                    │
        │     scores = Q @ K^T / sqrt(d_k)   │                    │
        │        │                           │                    │
        │        ▼                           │                    │
        │   Probability Activation           │                    │
        │        │                           │                    │
        │        ▼                           ▼                    │
        │     Dropout ──► weights @ V                             │
        │        │                                                │
        │        ▼                                                │
        │   Output Projection                                     │
        │        │                                                │
        │        ▼                                                │
        │  Output [B, seq, dim]                                   │
        └─────────────────────────────────────────────────────────┘

        Hierarchical Mode (num_anchor_tokens=K > 0):

        ┌─────────────────────────────────────────────────────────┐
        │  Input [B, seq, dim]                                    │
        │         │                                               │
        │         ├─────────────────────────────┐                 │
        │         ▼                             ▼                 │
        │  Anchor Tokens [:K]            Query Tokens [K:]        │
        │         │                             │                 │
        │   ┌─────┴─────┐                       │                 │
        │   ▼     ▼     ▼                       ▼                 │
        │  Q_a  K_a   V_a               Q_query (sep. proj)       │
        │   │     │     │                       │                 │
        │   │     │     │                       │                 │
        │   ▼     │     │                       │                 │
        │  Q_combined = [Q_a ; Q_query]         │                 │
        │         │                             │                 │
        │         ▼                             │                 │
        │  scores = Q_combined @ K_a^T / sqrt(d_k)                │
        │         │                                               │
        │         ▼                                               │
        │  Probability Activation                                 │
        │         │                                               │
        │         ▼                                               │
        │      Dropout ──► weights @ V_a                          │
        │         │                                               │
        │         ▼                                               │
        │  Output Projection                                      │
        │         │                                               │
        │         ▼                                               │
        │  Output [B, seq, dim]                                   │
        └─────────────────────────────────────────────────────────┘

    :param dim: Integer, input/output dimension of the attention layer.
        Must be positive and divisible by num_heads.
    :type dim: int
    :param num_heads: Integer, number of attention heads.
        Must be positive and divide dim evenly.
    :type num_heads: int
    :param head_dim: Optional integer, dimension of each attention head.
        If None, computed as ``dim // num_heads``. Defaults to None.
    :type head_dim: Optional[int]
    :param dropout_rate: Float, dropout rate applied to attention weights.
        Must be in range [0.0, 1.0]. Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias: Boolean, whether to use bias in linear projections.
        Defaults to True.
    :type use_bias: bool
    :param probability_type: String, type of probability function for attention
        scores (e.g., 'softmax', 'sparsemax', 'adaptive').
        Defaults to 'softmax'.
    :type probability_type: str
    :param probability_config: Optional dictionary containing configuration for
        the probability layer. Defaults to None.
    :type probability_config: Optional[Dict[str, Any]]
    :param kernel_initializer: String or Initializer instance for kernel weights.
        Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param bias_initializer: String or Initializer instance for bias vectors.
        Defaults to 'zeros'.
    :type bias_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
        Defaults to None.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for bias weights.
        Defaults to None.
    :type bias_regularizer: Optional[regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the Layer base class.
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
        """Initialize the AnchorAttention layer.

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

        # Scaling factor: 1/sqrt(d_k)
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

        :param input_shape: Shape tuple of the input tensor, expected as
            ``(batch_size, sequence_length, dim)``.
        :type input_shape: Tuple[Optional[int], ...]

        :raises ValueError: If input is not 3D or last dimension does not match dim.
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

        :param x: Input tensor of shape ``(batch_size, seq_len, dim)``.
        :type x: keras.KerasTensor
        :param num_anchor_tokens: Optional integer specifying number of anchor tokens
            from the beginning of the sequence. If None, applies standard
            self-attention to all tokens. Defaults to None.
        :type num_anchor_tokens: Optional[int]
        :param training: Boolean indicating training mode for dropout.
            Defaults to None.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(batch_size, seq_len, dim)``.
        :rtype: keras.KerasTensor
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

        All tokens attend to all other tokens with ``O(N^2)`` complexity.

        :param x: Input tensor of shape ``(batch_size, seq_len, dim)``.
        :type x: keras.KerasTensor
        :param training: Boolean indicating training mode.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(batch_size, seq_len, dim)``.
        :rtype: keras.KerasTensor
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

        :param x: Input tensor of shape ``(batch_size, seq_len, dim)``.
        :type x: keras.KerasTensor
        :param num_anchor_tokens: Number of anchor tokens (first K tokens).
        :type num_anchor_tokens: int
        :param training: Boolean indicating training mode.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(batch_size, seq_len, dim)``.
        :rtype: keras.KerasTensor
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
        """Compute output shape, identical to input shape.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Shape tuple identical to input_shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all constructor arguments.
        :rtype: Dict[str, Any]
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
