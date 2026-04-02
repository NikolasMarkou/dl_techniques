"""
Asymmetric cross-attention from the Perceiver architecture.

This layer is a specialized form of cross-attention that serves as the
core building block of the Perceiver and Perceiver IO models. Its primary
function is to create a scalable information bottleneck, enabling a deep
transformer model to process very large and high-dimensional inputs (like
images or audio) without incurring quadratic computational complexity.

The key innovation is decoupling the processing network's depth from the
input data's size via an asymmetric attention mechanism between a small,
fixed-size latent array (queries) and a large, variable-size data array
(keys and values). The resulting attention matrix has shape ``(N, M)``
with complexity ``O(N * M)`` instead of ``O(M^2)``, where ``N << M``.

References:
    - Jaegle, A., et al. (2021). "Perceiver: General Perception with
      Iterative Attention".
    - Jaegle, A., et al. (2021). "Perceiver IO: A General Architecture for
      Structured Inputs & Outputs".
    - Vaswani, A., et al. (2017). "Attention Is All You Need".
"""

import keras
from typing import Optional, Any, Dict, Tuple, Union, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .multi_head_cross_attention import MultiHeadCrossAttention

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PerceiverAttention(keras.layers.Layer):
    """Perceiver-style asymmetric cross-attention with shared projection interface.

    Implements cross-attention where queries and key-value pairs come from different
    sources, following the Perceiver architecture. A small, fixed-size latent array
    forms queries that attend to a large data array providing keys and values:
    ``Q = X_lat W_q``, ``K = X_data W_k``, ``V = X_data W_v``,
    ``Output = softmax(Q K^T / sqrt(d_k)) V``. This layer wraps
    ``MultiHeadCrossAttention`` with ``shared_qk_projections=False`` to provide
    a specialized Perceiver interface with separate projections for maximum
    cross-modal flexibility.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────┐   ┌──────────────────────────┐
        │ Query Input              │   │ KV Input                 │
        │ [B, Q_seq, dim]          │   │ [B, KV_seq, dim]         │
        └────────────┬─────────────┘   └────────────┬─────────────┘
                     │                              │
                     ▼                              ▼
        ┌────────────────────────────────────────────────────────┐
        │           MultiHeadCrossAttention                      │
        │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
        │  │ Q_proj   │  │ K_proj   │  │ V_proj   │              │
        │  │ (query)  │  │ (kv)     │  │ (kv)     │              │
        │  └────┬─────┘  └────┬─────┘  └────┬─────┘              │
        │       ▼             ▼             ▼                    │
        │  ┌─────────────────────────────────────────┐           │
        │  │ Cross-Attention(Q, K, V)                │           │
        │  │ softmax(Q K^T / sqrt(d_k)) V            │           │
        │  └──────────────────┬──────────────────────┘           │
        │                     ▼                                  │
        │  ┌─────────────────────────────────────────┐           │
        │  │ Output Projection                       │           │
        │  └──────────────────┬──────────────────────┘           │
        └─────────────────────┼──────────────────────────────────┘
                              ▼
        ┌─────────────────────────────────────────────┐
        │ Output [B, Q_seq, dim]                      │
        └─────────────────────────────────────────────┘

    :param dim: Input/output dimension. Must be positive and divisible by num_heads.
    :type dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param dropout_rate: Dropout rate for attention weights, between 0.0 and 1.0.
    :type dropout_rate: float
    :param use_bias: Whether to use bias in linear projections.
    :type use_bias: bool
    :param kernel_initializer: Initializer for kernel weights.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for bias vectors.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any

    :raises ValueError: If dim is not divisible by num_heads.
    :raises ValueError: If input shapes are invalid.
    :raises ValueError: If parameters are out of valid ranges.
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

        # Store ALL configuration parameters
        self.dim = dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # CREATE the underlying MultiHeadCrossAttention layer
        # Use shared_qk_projections=False for flexible cross-attention with separate projections
        self.cross_attention = MultiHeadCrossAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,  # Note: parameter name is 'dropout' in MultiHeadCrossAttention
            shared_qk_projections=False,  # Separate projections for cross-attention flexibility
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="cross_attention"
        )

    def build(
            self,
            input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
    ) -> None:
        """Build the layer by creating weight variables and building sub-layers.

        :param input_shape: Shape of input tensor(s). Can be a single shape tuple
            or a list of two shape tuples for query and kv inputs.
        :type input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
        """
        # Handle different input formats
        if isinstance(input_shape, list):
            # Two separate inputs for cross-attention
            if len(input_shape) != 2:
                raise ValueError(f"Expected 2 inputs for cross-attention, got {len(input_shape)}")
            query_shape, kv_shape = input_shape
        else:
            # Single input shape (will be used for both query and kv in self-attention mode)
            query_shape = kv_shape = input_shape

        # Validate shapes
        if len(query_shape) != 3:
            raise ValueError(f"Query input must be 3D, got shape {query_shape}")
        if len(kv_shape) != 3:
            raise ValueError(f"KV input must be 3D, got shape {kv_shape}")
        if query_shape[-1] != self.dim:
            raise ValueError(f"Query last dimension ({query_shape[-1]}) must match dim ({self.dim})")
        if kv_shape[-1] != self.dim:
            raise ValueError(f"KV last dimension ({kv_shape[-1]}) must match dim ({self.dim})")

        # Build the wrapped cross-attention layer explicitly for serialization
        self.cross_attention.build(input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            query_input: keras.KerasTensor,
            kv_input: Optional[keras.KerasTensor] = None,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply Perceiver cross-attention.

        :param query_input: Query tensor of shape ``(batch_size, query_seq_len, dim)``.
        :type query_input: keras.KerasTensor
        :param kv_input: Key-Value tensor of shape ``(batch_size, kv_seq_len, dim)``.
            If ``None``, uses query_input for self-attention mode.
        :type kv_input: Optional[keras.KerasTensor]
        :param attention_mask: Optional attention mask of shape
            ``(batch_size, seq_len, seq_len)`` or ``(batch_size, 1, seq_len, seq_len)``.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Whether in training mode.
        :type training: Optional[bool]

        :return: Output tensor with same shape as query_input.
        :rtype: keras.KerasTensor
        """
        return self.cross_attention(
            query_input=query_input,
            kv_input=kv_input,  # Can be None for self-attention mode
            attention_mask=attention_mask,
            training=training
        )

    def compute_output_shape(
            self,
            input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape (same as query input shape).

        :param input_shape: Input shape(s), either single or list of two shapes.
        :type input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]

        :return: Output shape tuple, same as query input shape.
        :rtype: Tuple[Optional[int], ...]
        """
        if isinstance(input_shape, list):
            return input_shape[0]  # Same as query input shape
        else:
            return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all layer configuration parameters.
        :rtype: Dict[str, Any]
        """
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
        })
        return config

# ---------------------------------------------------------------------
