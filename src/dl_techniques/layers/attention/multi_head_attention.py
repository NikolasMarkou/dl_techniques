"""Multi-Head Attention Layer with Mask Support.

This module implements a Multi-Head Self-Attention mechanism optimized for vision 
and sequence modeling tasks. The implementation follows Keras 3.x best practices 
with backend-agnostic operations and comprehensive serialization support.

Key Features:
    - Backend-agnostic implementation using keras.ops
    - Flexible attention masking support (sequence-level, full, per-head)
    - Proper serialization with modern Keras 3 patterns
    - Configurable initializers and regularizers
    - Dropout support for attention weights
    - Type-safe implementation with comprehensive documentation

Attention Mask Support:
    The layer supports three different attention mask formats:

    1. Sequence-level mask (batch_size, seq_len):
       - Masks entire positions in the sequence (e.g., padding tokens)
       - Applied to key positions across all attention heads

    2. Full attention mask (batch_size, seq_len, seq_len):
       - Controls which query positions can attend to which key positions
       - Useful for causal attention, bidirectional constraints, etc.

    3. Per-head mask (batch_size, num_heads, seq_len, seq_len):
       - Different mask for each attention head
       - Maximum flexibility for complex attention patterns

Example Usage:
    ```python
    # Basic multi-head attention
    attn = MultiHeadAttention(embed_dim=512, num_heads=8, dropout_rate=0.1)
    output = attn(input_tensor)

    # With sequence-level masking (padding)
    padding_mask = keras.ops.ones((batch_size, seq_len))
    padding_mask = keras.ops.slice_update(padding_mask, [0, 100],
                                          keras.ops.zeros((batch_size, seq_len - 100)))
    output = attn(input_tensor, attention_mask=padding_mask)

    # With causal masking
    seq_len = input_tensor.shape[1]
    causal_mask = keras.ops.tri(seq_len)
    causal_mask = keras.ops.expand_dims(causal_mask, 0)  # Add batch dimension
    output = attn(input_tensor, attention_mask=causal_mask)

    # In a model context
    inputs = keras.Input(shape=(seq_len, embed_dim))
    x = MultiHeadAttention(embed_dim=256, num_heads=4)(inputs)
    model = keras.Model(inputs=inputs, outputs=x)
    ```
"""

import keras
from typing import Optional, Tuple, Union, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .multi_head_cross_attention import MultiHeadCrossAttention

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MultiHeadAttention(keras.layers.Layer):
    """
    Multi-Head Self-Attention mechanism with comprehensive masking support.

    This layer provides a clean interface for self-attention operations by wrapping
    the more general `MultiHeadCrossAttention` layer. It demonstrates the wrapper
    pattern for creating specialized interfaces while maintaining robust serialization
    and leveraging existing, well-tested implementations.

    **Intent**: Provide a streamlined, user-friendly interface specifically for
    self-attention use cases (vision transformers, sequence modeling) while
    internally leveraging the robust `MultiHeadCrossAttention` implementation
    with its comprehensive serialization support and flexible masking capabilities.

    **Architecture**:
    ```
    Input [B, seq, dim]
           ↓
    MultiHeadCrossAttention(shared_qk_projections=True)
           ↓ (self-attention mode)
    Q, K, V = QKV_proj(input)
           ↓
    Attention(Q, K, V) + Masking
           ↓
    Output [B, seq, dim]
    ```

    **Wrapper Pattern Benefits**:
    - **Simplified Interface**: Focused API for self-attention use cases
    - **Robust Implementation**: Leverages battle-tested `MultiHeadCrossAttention`
    - **Consistent Behavior**: Same masking and serialization as cross-attention
    - **Maintenance**: Single source of truth for attention mechanisms

    Args:
        dim: Integer, dimension of input embeddings. Must be positive
            and divisible by num_heads.
        num_heads: Integer, number of attention heads. Must be positive.
            Defaults to 8.
        dropout_rate: Float, dropout rate for attention weights. Must be between
            0.0 and 1.0. Defaults to 0.0.
        kernel_initializer: String or Initializer for weight matrices.
            Defaults to "he_normal".
        kernel_regularizer: Optional regularizer for weight matrices.
        use_bias: Boolean, whether to use bias in dense layers.
            Defaults to False.
        **kwargs: Additional layer arguments.

    Call arguments:
        x: Input tensor of shape (batch_size, seq_len, dim).
        attention_mask: Optional attention mask tensor. Supported shapes:
            - (batch_size, seq_len): mask for each sequence position
            - (batch_size, seq_len, seq_len): full attention mask
            - (batch_size, num_heads, seq_len, seq_len): per-head mask
            Values should be 1 for positions to attend to and 0 for masked positions.
        training: Boolean indicating whether in training mode.

    Returns:
        Attention output tensor of shape (batch_size, seq_len, dim).

    Raises:
        ValueError: If dim is not divisible by num_heads.
        ValueError: If parameters are invalid (negative values, etc.).

    Example:
        ```python
        # Basic usage
        attn = MultiHeadAttention(dim=512, num_heads=8, dropout_rate=0.1)
        output = attn(input_tensor)

        # With padding mask
        padding_mask = keras.ops.ones((batch_size, seq_len))
        masked_positions = keras.ops.zeros((batch_size, 50))
        padding_mask = keras.ops.concatenate([
            padding_mask[:, :-50], masked_positions
        ], axis=1)
        output = attn(input_tensor, attention_mask=padding_mask)

        # In a Transformer block
        inputs = keras.Input(shape=(seq_len, dim))
        attention_output = MultiHeadAttention(
            dim=256, num_heads=4, dropout_rate=0.1
        )(inputs)
        model = keras.Model(inputs=inputs, outputs=attention_output)
        ```
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout_rate: float = 0.0,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_bias: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store ALL configuration parameters
        self.dim = dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_bias = use_bias

        # CREATE the underlying MultiHeadCrossAttention layer
        # Use shared_qk_projections=True for efficient self-attention
        self.cross_attention = MultiHeadCrossAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            shared_qk_projections=True,  # Efficient self-attention mode
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_initializer="zeros",  # Use default bias initializer
            name="cross_attention"
        )

    def build(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> None:
        """
        Build the layer by creating weight variables and building sub-layers.

        CRITICAL: Explicitly build the wrapped MultiHeadCrossAttention for
        robust serialization. This ensures all weight variables exist before
        weight restoration during model loading.
        """
        # Validate input shape
        if isinstance(input_shape, list):
            input_shape = tuple(input_shape)

        if len(input_shape) != 3:
            raise ValueError(f"Input must be 3D (batch, seq_len, dim), got shape {input_shape}")
        if input_shape[-1] != self.dim:
            raise ValueError(f"Input last dimension ({input_shape[-1]}) must match dim ({self.dim})")

        # Build the wrapped cross-attention layer explicitly for serialization
        self.cross_attention.build(tuple(input_shape))

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through self-attention mechanism.

        This method delegates to the underlying MultiHeadCrossAttention layer
        in self-attention mode (kv_input=None).
        """
        return self.cross_attention(
            query_input=inputs,
            kv_input=None,  # Self-attention: kv_input=None
            attention_mask=attention_mask,
            training=training
        )

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape - same as input shape for self-attention."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization - includes ALL constructor parameters."""
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_bias": self.use_bias,
        })
        return config

# ---------------------------------------------------------------------
