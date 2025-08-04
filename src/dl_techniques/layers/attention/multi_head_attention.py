"""Multi-Head Attention Layer with Mask Support.

This module implements a Multi-Head Self-Attention mechanism optimized for vision 
and sequence modeling tasks. The implementation follows Keras 3.x best practices 
with backend-agnostic operations and comprehensive serialization support.

Key Features:
    - Backend-agnostic implementation using keras.ops
    - Flexible attention masking support (sequence-level, full, per-head)
    - Proper serialization with get_config/build_from_config methods
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
    padding_mask = tf.ones((batch_size, seq_len))
    padding_mask[:, 100:] = 0  # Mask positions 100 and beyond
    output = attn(input_tensor, attention_mask=padding_mask)

    # With causal masking
    seq_len = input_tensor.shape[1]
    causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    causal_mask = tf.expand_dims(causal_mask, 0)  # Add batch dimension
    output = attn(input_tensor, attention_mask=causal_mask)

    # In a model context
    inputs = keras.Input(shape=(seq_len, embed_dim))
    x = MultiHeadAttention(embed_dim=256, num_heads=4)(inputs)
    model = keras.Model(inputs=inputs, outputs=x)
    ```
"""

import keras
from keras import ops
from typing import Optional, Tuple, Union, Any, Dict

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MultiHeadAttention(keras.layers.Layer):
    """Multi-Head Self Attention mechanism optimized for vision tasks.

    This implementation uses keras.ops for backend compatibility and follows
    the project's serialization patterns. Supports optional attention masking.

    Args:
        embed_dim: Dimension of input embeddings.
        num_heads: Number of attention heads.
        dropout_rate: Dropout rate for attention weights.
        kernel_initializer: Initializer for weight matrices.
        kernel_regularizer: Regularizer for weight matrices.
        use_bias: Whether to use bias in dense layers.
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout_rate: float = 0.0,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_bias: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_rate = dropout_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_bias = use_bias

        # Initialize to None, will be created in build()
        self.qkv = None
        self.proj = None
        self.dropout = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer's weights.

        Args:
            input_shape: Shape of the input tensor.
        """
        self._build_input_shape = input_shape

        # Create QKV projection layer
        self.qkv = keras.layers.Dense(
            self.embed_dim * 3,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_bias=self.use_bias,
            name="qkv"
        )

        # Create output projection layer
        self.proj = keras.layers.Dense(
            self.embed_dim,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_bias=self.use_bias,
            name="proj"
        )

        # Create dropout layer
        self.dropout = keras.layers.Dropout(self.dropout_rate)

        # Build sublayers - handle shape conversion properly
        self.qkv.build(input_shape)

        # Convert to list for consistent manipulation
        input_shape_list = list(input_shape)
        proj_shape = tuple(input_shape_list[:-1] + [self.embed_dim])
        self.proj.build(proj_shape)

        super().build(input_shape)

    def call(
            self,
            x: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim).
            attention_mask: Optional attention mask tensor. Can be:
                - Shape (batch_size, seq_len): mask for each sequence position
                - Shape (batch_size, seq_len, seq_len): full attention mask
                - Shape (batch_size, num_heads, seq_len, seq_len): per-head mask
                Values should be 1 for positions to attend to and 0 for masked positions.
            training: Whether in training mode.

        Returns:
            Attention output tensor of shape (batch_size, seq_len, embed_dim).
        """
        batch_size = ops.shape(x)[0]
        seq_len = ops.shape(x)[1]

        # Generate Q, K, V
        qkv = self.qkv(x)  # (batch_size, seq_len, embed_dim * 3)
        qkv = ops.reshape(qkv, (batch_size, seq_len, 3, self.num_heads, self.head_dim))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))  # (3, batch_size, num_heads, seq_len, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (batch_size, num_heads, seq_len, head_dim)

        # Compute attention scores
        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        # attn shape: (batch_size, num_heads, seq_len, seq_len)

        # Apply attention mask if provided
        if attention_mask is not None:
            attn = self._apply_attention_mask(attn, attention_mask)

        attn = ops.softmax(attn, axis=-1)
        attn = self.dropout(attn, training=training)

        # Apply attention to values
        x = ops.matmul(attn, v)  # (batch_size, num_heads, seq_len, head_dim)
        x = ops.transpose(x, (0, 2, 1, 3))  # (batch_size, seq_len, num_heads, head_dim)
        x = ops.reshape(x, (batch_size, seq_len, self.embed_dim))

        return self.proj(x)

    def _apply_attention_mask(
            self,
            attention_scores: keras.KerasTensor,
            attention_mask: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Apply attention mask to attention scores.

        Args:
            attention_scores: Attention scores tensor of shape
                (batch_size, num_heads, seq_len, seq_len).
            attention_mask: Attention mask tensor. Supported shapes:
                - (batch_size, seq_len): mask for each sequence position
                - (batch_size, seq_len, seq_len): full attention mask
                - (batch_size, num_heads, seq_len, seq_len): per-head mask

        Returns:
            Masked attention scores tensor.
        """
        # Convert mask to the same dtype as attention scores
        attention_mask = ops.cast(attention_mask, attention_scores.dtype)

        # Handle different mask shapes based on tensor rank
        mask_ndim = len(attention_mask.shape)

        if mask_ndim == 2:
            # Shape: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
            # This masks the key positions
            attention_mask = ops.expand_dims(attention_mask, axis=1)  # (batch_size, 1, seq_len)
            attention_mask = ops.expand_dims(attention_mask, axis=1)  # (batch_size, 1, 1, seq_len)

        elif mask_ndim == 3:
            # Shape: (batch_size, seq_len, seq_len) -> (batch_size, 1, seq_len, seq_len)
            attention_mask = ops.expand_dims(attention_mask, axis=1)  # (batch_size, 1, seq_len, seq_len)

        elif mask_ndim == 4:
            # Shape: (batch_size, num_heads, seq_len, seq_len) - already correct
            pass
        else:
            raise ValueError(f"Unsupported attention_mask rank: {mask_ndim}. Expected 2, 3, or 4.")

        # Create mask where 0 becomes -inf and 1 stays 0
        mask_value = -1e9  # Large negative value for masking
        mask = (1.0 - attention_mask) * mask_value

        # Apply mask to attention scores
        attention_scores = attention_scores + mask

        return attention_scores

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape.
        """
        # Convert to list for consistent manipulation, then back to tuple
        input_shape_list = list(input_shape)
        return tuple(input_shape_list)

    def get_config(self) -> Dict[str, Any]:
        """Get the layer configuration.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_bias": self.use_bias,
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the build configuration.

        Returns:
            Dictionary containing the build configuration.
        """
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------