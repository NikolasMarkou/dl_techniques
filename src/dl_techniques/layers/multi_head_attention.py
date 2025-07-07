import keras
from keras import ops
from typing import Optional, Tuple, Union, Any, Dict

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MultiHeadAttention(keras.layers.Layer):
    """Multi-Head Self Attention mechanism optimized for vision tasks.

    This implementation uses keras.ops for backend compatibility and follows
    the project's serialization patterns.

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

        # Build sublayers
        self.qkv.build(input_shape)
        self.proj.build(input_shape[:-1] + (self.embed_dim,))

        super().build(input_shape)

    def call(
            self,
            x: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim).
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
        attn = ops.softmax(attn, axis=-1)
        attn = self.dropout(attn, training=training)

        # Apply attention to values
        x = ops.matmul(attn, v)  # (batch_size, num_heads, seq_len, head_dim)
        x = ops.transpose(x, (0, 2, 1, 3))  # (batch_size, seq_len, num_heads, head_dim)
        x = ops.reshape(x, (batch_size, seq_len, self.embed_dim))

        return self.proj(x)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape.
        """
        return input_shape

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
