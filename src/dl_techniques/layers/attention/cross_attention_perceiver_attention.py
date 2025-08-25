import keras
from keras import ops
from typing import Optional, Any, Dict, Tuple, Union, List


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class PerceiverAttention(keras.layers.Layer):
    """
    Cross-attention mechanism from the Perceiver architecture.

    This layer implements cross-attention where queries and key-value pairs
    come from different sources, following the Perceiver architecture design.
    This enables flexible cross-modal processing, set-to-set transformations,
    and latent space learning by allowing queries from one modality to attend
    to keys and values from another modality.

    The key difference from standard self-attention is that:
    - Queries come from one input (query_input)
    - Keys and Values come from another input (kv_input)
    - This enables flexible cross-modal or cross-domain attention

    The attention mechanism follows the standard scaled dot-product attention:
    Attention(Q, K, V) = softmax(QK^T / √d_k)V

    Where Q comes from query_input and K, V come from kv_input.

    Args:
        dim: Integer, input/output dimension of the attention layer.
            Must be positive and divisible by num_heads.
        num_heads: Integer, number of attention heads. Must be positive
            and divide dim evenly. Defaults to 8.
        dropout: Float, dropout rate for attention weights. Must be between
            0.0 and 1.0. Defaults to 0.0.
        use_bias: Boolean, whether to use bias in linear projections.
            Defaults to True.
        kernel_initializer: String or Initializer, initializer for the kernel weights.
            Defaults to "glorot_uniform".
        bias_initializer: String or Initializer, initializer for the bias vector.
            Defaults to "zeros".
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Two inputs when called with both arguments:
        - query_input: 3D tensor with shape `(batch_size, query_seq_len, dim)`
        - kv_input: 3D tensor with shape `(batch_size, kv_seq_len, dim)`

        Single input when called with one argument (self-attention mode):
        - input: 3D tensor with shape `(batch_size, seq_len, dim)`

    Output shape:
        3D tensor with shape: `(batch_size, query_seq_len, dim)`

    Call arguments:
        query_input: Query tensor of shape (batch_size, query_seq_len, dim).
        kv_input: Optional Key-Value tensor of shape (batch_size, kv_seq_len, dim).
            If None, uses query_input for both (self-attention mode).
        training: Boolean indicating whether the layer should behave in training
            mode (applying dropout) or inference mode.

    Returns:
        Output tensor with same shape as query_input after cross-attention.

    Raises:
        ValueError: If dim is not divisible by num_heads.
        ValueError: If input shapes are invalid.
        ValueError: If parameters are out of valid ranges.

    Example:
        ```python
        # Cross-attention between different modalities
        visual_features = keras.random.normal((2, 196, 256))  # ViT patches
        text_features = keras.random.normal((2, 77, 256))    # Text tokens

        perceiver_attn = PerceiverAttention(dim=256, num_heads=8, dropout=0.1)

        # Text attending to visual features
        text_to_visual = perceiver_attn(text_features, visual_features)
        print(text_to_visual.shape)  # (2, 77, 256)

        # Visual attending to text features
        visual_to_text = perceiver_attn(visual_features, text_features)
        print(visual_to_text.shape)  # (2, 196, 256)

        # Self-attention mode (single input)
        self_attended = perceiver_attn(visual_features)
        print(self_attended.shape)  # (2, 196, 256)

        # With custom regularization
        regularized_attn = PerceiverAttention(
            dim=512,
            num_heads=16,
            dropout=0.2,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )
        ```

    Notes:
        This layer is particularly useful for:
        - Cross-modal attention (vision-language, audio-visual)
        - Latent bottleneck architectures (Perceiver, Perceiver IO)
        - Set-to-set transformations
        - Memory-augmented networks
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            dropout: float = 0.0,
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
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be between 0 and 1, got {dropout}")

        # Store ALL configuration parameters
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout_rate = dropout
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Scale factor for attention scores
        self.scale = 1.0 / ops.sqrt(float(self.head_dim))

        # CREATE all sub-layers in __init__ (they are unbuilt)
        self.q_dense = keras.layers.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="q"
        )

        self.kv_dense = keras.layers.Dense(
            self.dim * 2,  # For both key and value
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="kv"
        )

        self.proj_dense = keras.layers.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="proj"
        )

        # Dropout layer (only if dropout > 0)
        if self.dropout_rate > 0.0:
            self.dropout_layer = keras.layers.Dropout(self.dropout_rate, name="dropout")
        else:
            self.dropout_layer = None

    def build(self, input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]) -> None:
        """
        Build the layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.

        Args:
            input_shape: Shape of input tensor(s). Can be a single shape tuple
                or a list of two shape tuples for query and kv inputs.

        Raises:
            ValueError: If input shapes are invalid.
        """
        # Handle different input formats
        if isinstance(input_shape, list):
            # Two separate inputs
            if len(input_shape) != 2:
                raise ValueError(f"Expected 2 inputs, got {len(input_shape)}")
            query_shape, kv_shape = input_shape
        else:
            # Single input shape (will be used for both query and kv)
            query_shape = kv_shape = input_shape

        # Validate shapes
        if len(query_shape) != 3:
            raise ValueError(f"Query input must be 3D, got shape {query_shape}")
        if len(kv_shape) != 3:
            raise ValueError(f"KV input must be 3D, got shape {kv_shape}")

        if query_shape[-1] != self.dim:
            raise ValueError(f"Query last dimension ({query_shape[-1]}) "
                             f"must match dim ({self.dim})")
        if kv_shape[-1] != self.dim:
            raise ValueError(f"KV last dimension ({kv_shape[-1]}) "
                             f"must match dim ({self.dim})")

        # Build sub-layers in computational order
        # Query projection
        self.q_dense.build(query_shape)

        # Key-Value projection
        self.kv_dense.build(kv_shape)

        # Output projection - input shape after attention computation
        # Shape: (batch_size, query_seq_len, dim)
        proj_input_shape = (query_shape[0], query_shape[1], self.dim)
        self.proj_dense.build(proj_input_shape)

        # Dropout layer (if exists)
        if self.dropout_layer is not None:
            # Dropout can handle any shape, using attention weights shape
            # Shape: (batch_size, num_heads, query_seq_len, kv_seq_len)
            attn_shape = (query_shape[0], self.num_heads, query_shape[1], kv_shape[1])
            self.dropout_layer.build(attn_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            query_input: keras.KerasTensor,
            kv_input: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply Perceiver cross-attention.

        Args:
            query_input: Query tensor of shape (batch_size, query_seq_len, dim).
            kv_input: Key-Value tensor of shape (batch_size, kv_seq_len, dim).
                If None, uses query_input for both (self-attention mode).
            training: Boolean indicating training mode.

        Returns:
            Output tensor with same shape as query_input.
        """
        if kv_input is None:
            kv_input = query_input

        batch_size = ops.shape(query_input)[0]
        query_seq_len = ops.shape(query_input)[1]
        kv_seq_len = ops.shape(kv_input)[1]

        # Project queries
        q = self.q_dense(query_input)  # (batch_size, query_seq_len, dim)
        q = ops.reshape(q, (batch_size, query_seq_len, self.num_heads, self.head_dim))
        q = ops.transpose(q, (0, 2, 1, 3))  # (batch_size, num_heads, query_seq_len, head_dim)

        # Project keys and values
        kv = self.kv_dense(kv_input)  # (batch_size, kv_seq_len, dim * 2)
        kv = ops.reshape(kv, (batch_size, kv_seq_len, 2, self.num_heads, self.head_dim))
        kv = ops.transpose(kv, (2, 0, 3, 1, 4))  # (2, batch_size, num_heads, kv_seq_len, head_dim)

        k, v = kv[0], kv[1]

        # Scaled dot-product attention
        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        # scores shape: (batch_size, num_heads, query_seq_len, kv_seq_len)

        attn_weights = ops.softmax(scores, axis=-1)

        if self.dropout_layer is not None:
            attn_weights = self.dropout_layer(attn_weights, training=training)

        # Apply attention to values
        out = ops.matmul(attn_weights, v)  # (batch_size, num_heads, query_seq_len, head_dim)

        # Reshape and project
        out = ops.transpose(out, (0, 2, 1, 3))  # (batch_size, query_seq_len, num_heads, head_dim)
        out = ops.reshape(out, (batch_size, query_seq_len, self.dim))
        out = self.proj_dense(out)

        return out

    def compute_output_shape(
            self,
            input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Input shape(s). Either single shape tuple or list of two shapes.

        Returns:
            Output shape tuple, same as query input shape.
        """
        if isinstance(input_shape, list):
            return input_shape[0]  # Same as query input shape
        else:
            return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Returns:
            Dictionary containing all layer configuration parameters.
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout_rate,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config