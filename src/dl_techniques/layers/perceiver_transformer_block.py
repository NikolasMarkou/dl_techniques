"""Complete Perceiver transformer block with cross-attention and MLP."""

import keras
from typing import Optional, Any, Dict, Tuple, Union, List


# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .attention.cross_attention_perceiver_attention import PerceiverAttention

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class PerceiverTransformerBlock(keras.layers.Layer):
    """Complete Perceiver transformer block with cross-attention.

    This block implements a complete Perceiver transformer layer that combines:
    1. Layer normalization
    2. Perceiver cross-attention (queries from one input, keys/values from another)
    3. Residual connection
    4. Layer normalization
    5. MLP (feed-forward network)
    6. Residual connection

    This is useful for cross-modal processing, set-to-set transformations,
    and latent space learning where different inputs need to interact.

    Args:
        dim: Integer, hidden dimension of the block. Must be positive.
        num_heads: Integer, number of attention heads. Must be positive and divide dim.
            Defaults to 8.
        mlp_ratio: Float, ratio of MLP hidden dimension to input dimension.
            Must be positive. Defaults to 4.0 (standard transformer ratio).
        dropout: Float, dropout rate for attention and MLP. Must be between 0 and 1.
            Defaults to 0.0.
        activation: String or callable, activation function for MLP. Defaults to "gelu".
        use_bias: Boolean, whether to use bias in projections. Defaults to True.
        kernel_initializer: String or Initializer, initializer for kernel weights.
            Defaults to "glorot_uniform".
        bias_initializer: String or Initializer, initializer for bias vectors.
            Defaults to "zeros".
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Two inputs (can be called with single input for self-attention):
        - query_input: 3D tensor with shape `(batch_size, query_seq_len, dim)`
        - kv_input: 3D tensor with shape `(batch_size, kv_seq_len, dim)` (optional)

    Output shape:
        3D tensor with shape: `(batch_size, query_seq_len, dim)`

    Call arguments:
        query_input: Query tensor of shape (batch_size, query_seq_len, dim).
        kv_input: Key-Value tensor of shape (batch_size, kv_seq_len, dim).
            If None, uses query_input for both (self-attention mode).
        training: Boolean indicating whether the layer should behave in training
            mode or inference mode.

    Returns:
        Output tensor with same shape as query_input.

    Example:
        ```python
        # Cross-modal transformer block
        text_features = keras.random.normal((2, 77, 512))    # Text tokens
        visual_features = keras.random.normal((2, 196, 512)) # Image patches

        perceiver_block = PerceiverBlock(dim=512, num_heads=8, mlp_ratio=4.0)

        # Text attending to visual features and processing through MLP
        output = perceiver_block(text_features, visual_features)
        print(output.shape)  # (2, 77, 512) - same as query input

        # Can also be used for latent-to-input processing
        latents = keras.random.normal((2, 32, 512))  # Learned latent queries
        processed_latents = perceiver_block(latents, visual_features)
        print(processed_latents.shape)  # (2, 32, 512)

        # Self-attention mode
        output = perceiver_block(text_features)  # Uses text_features for both Q and KV
        ```

    Raises:
        ValueError: If dim is not positive, num_heads doesn't divide dim, or mlp_ratio
            is not positive.
        ValueError: If dropout is not between 0 and 1.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            dropout: float = 0.0,
            activation: Union[str, callable] = "gelu",
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
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be between 0 and 1, got {dropout}")

        # Store ALL configuration
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Calculate MLP hidden dimension
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        # CREATE all sub-layers in __init__ (they are unbuilt)
        self.norm1_q = keras.layers.LayerNormalization(
            epsilon=1e-6,
            name="norm1_q"
        )

        self.norm1_kv = keras.layers.LayerNormalization(
            epsilon=1e-6,
            name="norm1_kv"
        )

        self.attention = PerceiverAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="attention"
        )

        self.norm2 = keras.layers.LayerNormalization(
            epsilon=1e-6,
            name="norm2"
        )

        self.mlp_dense1 = keras.layers.Dense(
            self.mlp_hidden_dim,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="mlp_dense1"
        )

        self.mlp_dense2 = keras.layers.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="mlp_dense2"
        )

        if self.dropout_rate > 0.0:
            self.dropout_layer = keras.layers.Dropout(self.dropout_rate, name="dropout")
        else:
            self.dropout_layer = None

    def build(self, input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]) -> None:
        """Build the layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
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
        self.norm1_q.build(query_shape)
        self.norm1_kv.build(kv_shape)

        # Attention layer needs to be built with proper input shapes
        self.attention.build([query_shape, kv_shape])

        # After attention, shape remains same as query
        attn_output_shape = query_shape

        self.norm2.build(attn_output_shape)
        self.mlp_dense1.build(attn_output_shape)

        # MLP dense1 output shape
        mlp_intermediate_shape = list(attn_output_shape)
        mlp_intermediate_shape[-1] = self.mlp_hidden_dim
        mlp_intermediate_shape = tuple(mlp_intermediate_shape)

        if self.dropout_layer is not None:
            self.dropout_layer.build(mlp_intermediate_shape)

        self.mlp_dense2.build(mlp_intermediate_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            query_input: keras.KerasTensor,
            kv_input: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply Perceiver block processing.

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

        # First residual connection: Cross-attention
        normed_q = self.norm1_q(query_input, training=training)
        normed_kv = self.norm1_kv(kv_input, training=training)

        attn_output = self.attention(
            normed_q,
            normed_kv,
            training=training
        )

        # Add residual connection
        x = query_input + attn_output

        # Second residual connection: MLP
        normed_x = self.norm2(x, training=training)

        # MLP forward pass
        mlp_output = self.mlp_dense1(normed_x, training=training)

        if self.dropout_layer is not None:
            mlp_output = self.dropout_layer(mlp_output, training=training)

        mlp_output = self.mlp_dense2(mlp_output, training=training)

        # Add residual connection
        output = x + mlp_output

        return output

    def compute_output_shape(self, input_shape: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer."""
        if isinstance(input_shape, list):
            return input_shape[0]  # Same as query input shape
        else:
            return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout_rate,
            "activation": keras.activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config