"""Complete Perceiver transformer block with cross-attention and MLP."""

import keras
from keras import ops
from typing import Optional, Any, Dict, Tuple

from dl_techniques.layers.perceiver_attention import PerceiverAttention
from dl_techniques.utils.logger import logger


@keras.saving.register_keras_serializable()
class PerceiverBlock(keras.layers.Layer):
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
        dim: Integer, hidden dimension of the block.
        num_heads: Integer, number of attention heads. Defaults to 8.
        mlp_ratio: Float, ratio of MLP hidden dimension to input dimension.
            Defaults to 4.0 (standard transformer ratio).
        dropout: Float, dropout rate for attention and MLP. Defaults to 0.0.
        activation: String or callable, activation function for MLP. Defaults to "gelu".
        use_bias: Boolean, whether to use bias in projections. Defaults to True.
        kernel_initializer: Initializer for kernel weights.
        bias_initializer: Initializer for bias vectors.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Two inputs:
        - query_input: 3D tensor with shape `(batch_size, query_seq_len, dim)`
        - kv_input: 3D tensor with shape `(batch_size, kv_seq_len, dim)`

    Output shape:
        3D tensor with shape: `(batch_size, query_seq_len, dim)`

    Call arguments:
        query_input: Query tensor of shape (batch_size, query_seq_len, dim).
        kv_input: Key-Value tensor of shape (batch_size, kv_seq_len, dim).
        training: Boolean indicating whether the layer should behave in training
            mode or inference mode.

    Returns:
        Output tensor with same shape as query_input.

    Example:
        >>> # Cross-modal transformer block
        >>> text_features = keras.random.normal((2, 77, 512))    # Text tokens
        >>> visual_features = keras.random.normal((2, 196, 512)) # Image patches
        >>>
        >>> perceiver_block = PerceiverBlock(dim=512, num_heads=8, mlp_ratio=4.0)
        >>>
        >>> # Text attending to visual features and processing through MLP
        >>> output = perceiver_block(text_features, visual_features)
        >>> print(output.shape)  # (2, 77, 512) - same as query input
        >>>
        >>> # Can also be used for latent-to-input processing
        >>> latents = keras.random.normal((2, 32, 512))  # Learned latent queries
        >>> processed_latents = perceiver_block(latents, visual_features)
        >>> print(processed_latents.shape)  # (2, 32, 512)
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            dropout: float = 0.0,
            activation: str = "gelu",
            use_bias: bool = True,
            kernel_initializer: str = "glorot_uniform",
            bias_initializer: str = "zeros",
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        # Calculate MLP hidden dimension
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        # Store build information
        self._build_input_shape = None

        # Will be created in build()
        self.norm1_q = None
        self.norm1_kv = None
        self.attention = None
        self.norm2 = None
        self.mlp_dense1 = None
        self.mlp_dense2 = None
        self.dropout_layer = None

    def build(self, input_shape):
        """Build the layer weights."""
        # Handle different input formats
        if isinstance(input_shape, list):
            # Two separate inputs
            if len(input_shape) != 2:
                raise ValueError(f"Expected 2 inputs, got {len(input_shape)}")
            query_shape, kv_shape = input_shape
            self._build_input_shape = input_shape
        else:
            # Single input shape (will be used for both query and kv)
            query_shape = kv_shape = input_shape
            self._build_input_shape = input_shape

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

        # Layer normalization for query input
        self.norm1_q = keras.layers.LayerNormalization(
            epsilon=1e-6,
            name="norm1_q"
        )

        # Layer normalization for key-value input
        self.norm1_kv = keras.layers.LayerNormalization(
            epsilon=1e-6,
            name="norm1_kv"
        )

        # Perceiver cross-attention
        self.attention = PerceiverAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="attention"
        )

        # Layer normalization before MLP
        self.norm2 = keras.layers.LayerNormalization(
            epsilon=1e-6,
            name="norm2"
        )

        # MLP layers
        self.mlp_dense1 = keras.layers.Dense(
            self.mlp_hidden_dim,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="mlp_dense1"
        )

        self.mlp_dense2 = keras.layers.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="mlp_dense2"
        )

        # Dropout layer for MLP
        if self.dropout_rate > 0.0:
            self.dropout_layer = keras.layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, query_input, kv_input=None, training=None):
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
        normed_q = self.norm1_q(query_input)
        normed_kv = self.norm1_kv(kv_input)

        attn_output = self.attention(
            normed_q,
            normed_kv,
            training=training
        )

        # Add residual connection
        x = query_input + attn_output

        # Second residual connection: MLP
        normed_x = self.norm2(x)

        # MLP forward pass
        mlp_output = self.mlp_dense1(normed_x)

        if self.dropout_layer is not None:
            mlp_output = self.dropout_layer(mlp_output, training=training)

        mlp_output = self.mlp_dense2(mlp_output)

        # Add residual connection
        output = x + mlp_output

        return output

    def compute_output_shape(self, input_shape):
        """Compute the output shape of the layer."""
        if isinstance(input_shape, list):
            return input_shape[0]  # Same as query input shape
        else:
            return input_shape

    def get_config(self):
        """Returns the layer configuration."""
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
        })
        return config

    def get_build_config(self):
        """Get build configuration."""
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config):
        """Build from configuration."""
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])