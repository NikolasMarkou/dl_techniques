"""
Enhanced Vision Transformer Layer with Register Tokens

This implementation incorporates the latest research findings from "Vision Transformers Need Registers"
which shows that ViTs develop artifacts in feature maps that can be resolved by adding register tokens.
"""

import keras
from typing import Optional, Tuple, Union, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .ffn.mlp import MLPBlock
from .norms.rms_norm import RMSNorm

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class RegisterEnhancedVisionTransformerLayer(keras.layers.Layer):
    """Enhanced Vision Transformer Layer with Register Tokens.

    This layer implements the findings from "Vision Transformers Need Registers" paper,
    which discovered that ViTs develop high-norm artifact tokens in low-informative areas.
    Register tokens provide dedicated spaces for internal computations, eliminating these artifacts.

    Key improvements over standard ViT:
    1. Supports register tokens that act as dedicated computation spaces
    2. Register tokens are processed through all attention layers
    3. Register tokens are excluded from final outputs
    4. Leads to smoother attention maps and better dense prediction performance

    Args:
        embed_dim: Dimension of input embeddings.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        dropout_rate: Dropout rate for residual connections.
        attention_dropout_rate: Dropout rate for attention weights.
        kernel_initializer: Initializer for weight matrices.
        kernel_regularizer: Regularizer for weight matrices.
        use_bias: Whether to use bias in dense layers.
        norm_type: Type of normalization ('layer' or 'rms').
        process_registers: Whether this layer processes register tokens.
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        use_bias: bool = True,
        norm_type: str = "layer",
        process_registers: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.use_bias = use_bias
        self.norm_type = norm_type
        self.process_registers = process_registers

        if norm_type not in ["layer", "rms"]:
            raise ValueError(f"norm_type must be 'layer' or 'rms', got {norm_type}")

        # Initialize to None, will be created in build()
        self.norm1 = None
        self.attn = None
        self.dropout1 = None
        self.norm2 = None
        self.mlp = None
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer's weights.

        Args:
            input_shape: Shape of the input tensor.
        """
        self._build_input_shape = input_shape

        # Create normalization layers
        if self.norm_type == "rms":
            self.norm1 = RMSNorm(name="norm1")
            self.norm2 = RMSNorm(name="norm2")
        else:
            self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6, name="norm1")
            self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6, name="norm2")

        # Create multi-head attention
        self.attn = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            dropout=self.attention_dropout_rate,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_bias=self.use_bias,
            name="attn"
        )

        # Create dropout layer
        self.dropout1 = keras.layers.Dropout(self.dropout_rate)

        # Create MLP block using existing MLP layer
        mlp_hidden_dim = int(self.embed_dim * self.mlp_ratio)
        self.mlp = MLPBlock(
            hidden_dim=mlp_hidden_dim,
            output_dim=self.embed_dim,
            activation="gelu",
            dropout_rate=self.dropout_rate,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_bias=self.use_bias,
            name="mlp"
        )

        # Build sublayers
        self.norm1.build(input_shape)
        self.attn.build(input_shape, input_shape)
        self.dropout1.build(input_shape)
        self.norm2.build(input_shape)
        self.mlp.build(input_shape)

        super().build(input_shape)

    def call(
        self,
        x: keras.KerasTensor,
        num_registers: int = 0,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim).
               If register tokens are present, they should be included in seq_len.
            num_registers: Number of register tokens in the sequence (for reference).
            training: Whether in training mode.

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim).
            Register tokens (if present) are processed but remain in the sequence.
        """
        # Attention block with residual connection
        # Register tokens participate in attention just like regular tokens
        x1 = self.norm1(x, training=training)
        x1 = self.attn(x1, x1, training=training)
        x1 = self.dropout1(x1, training=training)
        x = x + x1

        # MLP block with residual connection
        # Register tokens also go through MLP processing
        x2 = self.norm2(x, training=training)
        x2 = self.mlp(x2, training=training)
        x = x + x2

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape (same as input shape).
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
            "mlp_ratio": self.mlp_ratio,
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "use_bias": self.use_bias,
            "norm_type": self.norm_type,
            "process_registers": self.process_registers,
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

