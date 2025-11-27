"""
Progressive Focused Transformer (PFT) Block Module.

This module implements a complete PFT transformer block that combines:
- Progressive Focused Attention (PFA)
- Feed-Forward Network (FFN)
- Residual connections
- Normalization layers

References:
    Long, Wei, et al. "Progressive Focused Transformer for Single Image Super-Resolution."
    CVPR 2025.
"""

import keras
from typing import Optional, Tuple, Literal
from ..attention.progressive_focused_attention import ProgressiveFocusedAttention


@keras.saving.register_keras_serializable()
class PFTBlock(keras.layers.Layer):
    """
    Progressive Focused Transformer Block.

    A complete transformer block featuring:
    - Progressive Focused Attention with windowed/shifted windowed mechanism
    - Pre-normalization architecture (RMSNorm or LayerNorm)
    - Feed-forward network (configurable type via FFN factory)
    - Residual connections
    - Optional stochastic depth

    This block is the fundamental building unit of the PFT-SR architecture,
    designed to progressively refine attention patterns across layers while
    maintaining computational efficiency through windowed attention.

    Args:
        dim: Integer, the embedding dimension.
        num_heads: Integer, number of attention heads.
        window_size: Integer, size of the attention window. Default: 8.
        shift_size: Integer, shift size for SW-MSA. Use 0 for regular W-MSA,
            and window_size // 2 for shifted window. Default: 0.
        mlp_ratio: Float, expansion ratio for the MLP hidden dimension. Default: 4.0.
        qkv_bias: Boolean, whether to include bias in QKV projections. Default: True.
        attention_dropout: Float, dropout rate for attention weights. Default: 0.0.
        projection_dropout: Float, dropout rate for projections. Default: 0.0.
        drop_path: Float, stochastic depth rate. Default: 0.0.
        norm_type: String, type of normalization ('layer_norm' or 'rms_norm'). Default: 'layer_norm'.
        ffn_activation: String, activation function for FFN. Default: 'gelu'.
        use_lepe: Boolean, whether to use Locally-Enhanced Positional Encoding. Default: True.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Tuple of:
        - x: Tensor of shape `(batch_size, height, width, dim)`.
        - prev_attn_map: Optional attention map from previous layer.

    Output shape:
        Tuple of:
        - output: Tensor of shape `(batch_size, height, width, dim)`.
        - attn_map: Attention map tensor for next layer.

    Example:
        >>> import keras
        >>> x = keras.random.normal((2, 64, 64, 96))
        >>> block = PFTBlock(dim=96, num_heads=3, window_size=8, shift_size=0)
        >>> output, attn_map = block((x, None))
        >>> print(output.shape)
        (2, 64, 64, 96)
        >>>
        >>> # Shifted window block
        >>> shifted_block = PFTBlock(dim=96, num_heads=3, window_size=8, shift_size=4)
        >>> output, attn_map = shifted_block((output, attn_map))
        >>> print(output.shape)
        (2, 64, 64, 96)
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: int = 8,
            shift_size: int = 0,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            attention_dropout: float = 0.0,
            projection_dropout: float = 0.0,
            drop_path: float = 0.0,
            norm_type: Literal['layer_norm', 'rms_norm'] = 'layer_norm',
            ffn_activation: str = 'gelu',
            use_lepe: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.attention_dropout = attention_dropout
        self.projection_dropout = projection_dropout
        self.drop_path_rate = drop_path
        self.norm_type = norm_type
        self.ffn_activation = ffn_activation
        self.use_lepe = use_lepe

        # Validate shift size
        if self.shift_size >= self.window_size:
            raise ValueError(
                f"shift_size ({self.shift_size}) must be less than "
                f"window_size ({self.window_size})"
            )

    def build(self, input_shape):
        """
        Build layer components.

        Args:
            input_shape: Shape tuple or list of shape tuples.
        """
        # Handle input shape (x, prev_attn_map)
        if isinstance(input_shape, list):
            x_shape = input_shape[0]
        else:
            x_shape = input_shape

        # Normalization layers
        if self.norm_type == 'rms_norm':
            self.norm1 = keras.layers.LayerNormalization(
                epsilon=1e-6,
                name="norm1"
            )  # Using LayerNorm as proxy for RMSNorm
            self.norm2 = keras.layers.LayerNormalization(
                epsilon=1e-6,
                name="norm2"
            )
        else:
            self.norm1 = keras.layers.LayerNormalization(
                epsilon=1e-6,
                name="norm1"
            )
            self.norm2 = keras.layers.LayerNormalization(
                epsilon=1e-6,
                name="norm2"
            )

        # Progressive Focused Attention
        self.attn = ProgressiveFocusedAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            shift_size=self.shift_size,
            qkv_bias=self.qkv_bias,
            attention_dropout=self.attention_dropout,
            projection_dropout=self.projection_dropout,
            use_lepe=self.use_lepe,
            name="pfa"
        )

        # Feed-Forward Network
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = keras.Sequential([
            keras.layers.Dense(mlp_hidden_dim, name="fc1"),
            keras.layers.Activation(self.ffn_activation, name="act"),
            keras.layers.Dense(self.dim, name="fc2"),
        ], name="mlp")

        if self.projection_dropout > 0.0:
            self.mlp.add(keras.layers.Dropout(self.projection_dropout))

        # Stochastic depth
        if self.drop_path_rate > 0.0:
            self.drop_path = keras.layers.Dropout(
                self.drop_path_rate,
                noise_shape=(None, 1, 1, 1),
                name="drop_path"
            )
        else:
            self.drop_path = None

        super().build(input_shape)

    def call(
            self,
            inputs,
            training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """
        Forward pass of PFT block.

        Args:
            inputs: Tuple of (x, prev_attn_map) where:
                - x: Input tensor of shape (batch_size, height, width, dim).
                - prev_attn_map: Optional previous attention map.
            training: Boolean or boolean tensor, whether in training mode.

        Returns:
            Tuple of (output, attention_map).
        """
        # Unpack inputs
        if isinstance(inputs, (list, tuple)):
            x, prev_attn_map = inputs
        else:
            x = inputs
            prev_attn_map = None

        # Store shortcut
        shortcut = x

        # Pre-normalization + Progressive Focused Attention
        x = self.norm1(x)
        x, attn_map = self.attn(x, prev_attn_map=prev_attn_map, training=training)

        # Apply stochastic depth
        if self.drop_path is not None:
            x = self.drop_path(x, training=training)

        # First residual connection
        x = shortcut + x

        # Store shortcut for second residual
        shortcut = x

        # Pre-normalization + FFN
        x = self.norm2(x)
        x = self.mlp(x, training=training)

        # Apply stochastic depth
        if self.drop_path is not None:
            x = self.drop_path(x, training=training)

        # Second residual connection
        x = shortcut + x

        return x, attn_map

    def get_config(self):
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "shift_size": self.shift_size,
            "mlp_ratio": self.mlp_ratio,
            "qkv_bias": self.qkv_bias,
            "attention_dropout": self.attention_dropout,
            "projection_dropout": self.projection_dropout,
            "drop_path": self.drop_path_rate,
            "norm_type": self.norm_type,
            "ffn_activation": self.ffn_activation,
            "use_lepe": self.use_lepe,
        })
        return config