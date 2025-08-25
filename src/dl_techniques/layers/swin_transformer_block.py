import keras
from keras import ops
from typing import Tuple, Optional, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from ..utils.tensors import window_reverse, window_partition

from .ffn.swin_mlp import SwinMLP
from .stochastic_depth import StochasticDepth
from .attention.window_attention import WindowAttention


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SwinTransformerBlock(keras.layers.Layer):
    """
    Swin Transformer Block with windowed multi-head self-attention.

    This layer implements a Swin Transformer block that performs windowed multi-head
    self-attention followed by an MLP. It supports shifted windows for cross-window
    connections and includes optional stochastic depth regularization.

    The block follows the architecture from "Swin Transformer: Hierarchical Vision
    Transformer using Shifted Windows" by Liu et al. (2021). It processes input
    through the following sequence:
    1. Layer normalization
    2. Windowed multi-head self-attention (with optional shifting)
    3. Stochastic depth + residual connection
    4. Layer normalization
    5. MLP (Swin MLP variant)
    6. Stochastic depth + residual connection

    Key Features:
    - Window-based attention for computational efficiency
    - Optional shifted windows for cross-window connections
    - Stochastic depth regularization for deep networks
    - Configurable MLP expansion ratio

    Args:
        dim: Integer, number of input channels. Must be positive.
        num_heads: Integer, number of attention heads. Must be positive and
            divide dim evenly.
        window_size: Integer, window size for attention. Must be positive.
            Defaults to 8.
        shift_size: Integer, shift size for shifted window attention. Must be
            non-negative and less than window_size. Defaults to 0.
        mlp_ratio: Float, ratio of mlp hidden dim to embedding dim. Must be positive.
            Defaults to 4.0.
        qkv_bias: Boolean, if True, add a learnable bias to query, key, value.
            Defaults to True.
        dropout_rate: Float, dropout rate for MLP and projection. Must be in [0, 1).
            Defaults to 0.0.
        attn_dropout_rate: Float, attention dropout rate. Must be in [0, 1).
            Defaults to 0.0.
        drop_path: Float, stochastic depth rate. Must be in [0, 1). Defaults to 0.0.
        activation: String or callable, activation function for MLP. Defaults to "gelu".
        use_bias: Boolean, whether to use bias in layer normalization and projections.
            Defaults to True.
        kernel_initializer: String or initializer, initializer for the kernel weights.
            Defaults to "glorot_uniform".
        bias_initializer: String or initializer, initializer for the bias vector.
            Defaults to "zeros".
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias vector.
        activity_regularizer: Optional regularizer for layer output.
        **kwargs: Additional keyword arguments for Layer base class.

    Input shape:
        4D tensor with shape `(batch_size, height, width, channels)`

    Output shape:
        4D tensor with shape `(batch_size, height, width, channels)`

    Example:
        ```python
        # Basic usage
        block = SwinTransformerBlock(dim=96, num_heads=3, window_size=7)
        inputs = keras.Input(shape=(224, 224, 96))
        outputs = block(inputs)

        # With shifted windows and stochastic depth
        block = SwinTransformerBlock(
            dim=192,
            num_heads=6,
            window_size=7,
            shift_size=3,  # Half of window_size
            drop_path=0.1
        )

        # Custom configuration
        block = SwinTransformerBlock(
            dim=384,
            num_heads=12,
            window_size=8,
            mlp_ratio=6.0,
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
            activation='swish'
        )
        ```

    Raises:
        ValueError: If window_size <= 0, shift_size < 0, shift_size >= window_size,
            mlp_ratio <= 0, dim <= 0, num_heads <= 0, or dropout rates are not
            in valid range [0, 1).

    References:
        - Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
          (Liu et al., 2021): https://arxiv.org/abs/2103.14030
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: int = 8,
            shift_size: int = 0,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            dropout_rate: float = 0.0,
            attn_dropout_rate: float = 0.0,
            drop_path: float = 0.0,
            activation: Union[str, callable] = "gelu",
            use_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(
            activity_regularizer=activity_regularizer,
            **kwargs
        )

        # Validate arguments
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if shift_size < 0:
            raise ValueError(f"shift_size must be non-negative, got {shift_size}")
        if shift_size >= window_size:
            raise ValueError(f"shift_size ({shift_size}) must be less than window_size ({window_size})")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if not (0 <= dropout_rate < 1):
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
        if not (0 <= attn_dropout_rate < 1):
            raise ValueError(f"attn_dropout_rate must be in [0, 1), got {attn_dropout_rate}")
        if not (0 <= drop_path < 1):
            raise ValueError(f"drop_path must be in [0, 1), got {drop_path}")

        # Store ALL configuration parameters
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.drop_path_rate = drop_path
        self.activation = activation
        self.use_bias = use_bias

        # Store initializers and regularizers
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # CREATE all sub-layers in __init__ (they are unbuilt)
        # Following Pattern 2: Composite Layer from the guide

        # Layer normalization layers
        self.norm1 = keras.layers.LayerNormalization(
            epsilon=1e-5,
            center=self.use_bias,
            beta_initializer=self.bias_initializer if self.use_bias else "zeros",
            beta_regularizer=self.bias_regularizer if self.use_bias else None,
            name="norm1"
        )

        self.norm2 = keras.layers.LayerNormalization(
            epsilon=1e-5,
            center=self.use_bias,
            beta_initializer=self.bias_initializer if self.use_bias else "zeros",
            beta_regularizer=self.bias_regularizer if self.use_bias else None,
            name="norm2"
        )

        # Window attention layer
        self.attn = WindowAttention(
            dim=self.dim,
            window_size=self.window_size,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            attn_dropout_rate=self.attn_dropout_rate,
            proj_dropout_rate=self.dropout_rate,
            proj_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="attn"
        )

        # Stochastic depth layer (optional)
        if self.drop_path_rate > 0.0:
            self.drop_path_layer = StochasticDepth(
                drop_path_rate=self.drop_path_rate,
                name="drop_path"
            )
        else:
            self.drop_path_layer = None

        # MLP layer
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = SwinMLP(
            hidden_dim=mlp_hidden_dim,
            use_bias=self.use_bias,
            out_dim=self.dim,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="mlp"
        )

        logger.debug(f"Initialized SwinTransformerBlock with dim={dim}, num_heads={num_heads}, "
                     f"window_size={window_size}, shift_size={shift_size}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
        This ensures all weight variables exist before weight restoration.

        Args:
            input_shape: Shape tuple of the input tensor (B, H, W, C).

        Raises:
            ValueError: If input_shape is not 4D or channels dimension doesn't match dim.
        """
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape (B, H, W, C), got {input_shape}")

        if input_shape[-1] is not None and input_shape[-1] != self.dim:
            raise ValueError(f"Input channels ({input_shape[-1]}) must match dim ({self.dim})")

        # Build sub-layers in computational order
        # Normalization and MLP layers operate on the main input shape
        self.norm1.build(input_shape)
        self.norm2.build(input_shape)
        self.mlp.build(input_shape)

        # Build stochastic depth layer if it exists
        if self.drop_path_layer is not None:
            self.drop_path_layer.build(input_shape)

        # Attention layer expects windowed input shape
        # After window partitioning: (batch * num_windows, window_size * window_size, channels)
        attn_input_shape = (None, self.window_size * self.window_size, self.dim)
        self.attn.build(attn_input_shape)

        # Always call parent build at the end
        super().build(input_shape)
        logger.debug(f"Built SwinTransformerBlock with input_shape={input_shape}")

    def call(
            self,
            x: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the SwinTransformerBlock layer.

        Args:
            x: Input tensor of shape (B, H, W, C).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor of shape (B, H, W, C).

        Raises:
            ValueError: If input tensor is not 4D.
        """
        if len(ops.shape(x)) != 4:
            raise ValueError(f"Expected 4D input tensor, got shape {ops.shape(x)}")

        B, H, W, C = ops.shape(x)[0], ops.shape(x)[1], ops.shape(x)[2], ops.shape(x)[3]
        shortcut = x

        # =============================================
        # Multi-head Self-Attention Block
        # =============================================

        # Layer Norm 1
        x = self.norm1(x, training=training)

        # Window attention with optional cyclic shift
        if self.shift_size > 0:
            # Cyclic shift for shifted window attention
            shifted_x = ops.roll(x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x

        # Partition windows: (B, H, W, C) -> (B*num_windows, window_size, window_size, C)
        x_windows = window_partition(shifted_x, self.window_size)

        # Reshape for attention: (B*num_windows, window_size*window_size, C)
        x_windows = ops.reshape(x_windows, (-1, self.window_size * self.window_size, C))

        # Window-based multi-head self-attention
        attn_windows = self.attn(x_windows, training=training)

        # Reshape back to windows: (B*num_windows, window_size, window_size, C)
        attn_windows = ops.reshape(attn_windows, (-1, self.window_size, self.window_size, C))

        # Reverse window partition: (B*num_windows, window_size, window_size, C) -> (B, H, W, C)
        x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift if it was applied
        if self.shift_size > 0:
            x = ops.roll(x, shift=(self.shift_size, self.shift_size), axis=(1, 2))

        # Apply stochastic depth and residual connection
        if self.drop_path_layer is not None:
            x = shortcut + self.drop_path_layer(x, training=training)
        else:
            x = shortcut + x

        # =============================================
        # MLP Block
        # =============================================

        shortcut = x

        # Layer Norm 2
        x = self.norm2(x, training=training)

        # MLP
        x = self.mlp(x, training=training)

        # Apply stochastic depth and residual connection
        if self.drop_path_layer is not None:
            x = shortcut + self.drop_path_layer(x, training=training)
        else:
            x = shortcut + x

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input (B, H, W, C).

        Returns:
            Output shape tuple (B, H, W, C). SwinTransformerBlock preserves input shape.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        CRITICAL: Must include ALL __init__ parameters for proper serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "shift_size": self.shift_size,
            "mlp_ratio": self.mlp_ratio,
            "qkv_bias": self.qkv_bias,
            "dropout_rate": self.dropout_rate,
            "attn_dropout_rate": self.attn_dropout_rate,
            "drop_path": self.drop_path_rate,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
