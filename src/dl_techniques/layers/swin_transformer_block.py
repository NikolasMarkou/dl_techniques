import keras
from keras import ops
from typing import Tuple, Optional, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.tensors import window_reverse, window_partition

from .swin_mlp import SwinMLP
from .stochastic_depth import StochasticDepth
from .window_attention import WindowAttention


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SwinTransformerBlock(keras.layers.Layer):
    """Swin Transformer Block with windowed multi-head self-attention.

    This layer implements a Swin Transformer block that performs windowed multi-head
    self-attention followed by an MLP. It supports shifted windows for cross-window
    connections and includes optional stochastic depth regularization.

    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.
        window_size: Window size for attention. Must be positive. Defaults to 8.
        shift_size: Shift size for shifted window attention. Must be non-negative
            and less than window_size. Defaults to 0.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim. Must be positive.
            Defaults to 4.0.
        qkv_bias: If True, add a learnable bias to query, key, value. Defaults to True.
        drop: Dropout rate. Must be in [0, 1). Defaults to 0.0.
        attn_drop: Attention dropout rate. Must be in [0, 1). Defaults to 0.0.
        drop_path: Stochastic depth rate. Must be in [0, 1). Defaults to 0.0.
        act_layer: Activation layer. Defaults to "gelu".
        use_bias: Use bias in layer normalization and projections. Defaults to True.
        kernel_initializer: Initializer for the kernel weights. Defaults to "glorot_uniform".
        bias_initializer: Initializer for the bias vector. Defaults to "zeros".
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias vector.
        activity_regularizer: Optional regularizer for layer output.
        **kwargs: Additional keyword arguments for Layer base class.

    Input shape:
        4D tensor with shape (batch_size, height, width, channels).

    Output shape:
        4D tensor with shape (batch_size, height, width, channels).

    Raises:
        ValueError: If window_size <= 0, shift_size < 0, shift_size >= window_size,
            mlp_ratio <= 0, or dropout rates are not in valid range.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: int = 8,
            shift_size: int = 0,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            drop_path: float = 0.0,
            act_layer: str = "gelu",
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
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if shift_size < 0:
            raise ValueError(f"shift_size must be non-negative, got {shift_size}")
        if shift_size >= window_size:
            raise ValueError(f"shift_size ({shift_size}) must be less than window_size ({window_size})")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if not (0 <= drop < 1):
            raise ValueError(f"drop must be in [0, 1), got {drop}")
        if not (0 <= attn_drop < 1):
            raise ValueError(f"attn_drop must be in [0, 1), got {attn_drop}")
        if not (0 <= drop_path < 1):
            raise ValueError(f"drop_path must be in [0, 1), got {drop_path}")

        # Store configuration parameters
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop
        self.attn_drop_rate = attn_drop
        self.drop_path_rate = drop_path
        self.act_layer = act_layer
        self.use_bias = use_bias

        # Store initializers and regularizers
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Initialize layers to None - will be created in build()
        self.norm1 = None
        self.attn = None
        self.drop_path = None
        self.norm2 = None
        self.mlp = None

        # Store build input shape for serialization
        self._build_input_shape = None

        logger.debug(f"Initialized SwinTransformerBlock with dim={dim}, num_heads={num_heads}, "
                     f"window_size={window_size}, shift_size={shift_size}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer's weights and sublayers.

        Args:
            input_shape: Shape tuple of the input tensor (B, H, W, C).

        Raises:
            ValueError: If input_shape is not 4D or channels dimension doesn't match dim.
        """
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape (B, H, W, C), got {input_shape}")

        if input_shape[-1] is not None and input_shape[-1] != self.dim:
            raise ValueError(f"Input channels ({input_shape[-1]}) must match dim ({self.dim})")

        # Store input shape for serialization
        self._build_input_shape = input_shape

        # Create normalization layers
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

        # Create attention layer
        self.attn = WindowAttention(
            dim=self.dim,
            window_size=self.window_size,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            attn_drop=self.attn_drop_rate,
            proj_drop=self.drop_rate,
            proj_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="attn"
        )

        # Create stochastic depth layer if needed
        if self.drop_path_rate > 0.0:
            self.drop_path = StochasticDepth(
                drop_path_rate=self.drop_path_rate,
                name="drop_path"
            )

        # Create MLP layer
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = SwinMLP(
            hidden_dim=mlp_hidden_dim,
            use_bias=self.use_bias,
            out_dim=self.dim,
            act_layer=self.act_layer,
            drop=self.drop_rate,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="mlp"
        )

        # ------------------- FIX STARTS HERE -------------------
        # A parent layer's build() method MUST build all its children.
        # This ensures they are ready to receive weights during model loading.

        # norm1, norm2, mlp, and drop_path all operate on the main input shape.
        self.norm1.build(input_shape)
        self.norm2.build(input_shape)
        self.mlp.build(input_shape)
        if self.drop_path is not None:
            self.drop_path.build(input_shape)

        # Explicitly build the attention sublayer to prevent "out of scope" errors
        # during graph tracing. The attention layer expects an input of shape
        # (batch, num_tokens, channels).
        attn_input_shape = (None, self.window_size * self.window_size, self.dim)
        self.attn.build(attn_input_shape)
        # -------------------- FIX ENDS HERE --------------------

        super().build(input_shape)
        logger.debug(f"Built SwinTransformerBlock with input_shape={input_shape}")

    def call(self, x: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass of the SwinTransformerBlock layer.

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

        # Layer Norm 1
        x = self.norm1(x, training=training)

        # Window attention with optional cyclic shift
        if self.shift_size > 0:
            # Cyclic shift
            shifted_x = ops.roll(x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # (B*num_windows, window_size, window_size, C)

        # Reshape for attention
        x_windows = ops.reshape(x_windows, (-1, self.window_size * self.window_size, C))  # (B*num_windows, N, C)

        # Window attention
        attn_windows = self.attn(x_windows, training=training)  # (B*num_windows, N, C)

        # Reshape and reverse window partition
        attn_windows = ops.reshape(attn_windows, (-1, self.window_size, self.window_size, C))

        if self.shift_size > 0:
            # Reverse cyclic shift
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)
            x = ops.roll(shifted_x, shift=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x = window_reverse(attn_windows, self.window_size, H, W)

        # Apply drop path and residual connection
        if self.drop_path is not None:
            x = shortcut + self.drop_path(x, training=training)
        else:
            x = shortcut + x

        # MLP block
        shortcut = x
        x = self.norm2(x, training=training)
        x = self.mlp(x, training=training)

        if self.drop_path is not None:
            x = shortcut + self.drop_path(x, training=training)
        else:
            x = shortcut + x

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input (B, H, W, C).

        Returns:
            Output shape tuple (B, H, W, C).
        """
        # SwinTransformerBlock preserves input shape
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer configuration for serialization.

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
            "drop": self.drop_rate,
            "attn_drop": self.attn_drop_rate,
            "drop_path": self.drop_path_rate,
            "act_layer": self.act_layer,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the config needed to build the layer from a config.

        Returns:
            Dictionary containing the build configuration.
        """
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------