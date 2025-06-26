import keras
from keras import ops
from typing import Tuple, Optional, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .swin_transformer_block import SwinTransformerBlock

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SwinConvBlock(keras.layers.Layer):
    """Swin-Conv block combining transformer and convolutional paths.

    This block combines the efficiency of convolutional operations with the
    long-range modeling capability of Swin Transformer blocks.

    Args:
        conv_dim: Number of channels for the convolutional path.
        trans_dim: Number of channels for the transformer path.
        head_dim: Dimension of each attention head. Defaults to 32.
        window_size: Size of attention window. Defaults to 8.
        drop_path: Stochastic depth rate. Defaults to 0.0.
        block_type: Type of attention block ("W" for window, "SW" for shifted window). Defaults to "W".
        input_resolution: Input resolution for optimization. Defaults to None.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim. Defaults to 4.0.
        **kwargs: Additional keyword arguments for Layer base class.

    Raises:
        ValueError: If trans_dim is not divisible by head_dim.
    """

    def __init__(
            self,
            conv_dim: int,
            trans_dim: int,
            head_dim: int = 32,
            window_size: int = 8,
            drop_path: float = 0.0,
            block_type: str = "W",
            input_resolution: Optional[int] = None,
            mlp_ratio: float = 4.0,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if trans_dim % head_dim != 0:
            raise ValueError(f"trans_dim ({trans_dim}) must be divisible by head_dim ({head_dim})")

        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.num_heads = trans_dim // head_dim
        self.window_size = window_size
        self.drop_path_rate = drop_path
        self.block_type = block_type
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio

        # If input resolution is too small, use regular window attention
        if self.input_resolution is not None and self.input_resolution <= self.window_size:
            self.block_type = "W"
            logger.info(f"Input resolution {self.input_resolution} <= window_size {self.window_size}, "
                        f"switching to regular window attention")

        # Initialize layers to None - will be created in build()
        self.conv1_1 = None
        self.trans_block = None
        self.conv_block = None
        self.conv1_2 = None

        # Store build input shape for serialization
        self._build_input_shape = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer's weights and sublayers.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        # Initial 1x1 conv to process input
        self.conv1_1 = keras.layers.Conv2D(
            self.conv_dim + self.trans_dim,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=True,
            name="conv1_1"
        )

        # Shift size for shifted window attention
        shift_size = self.window_size // 2 if self.block_type == "SW" else 0

        # Transformer block
        self.trans_block = SwinTransformerBlock(
            dim=self.trans_dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            shift_size=shift_size,
            mlp_ratio=self.mlp_ratio,
            drop_path=self.drop_path_rate,
            name="trans_block"
        )

        # Convolutional block
        self.conv_block = keras.Sequential([
            keras.layers.Conv2D(
                self.conv_dim,
                kernel_size=3,
                strides=1,
                padding="same",
                use_bias=False,
                name="conv1"
            ),
            keras.layers.ReLU(name="relu"),
            keras.layers.Conv2D(
                self.conv_dim,
                kernel_size=3,
                strides=1,
                padding="same",
                use_bias=False,
                name="conv2"
            )
        ], name="conv_block")

        # Output 1x1 conv
        self.conv1_2 = keras.layers.Conv2D(
            self.conv_dim + self.trans_dim,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=True,
            name="conv1_2"
        )

        super().build(input_shape)

    def call(self, x: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass of the SwinConvBlock layer.

        Args:
            x: Input tensor of shape (B, H, W, C).
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor of shape (B, H, W, C).
        """
        shortcut = x

        # Split into convolutional and transformer paths
        x = self.conv1_1(x)
        conv_x, trans_x = ops.split(x, [self.conv_dim, self.trans_dim], axis=-1)

        # Convolutional path with residual connection
        conv_out = self.conv_block(conv_x, training=training)
        conv_x = conv_out + conv_x

        # Transformer path
        trans_x = self.trans_block(trans_x, training=training)

        # Combine paths and apply output conv
        x = ops.concatenate([conv_x, trans_x], axis=-1)
        x = self.conv1_2(x)

        # Residual connection
        x = x + shortcut

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input.

        Returns:
            Output shape tuple.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "conv_dim": self.conv_dim,
            "trans_dim": self.trans_dim,
            "head_dim": self.head_dim,
            "window_size": self.window_size,
            "drop_path": self.drop_path_rate,
            "block_type": self.block_type,
            "input_resolution": self.input_resolution,
            "mlp_ratio": self.mlp_ratio,
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
