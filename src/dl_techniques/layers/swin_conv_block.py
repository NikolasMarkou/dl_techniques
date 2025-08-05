"""
This module provides the `SwinConvBlock`, a hybrid Keras layer that synergistically
combines the strengths of convolutional neural networks (CNNs) and the Swin
Transformer architecture.

This block is designed to capture both local and global dependencies in image-like
data by processing features through two parallel pathways, which are then fused back
together. The core idea is to leverage the inductive biases and efficiency of
convolutions for local feature extraction, while simultaneously using the powerful,
long-range modeling capabilities of window-based self-attention.

Architectural Design:

The `SwinConvBlock` operates on a "split-transform-merge" principle within a residual
framework.

1.  **Input Processing and Splitting:**
    -   The input tensor first passes through a `1x1 Conv2D` layer to prepare the
        features for the two specialized paths.
    -   The output of this convolution is then split along the channel dimension into
        two separate tensors: one for the convolutional path (`conv_x`) and one for the
        Transformer path (`trans_x`). The size of these splits is controlled by
        `conv_dim` and `trans_dim`.

2.  **Parallel Pathways:**
    -   **Convolutional Path:** `conv_x` is processed by a standard residual
        convolutional block (typically two `3x3 Conv2D` layers with a ReLU activation).
        This path excels at learning local patterns, textures, and spatial hierarchies
        efficiently.
    -   **Transformer Path:** `trans_x` is processed by a `SwinTransformerBlock`. This
        path uses windowed self-attention (either regular 'W' or shifted 'SW') to
        model long-range dependencies and contextual relationships between different
        parts of the feature map.

3.  **Fusion and Output:**
    -   The outputs of the two parallel paths are concatenated back together along the
        channel dimension.
    -   This fused tensor is then passed through a final `1x1 Conv2D` layer to mix the
        features from both paths.
    -   A main residual connection adds the original input (`shortcut`) to the output
        of the entire block, ensuring stable gradient flow and allowing the network
        to easily bypass the block if needed.

By integrating these two paradigms, the `SwinConvBlock` aims to create a more powerful
and versatile building block for computer vision models, capable of learning a richer
set of features than either approach could alone.
"""

import keras
from keras import ops
from typing import Tuple, Optional, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from .swin_transformer_block import SwinTransformerBlock

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SwinConvBlock(keras.layers.Layer):
    """Swin-Conv block combining transformer and convolutional paths.

    This block combines the efficiency of convolutional operations with the
    long-range modeling capability of Swin Transformer blocks. It processes
    input through parallel convolutional and transformer paths, then combines
    the results with a residual connection.

    Args:
        conv_dim: Number of channels for the convolutional path. Must be positive.
        trans_dim: Number of channels for the transformer path. Must be positive.
        head_dim: Dimension of each attention head. Must be positive and divide trans_dim evenly.
            Defaults to 32.
        window_size: Size of attention window. Must be positive. Defaults to 8.
        drop_path: Stochastic depth rate. Must be in [0, 1). Defaults to 0.0.
        block_type: Type of attention block ("W" for window, "SW" for shifted window).
            Defaults to "W".
        input_resolution: Input resolution for optimization. If provided and less than or
            equal to window_size, regular window attention is used. Defaults to None.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim. Must be positive. Defaults to 4.0.
        use_bias: Whether to use bias in convolutions and projections. Defaults to True.
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
        ValueError: If conv_dim <= 0, trans_dim <= 0, head_dim <= 0, window_size <= 0,
            trans_dim is not divisible by head_dim, mlp_ratio <= 0, drop_path not in [0, 1),
            or block_type is not "W" or "SW".
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
        if conv_dim <= 0:
            raise ValueError(f"conv_dim must be positive, got {conv_dim}")
        if trans_dim <= 0:
            raise ValueError(f"trans_dim must be positive, got {trans_dim}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if trans_dim % head_dim != 0:
            raise ValueError(f"trans_dim ({trans_dim}) must be divisible by head_dim ({head_dim})")
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if not (0 <= drop_path < 1):
            raise ValueError(f"drop_path must be in [0, 1), got {drop_path}")
        if block_type not in ["W", "SW"]:
            raise ValueError(f"block_type must be 'W' or 'SW', got '{block_type}'")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if input_resolution is not None and input_resolution <= 0:
            raise ValueError(f"input_resolution must be positive when provided, got {input_resolution}")

        # Store configuration parameters
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.num_heads = trans_dim // head_dim
        self.window_size = window_size
        self.drop_path_rate = drop_path
        self.block_type = block_type
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio
        self.use_bias = use_bias

        # Store initializers and regularizers
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # If input resolution is too small, use regular window attention
        if self.input_resolution is not None and self.input_resolution <= self.window_size:
            original_block_type = self.block_type
            self.block_type = "W"
            logger.info(f"Input resolution {self.input_resolution} <= window_size {self.window_size}, "
                        f"switching from '{original_block_type}' to regular window attention 'W'")

        # Initialize layers to None - will be created in build()
        self.conv1_1 = None
        self.trans_block = None
        self.conv_block = None
        self.conv1_2 = None

        # Store build input shape for serialization
        self._build_input_shape = None

        logger.debug(f"Initialized SwinConvBlock with conv_dim={conv_dim}, trans_dim={trans_dim}, "
                     f"num_heads={self.num_heads}, window_size={window_size}, block_type='{block_type}'")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer's weights and sublayers.

        Args:
            input_shape: Shape tuple of the input tensor (B, H, W, C).

        Raises:
            ValueError: If input_shape is not 4D or total channels don't match expected dimensions.
        """
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape (B, H, W, C), got {input_shape}")

        input_channels = input_shape[-1]
        expected_channels = self.conv_dim + self.trans_dim

        if input_channels is not None and input_channels != expected_channels:
            raise ValueError(f"Input channels ({input_channels}) must match conv_dim + trans_dim "
                             f"({self.conv_dim} + {self.trans_dim} = {expected_channels})")

        # Store input shape for serialization
        self._build_input_shape = input_shape

        # Initial 1x1 conv to process input
        self.conv1_1 = keras.layers.Conv2D(
            filters=self.conv_dim + self.trans_dim,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer if self.use_bias else "zeros",
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer if self.use_bias else None,
            name="conv1_1"
        )

        # Determine shift size for shifted window attention
        shift_size = self.window_size // 2 if self.block_type == "SW" else 0

        # Transformer block
        self.trans_block = SwinTransformerBlock(
            dim=self.trans_dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            shift_size=shift_size,
            mlp_ratio=self.mlp_ratio,
            drop_path=self.drop_path_rate,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="trans_block"
        )

        # Convolutional block
        self.conv_block = keras.Sequential([
            keras.layers.Conv2D(
                filters=self.conv_dim,
                kernel_size=3,
                strides=1,
                padding="same",
                use_bias=False,  # First conv without bias
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="conv1"
            ),
            keras.layers.ReLU(name="relu"),
            keras.layers.Conv2D(
                filters=self.conv_dim,
                kernel_size=3,
                strides=1,
                padding="same",
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer if self.use_bias else "zeros",
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer if self.use_bias else None,
                name="conv2"
            )
        ], name="conv_block")

        # Output 1x1 conv
        self.conv1_2 = keras.layers.Conv2D(
            filters=self.conv_dim + self.trans_dim,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer if self.use_bias else "zeros",
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer if self.use_bias else None,
            name="conv1_2"
        )

        # Build all sublayers explicitly
        self.conv1_1.build(input_shape)

        # Build conv_block with input shape
        self.conv_block.build((input_shape[0], input_shape[1], input_shape[2], self.conv_dim))

        # Build transformer block with transformer path shape
        trans_input_shape = (input_shape[0], input_shape[1], input_shape[2], self.trans_dim)
        self.trans_block.build(trans_input_shape)

        # Build output conv with combined shape
        combined_shape = (input_shape[0], input_shape[1], input_shape[2], self.conv_dim + self.trans_dim)
        self.conv1_2.build(combined_shape)

        super().build(input_shape)
        logger.debug(f"Built SwinConvBlock with input_shape={input_shape}")

    def call(self, x: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """Forward pass of the SwinConvBlock layer.

        Args:
            x: Input tensor of shape (B, H, W, C) where C = conv_dim + trans_dim.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor of shape (B, H, W, C).

        Raises:
            ValueError: If input tensor is not 4D.
        """
        if len(ops.shape(x)) != 4:
            raise ValueError(f"Expected 4D input tensor, got shape {ops.shape(x)}")

        shortcut = x

        # Initial 1x1 conv to process input
        x = self.conv1_1(x, training=training)

        # Split into convolutional and transformer paths
        conv_x, trans_x = ops.split(x, [self.conv_dim], axis=-1)

        # Convolutional path with residual connection
        conv_out = self.conv_block(conv_x, training=training)
        conv_x = conv_out + conv_x  # Residual connection within conv path

        # Transformer path
        trans_x = self.trans_block(trans_x, training=training)

        # Combine paths and apply output conv
        x = ops.concatenate([conv_x, trans_x], axis=-1)
        x = self.conv1_2(x, training=training)

        # Main residual connection
        x = x + shortcut

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input (B, H, W, C).

        Returns:
            Output shape tuple (B, H, W, C).
        """
        # SwinConvBlock preserves input shape
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