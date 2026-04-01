"""
SwinConvBlock: A hybrid Keras layer that synergistically combines the strengths
of convolutional neural networks (CNNs) and the Swin Transformer architecture.

This block is designed to capture both local and global dependencies in image-like
data by processing features through two parallel pathways, which are then fused back
together. The core idea is to leverage the inductive biases and efficiency of
convolutions for local feature extraction, while simultaneously using the powerful,
long-range modeling capabilities of window-based self-attention.
"""

import keras
from keras import ops
from typing import Tuple, Optional, Dict, Any, Union

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .swin_transformer_block import SwinTransformerBlock

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SwinConvBlock(keras.layers.Layer):
    """
    Hybrid Swin-Conv block combining transformer and convolutional paths.

    Processes input through parallel convolutional and Swin Transformer
    pathways that specialize in complementary feature extraction (local
    patterns vs. long-range context), then fuses results with a 1x1
    convolution and a main residual connection.

    The mathematical operations are:
    ``x1 = Conv1x1(x)``, split into ``x_conv, x_trans``,
    ``x_conv' = Conv3x3(ReLU(Conv3x3(x_conv))) + x_conv``,
    ``x_trans' = SwinTransformerBlock(x_trans)``,
    ``output = Conv1x1(Concat[x_conv', x_trans']) + x``.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────────┐
        │  Input (B, H, W, conv_dim + trans_dim)   │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  1x1 Conv2D (feature preparation)        │
        └──────────────────┬───────────────────────┘
                           ▼
              ┌────────────┴────────────┐
              ▼                         ▼
        ┌───────────┐          ┌────────────────┐
        │ Conv Path │          │ Transformer    │
        │ 3x3 Conv  │          │ Path           │
        │ ReLU      │          │ SwinBlock      │
        │ 3x3 Conv  │          │ (W/SW-MSA+MLP) │
        │ +Residual │          │                │
        └─────┬─────┘          └──────┬─────────┘
              └────────────┬──────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  Concatenate ─► 1x1 Conv2D (fusion)      │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  + Main Residual from Input              │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │  Output (B, H, W, conv_dim + trans_dim)  │
        └──────────────────────────────────────────┘

    :param conv_dim: Channels for the convolutional path.
    :type conv_dim: int
    :param trans_dim: Channels for the transformer path.
    :type trans_dim: int
    :param head_dim: Dimension per attention head. Default: 32.
    :type head_dim: int
    :param window_size: Attention window size. Default: 8.
    :type window_size: int
    :param drop_path_rate: Stochastic depth rate. Default: 0.0.
    :type drop_path_rate: float
    :param block_type: ``'W'`` for regular or ``'SW'`` for shifted windows.
    :type block_type: str
    :param input_resolution: Optional spatial resolution hint; if
        ``<= window_size``, forces regular windows.
    :type input_resolution: Optional[int]
    :param mlp_ratio: MLP expansion ratio. Default: 4.0.
    :type mlp_ratio: float
    :param use_bias: Whether layers use bias. Default: True.
    :type use_bias: bool
    :param kernel_initializer: Kernel weight initializer.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Bias weight initializer.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Kernel weight regularizer.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Bias weight regularizer.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param activity_regularizer: Activity regularizer.
    :type activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param kwargs: Additional keyword arguments for the base Layer.
    :type kwargs: Any

    :raises ValueError: If dimension, rate, or block_type parameters are invalid.
    """

    def __init__(
        self,
        conv_dim: int,
        trans_dim: int,
        head_dim: int = 32,
        window_size: int = 8,
        drop_path_rate: float = 0.0,
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
            raise ValueError(
                f"trans_dim ({trans_dim}) must be divisible by head_dim ({head_dim})"
            )
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if not (0 <= drop_path_rate < 1):
            raise ValueError(f"drop_path_rate must be in [0, 1), got {drop_path_rate}")
        if block_type not in ["W", "SW"]:
            raise ValueError(f"block_type must be 'W' or 'SW', got '{block_type}'")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if input_resolution is not None and input_resolution <= 0:
            raise ValueError(
                f"input_resolution must be positive when provided, got {input_resolution}"
            )

        # Store ALL configuration parameters for serialization
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.num_heads = trans_dim // head_dim
        self.window_size = window_size
        self.drop_path_rate = drop_path_rate
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
        self.effective_block_type = self.block_type
        if self.input_resolution is not None and self.input_resolution <= self.window_size:
            original_block_type = self.block_type
            self.effective_block_type = "W"
            logger.info(
                f"Input resolution {self.input_resolution} <= window_size {self.window_size}, "
                f"switching from '{original_block_type}' to regular window attention 'W'"
            )

        # CREATE all sub-layers in __init__ (they are unbuilt)
        # Following Pattern 2: Composite Layer from the Keras 3 guide

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
        shift_size = self.window_size // 2 if self.effective_block_type == "SW" else 0

        # Transformer block for global feature modeling
        self.trans_block = SwinTransformerBlock(
            dim=self.trans_dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            shift_size=shift_size,
            mlp_ratio=self.mlp_ratio,
            stochastic_depth_rate=self.drop_path_rate,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="trans_block"
        )

        # Convolutional block for local feature extraction (residual block with two conv layers)
        self.conv_block = keras.Sequential([
            keras.layers.Conv2D(
                filters=self.conv_dim,
                kernel_size=3,
                strides=1,
                padding="same",
                use_bias=False,  # First conv without bias (common practice before activation)
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

        # Output 1x1 conv to mix features from both paths
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

        logger.debug(
            f"Initialized SwinConvBlock with conv_dim={conv_dim}, trans_dim={trans_dim}, "
            f"num_heads={self.num_heads}, window_size={window_size}, "
            f"block_type='{block_type}' -> effective='{self.effective_block_type}'"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and all sub-layers for serialization safety.

        :param input_shape: Shape tuple ``(B, H, W, C)`` where
            ``C = conv_dim + trans_dim``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If shape is not 4-D or channels mismatch.
        """
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape (B, H, W, C), got {input_shape}"
            )

        input_channels = input_shape[-1]
        expected_channels = self.conv_dim + self.trans_dim

        if input_channels is not None and input_channels != expected_channels:
            raise ValueError(
                f"Input channels ({input_channels}) must match conv_dim + trans_dim "
                f"({self.conv_dim} + {self.trans_dim} = {expected_channels})"
            )

        # Build sub-layers in computational order with appropriate shapes

        # 1. Initial 1x1 conv processes full input
        self.conv1_1.build(input_shape)

        # 2. After split, conv block processes only the conv_dim portion
        conv_input_shape = (input_shape[0], input_shape[1], input_shape[2], self.conv_dim)
        self.conv_block.build(conv_input_shape)

        # 3. Transformer block processes only the trans_dim portion
        trans_input_shape = (input_shape[0], input_shape[1], input_shape[2], self.trans_dim)
        self.trans_block.build(trans_input_shape)

        # 4. Output 1x1 conv processes concatenated features from both paths
        combined_shape = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            self.conv_dim + self.trans_dim
        )
        self.conv1_2.build(combined_shape)

        # Always call parent build at the end
        super().build(input_shape)
        logger.debug(f"Built SwinConvBlock with input_shape={input_shape}")

    def call(
        self,
        x: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass: split-transform-merge with main residual.

        :param x: Input tensor ``(B, H, W, C)``.
        :type x: keras.KerasTensor
        :param training: Training mode flag.
        :type training: Optional[bool]
        :return: Output tensor ``(B, H, W, C)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If input is not 4-D.
        """
        if len(ops.shape(x)) != 4:
            raise ValueError(f"Expected 4D input tensor, got shape {ops.shape(x)}")

        # Store shortcut for main residual connection
        shortcut = x

        # =============================================
        # Phase 1: Initial Feature Processing
        # =============================================

        # Initial 1x1 conv to process and prepare features
        x = self.conv1_1(x, training=training)

        # =============================================
        # Phase 2: Split into Parallel Pathways
        # =============================================

        # Split along channel dimension for parallel processing
        # conv_x: [B, H, W, conv_dim], trans_x: [B, H, W, trans_dim]
        conv_x, trans_x = ops.split(x, [self.conv_dim], axis=-1)

        # =============================================
        # Phase 3a: Convolutional Pathway (Local)
        # =============================================

        # Process through residual conv block for local pattern extraction
        conv_out = self.conv_block(conv_x, training=training)

        # Add internal residual connection for better gradient flow
        conv_x = conv_out + conv_x

        # =============================================
        # Phase 3b: Transformer Pathway (Global)
        # =============================================

        # Process through Swin Transformer block for long-range dependencies
        # (includes its own internal residual connections)
        trans_x = self.trans_block(trans_x, training=training)

        # =============================================
        # Phase 4: Merge and Fusion
        # =============================================

        # Concatenate pathway outputs along channel dimension
        x = ops.concatenate([conv_x, trans_x], axis=-1)

        # Final 1x1 conv to mix and fuse features from both pathways
        x = self.conv1_2(x, training=training)

        # =============================================
        # Phase 5: Main Residual Connection
        # =============================================

        # Add main skip connection combining with original input
        x = x + shortcut

        return x

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape (same as input).

        :param input_shape: Input shape tuple.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for serialization.

        :return: Dictionary containing all constructor parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "conv_dim": self.conv_dim,
            "trans_dim": self.trans_dim,
            "head_dim": self.head_dim,
            "window_size": self.window_size,
            "drop_path_rate": self.drop_path_rate,
            "block_type": self.block_type,  # Store original, not effective
            "input_resolution": self.input_resolution,
            "mlp_ratio": self.mlp_ratio,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config


# ---------------------------------------------------------------------