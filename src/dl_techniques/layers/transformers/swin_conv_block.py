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
and versatile building block for computer vision_heads models, capable of learning a richer
set of features than either approach could alone.
"""

import keras
from keras import ops
from typing import Tuple, Optional, Dict, Any, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .swin_transformer_block import SwinTransformerBlock

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SwinConvBlock(keras.layers.Layer):
    """
    Swin-Conv block combining transformer and convolutional paths.

    This hybrid block combines the efficiency of convolutional operations with the
    long-range modeling capability of Swin Transformer blocks. It processes input
    through parallel convolutional and transformer paths, then combines the results
    with a residual connection for enhanced feature learning.

    The block implements a "split-transform-merge" approach:
    1. Initial 1×1 convolution processes input features
    2. Features are split into conv and transformer pathways
    3. Conv path: Residual block with two 3×3 convolutions + ReLU
    4. Transformer path: SwinTransformerBlock with windowed attention
    5. Paths are concatenated and processed by final 1×1 convolution
    6. Main residual connection combines with original input

    This design leverages the complementary strengths of both architectures:
    - CNNs excel at local pattern recognition and spatial hierarchy
    - Transformers excel at long-range dependencies and contextual understanding

    Key Features:
    - Parallel convolutional and transformer processing
    - Configurable window size and attention type (W/SW)
    - Residual connections at multiple levels
    - Efficient channel splitting for pathway specialization

    Args:
        conv_dim: Integer, number of channels for the convolutional path. Must be positive.
        trans_dim: Integer, number of channels for the transformer path. Must be positive.
        head_dim: Integer, dimension of each attention head. Must be positive and divide
            trans_dim evenly. Defaults to 32.
        window_size: Integer, size of attention window. Must be positive. Defaults to 8.
        drop_path: Float, stochastic depth rate. Must be in [0, 1). Defaults to 0.0.
        block_type: String, type of attention block. Must be "W" for window attention
            or "SW" for shifted window attention. Defaults to "W".
        input_resolution: Optional integer, input resolution for optimization. If provided
            and less than or equal to window_size, regular window attention is used.
            Defaults to None.
        mlp_ratio: Float, ratio of mlp hidden dim to embedding dim in transformer block.
            Must be positive. Defaults to 4.0.
        use_bias: Boolean, whether to use bias in convolutions and projections.
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
        where channels = conv_dim + trans_dim

    Output shape:
        4D tensor with shape `(batch_size, height, width, channels)`

    Example:
        ```python
        # Basic usage
        block = SwinConvBlock(conv_dim=48, trans_dim=48, head_dim=24)
        inputs = keras.Input(shape=(224, 224, 96))  # 48 + 48 = 96 channels
        outputs = block(inputs)

        # With shifted window attention and stochastic depth
        block = SwinConvBlock(
            conv_dim=64,
            trans_dim=64,
            head_dim=32,
            window_size=7,
            block_type="SW",  # Shifted windows
            drop_path=0.1
        )
        inputs = keras.Input(shape=(56, 56, 128))  # 64 + 64 = 128 channels
        outputs = block(inputs)

        # Custom configuration with regularization
        block = SwinConvBlock(
            conv_dim=96,
            trans_dim=192,
            head_dim=48,
            window_size=8,
            mlp_ratio=6.0,
            kernel_regularizer=keras.regularizers.L2(1e-4),
            bias_regularizer=keras.regularizers.L2(1e-5)
        )
        inputs = keras.Input(shape=(28, 28, 288))  # 96 + 192 = 288 channels
        outputs = block(inputs)
        ```

    Raises:
        ValueError: If conv_dim <= 0, trans_dim <= 0, head_dim <= 0, window_size <= 0,
            trans_dim is not divisible by head_dim, mlp_ratio <= 0, drop_path not in [0, 1),
            block_type is not "W" or "SW", or input_resolution <= 0 when provided.

    Note:
        The input tensor must have channels equal to conv_dim + trans_dim for proper
        pathway splitting. When input_resolution <= window_size, the block automatically
        switches to regular window attention regardless of block_type setting.

    References:
        - Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
          (Liu et al., 2021): https://arxiv.org/abs/2103.14030
        - ConvNeXT: A ConvNet for the 2020s (Liu et al., 2022): https://arxiv.org/abs/2201.03545
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

        # Store ALL configuration parameters
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
        self.effective_block_type = self.block_type
        if self.input_resolution is not None and self.input_resolution <= self.window_size:
            original_block_type = self.block_type
            self.effective_block_type = "W"
            logger.info(f"Input resolution {self.input_resolution} <= window_size {self.window_size}, "
                        f"switching from '{original_block_type}' to regular window attention 'W'")

        # CREATE all sub-layers in __init__ (they are unbuilt)
        # Following Pattern 2: Composite Layer from the guide

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

        # Convolutional block (residual block with two conv layers)
        self.conv_block = keras.Sequential([
            keras.layers.Conv2D(
                filters=self.conv_dim,
                kernel_size=3,
                strides=1,
                padding="same",
                use_bias=False,  # First conv without bias (common practice)
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

        logger.debug(f"Initialized SwinConvBlock with conv_dim={conv_dim}, trans_dim={trans_dim}, "
                     f"num_heads={self.num_heads}, window_size={window_size}, "
                     f"block_type='{block_type}' -> effective='{self.effective_block_type}'")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
        This ensures all weight variables exist before weight restoration.

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

        # Build sub-layers in computational order with appropriate shapes

        # Initial 1x1 conv processes full input
        self.conv1_1.build(input_shape)

        # After initial conv, we have (B, H, W, conv_dim + trans_dim)
        conv1_1_output_shape = self.conv1_1.compute_output_shape(input_shape)

        # Conv block processes only the conv_dim portion
        conv_input_shape = (input_shape[0], input_shape[1], input_shape[2], self.conv_dim)
        self.conv_block.build(conv_input_shape)

        # Transformer block processes only the trans_dim portion
        trans_input_shape = (input_shape[0], input_shape[1], input_shape[2], self.trans_dim)
        self.trans_block.build(trans_input_shape)

        # Output 1x1 conv processes concatenated features
        combined_shape = (input_shape[0], input_shape[1], input_shape[2], self.conv_dim + self.trans_dim)
        self.conv1_2.build(combined_shape)

        # Always call parent build at the end
        super().build(input_shape)
        logger.debug(f"Built SwinConvBlock with input_shape={input_shape}")

    def call(
        self,
        x: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the SwinConvBlock layer.

        Implements the split-transform-merge paradigm with residual connections:
        1. Process input through initial 1×1 conv
        2. Split features into conv and transformer pathways
        3. Process each pathway independently
        4. Concatenate pathway outputs
        5. Apply final 1×1 conv and main residual connection

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

        # Store shortcut for main residual connection
        shortcut = x

        # =============================================
        # Initial Feature Processing
        # =============================================

        # Initial 1x1 conv to process and prepare features
        x = self.conv1_1(x, training=training)

        # =============================================
        # Split into Parallel Pathways
        # =============================================

        # Split along channel dimension for parallel processing
        conv_x, trans_x = ops.split(x, [self.conv_dim], axis=-1)

        # =============================================
        # Convolutional Pathway
        # =============================================

        # Process through residual conv block
        conv_out = self.conv_block(conv_x, training=training)

        # Add residual connection within conv path for better gradient flow
        conv_x = conv_out + conv_x

        # =============================================
        # Transformer Pathway
        # =============================================

        # Process through Swin Transformer block (includes its own residual connections)
        trans_x = self.trans_block(trans_x, training=training)

        # =============================================
        # Merge and Output
        # =============================================

        # Concatenate pathway outputs
        x = ops.concatenate([conv_x, trans_x], axis=-1)

        # Final 1x1 conv to mix features from both pathways
        x = self.conv1_2(x, training=training)

        # Main residual connection combining with original input
        x = x + shortcut

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape of the input (B, H, W, C).

        Returns:
            Output shape tuple (B, H, W, C). SwinConvBlock preserves input shape.
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
            "conv_dim": self.conv_dim,
            "trans_dim": self.trans_dim,
            "head_dim": self.head_dim,
            "window_size": self.window_size,
            "drop_path": self.drop_path_rate,
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
