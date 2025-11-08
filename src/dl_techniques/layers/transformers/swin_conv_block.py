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
    Hybrid Swin-Conv block combining transformer and convolutional paths in parallel.

    This advanced hybrid block synergistically combines the efficiency and inductive
    biases of convolutional neural networks with the long-range modeling capability
    of Swin Transformer blocks. It processes input through parallel convolutional
    and transformer pathways that specialize in complementary feature extraction,
    then combines the results with residual connections for enhanced representation
    learning.

    **Intent**: Capture both local spatial patterns (via CNNs) and long-range
    contextual dependencies (via Swin Transformers) in a single unified block that
    leverages the complementary strengths of both architectures. Particularly
    effective for vision tasks requiring multi-scale feature understanding.

    **Architecture**:
    ```
                             Input Tensor
                          (B, H, W, C_total)
                                 ↓
                         ┌───────────────┐
                         │  1×1 Conv2D   │  Initial feature processing
                         │ (C_total out) │
                         └───────────────┘
                                 ↓
                    ┌────────────┴────────────┐
                    │   Channel Split         │
                    │  [conv_dim | trans_dim] │
                    └────────────┬────────────┘
                         ↙               ↘
              ┌─────────────┐    ┌──────────────────┐
              │  Conv Path  │    │ Transformer Path │
              │  (conv_dim) │    │   (trans_dim)    │
              └─────────────┘    └──────────────────┘
                     ↓                     ↓
              ┌─────────────┐    ┌──────────────────┐
              │  Conv2D 3×3 │    │ SwinTransformer  │
              │     ReLU    │    │     Block        │
              │  Conv2D 3×3 │    │  (W-MSA/SW-MSA)  │
              └─────────────┘    └──────────────────┘
                     ↓                     ↓
              ┌─────────────┐    ┌──────────────────┐
              │ + Residual  │    │ (with internal   │
              │  Connection │    │  residual)       │
              └─────────────┘    └──────────────────┘
                     ↓                     ↓
                     └──────────┬──────────┘
                                ↓
                       ┌─────────────────┐
                       │  Concatenate    │
                       │ [conv_dim + ... │
                       │  ... trans_dim] │
                       └─────────────────┘
                                ↓
                       ┌─────────────────┐
                       │   1×1 Conv2D    │  Feature fusion
                       │  (C_total out)  │
                       └─────────────────┘
                                ↓
                       ┌─────────────────┐
                       │  + Main Skip    │  Main residual
                       │   Connection    │  from input
                       └─────────────────┘
                                ↓
                         Output Tensor
                        (B, H, W, C_total)

    Where: C_total = conv_dim + trans_dim
    ```

    **Split-Transform-Merge Flow**:
    ```
    Input Features
         ↓
    ┌─────────────────────────────────────┐
    │ Phase 1: Initial Processing         │
    │  • 1×1 Conv: Prepare features       │
    └─────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────┐
    │ Phase 2: Channel Split              │
    │  • Conv Path:  conv_dim channels    │
    │  • Trans Path: trans_dim channels   │
    └─────────────────────────────────────┘
         ↓
    ┌──────────────────┬──────────────────┐
    │   Conv Branch    │  Transformer Br. │
    │                  │                  │
    │ Local Patterns   │ Global Context   │
    │ • 3×3 Conv       │ • Window Attn    │
    │ • ReLU           │ • MLP            │
    │ • 3×3 Conv       │ • LayerNorms     │
    │ • + Skip         │ • + Skips        │
    └──────────────────┴──────────────────┘
         ↓           ↓
    ┌─────────────────────────────────────┐
    │ Phase 3: Merge & Fusion             │
    │  • Concatenate along channels       │
    │  • 1×1 Conv: Mix pathway features   │
    │  • + Main residual from input       │
    └─────────────────────────────────────┘
         ↓
    Output Features
    ```

    **Pathway Specialization**:
    ```
    Convolutional Path:              Transformer Path:
    ┌─────────────────────┐         ┌──────────────────────┐
    │ Strengths:          │         │ Strengths:           │
    │ • Local patterns    │         │ • Long-range deps    │
    │ • Spatial hierarchy │         │ • Global context     │
    │ • Translation inv.  │         │ • Adaptive attention │
    │ • Computational eff.│         │ • Content-based      │
    │                     │         │   relationships      │
    │ Captures:           │         │                      │
    │ • Edges, textures   │         │ Captures:            │
    │ • Local structures  │         │ • Semantic relations │
    │ • Spatial gradients │         │ • Contextual info    │
    └─────────────────────┘         │ • Position-aware     │
                                    │   features           │
                                    └──────────────────────┘
    ```

    **Mathematical Operations**:

    1. **Initial Processing**:
        x₁ = Conv1×1(x) ∈ ℝ^(B×H×W×(Dc+Dt))

    2. **Channel Split**:
        x_conv, x_trans = Split(x₁, [Dc, Dt])
        where Dc = conv_dim, Dt = trans_dim

    3. **Convolutional Path** (with residual):
        h_conv = Conv3×3(ReLU(Conv3×3(x_conv)))
        x_conv' = h_conv + x_conv

    4. **Transformer Path** (with internal residuals):
        x_trans' = SwinTransformerBlock(x_trans)
                 = x_trans + MLP(LN(x_trans + W-MSA(LN(x_trans))))

    5. **Merge and Fusion**:
        x₂ = Concat([x_conv', x_trans'], axis=-1)
        x₃ = Conv1×1(x₂)

    6. **Main Residual**:
        output = x₃ + x  (main skip connection)

    Where:
    - B = batch size, H = height, W = width
    - Dc = conv_dim, Dt = trans_dim
    - W-MSA = Window-based Multi-head Self-Attention
    - LN = LayerNormalization
    - MLP = Multi-Layer Perceptron

    Args:
        conv_dim: Integer, number of channels for the convolutional path. Must be
            positive. Determines the capacity of the CNN branch for local feature
            extraction.
        trans_dim: Integer, number of channels for the transformer path. Must be
            positive. Determines the capacity of the Swin Transformer branch for
            global feature modeling.
        head_dim: Integer, dimension of each attention head in the transformer path.
            Must be positive and divide trans_dim evenly. Controls the granularity
            of attention mechanisms. Defaults to 32.
        window_size: Integer, size of the attention window in the transformer path.
            Must be positive. Determines the local window for computing self-attention.
            Typical values are 7 or 8. Defaults to 8.
        drop_path_rate: Float, stochastic depth rate for regularization. Must be in
            range [0, 1). Higher values increase regularization by randomly dropping
            entire residual paths during training. Defaults to 0.0 (no stochastic depth).
        block_type: String, type of attention mechanism. Must be "W" for regular
            window attention or "SW" for shifted window attention. Shifted windows
            enable cross-window connections for better information flow. Defaults to "W".
        input_resolution: Optional integer, spatial resolution of input feature maps.
            If provided and less than or equal to window_size, the block automatically
            switches to regular window attention regardless of block_type setting,
            since shifting is unnecessary for small resolutions. Defaults to None.
        mlp_ratio: Float, ratio of MLP hidden dimension to embedding dimension in
            the transformer block. Must be positive. Higher values increase the
            capacity of the feed-forward network. Defaults to 4.0.
        use_bias: Boolean, whether to use bias terms in convolutions and linear
            projections. Setting to False can reduce parameters and is common in
            networks with normalization layers. Defaults to True.
        kernel_initializer: String or Initializer instance, initializer for the
            convolutional kernel weights. Accepts names like 'glorot_uniform',
            'he_normal', or Initializer objects. Defaults to 'glorot_uniform'.
        bias_initializer: String or Initializer instance, initializer for bias vectors.
            Only used when use_bias=True. Defaults to 'zeros'.
        kernel_regularizer: Optional Regularizer instance for kernel weights. Can be
            used to apply L1, L2, or custom regularization to prevent overfitting.
            Defaults to None (no regularization).
        bias_regularizer: Optional Regularizer instance for bias vectors. Only
            applied when use_bias=True. Defaults to None (no regularization).
        activity_regularizer: Optional Regularizer instance for the layer's output
            activations. Can be used to encourage sparsity or other properties in
            the learned representations. Defaults to None (no regularization).
        **kwargs: Additional keyword arguments passed to the Layer base class,
            such as 'name', 'trainable', 'dtype'.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`
        where channels = conv_dim + trans_dim.

        The spatial dimensions (height, width) should ideally be divisible by
        window_size for optimal attention computation, though the layer handles
        arbitrary sizes through padding.

    Output shape:
        4D tensor with shape: `(batch_size, height, width, channels)`.

        The output preserves the exact shape of the input tensor due to the
        main residual connection and shape-preserving convolutions.

    Attributes:
        conv_dim: Number of channels in the convolutional pathway.
        trans_dim: Number of channels in the transformer pathway.
        head_dim: Dimension of each attention head.
        num_heads: Number of attention heads (trans_dim // head_dim).
        window_size: Size of the attention window.
        drop_path_rate: Stochastic depth rate.
        block_type: Original attention type specified ("W" or "SW").
        effective_block_type: Actual attention type used (may differ if input is small).
        input_resolution: Optional input resolution hint.
        mlp_ratio: MLP expansion ratio in transformer block.
        use_bias: Whether bias terms are used.
        kernel_initializer: Kernel weight initializer.
        bias_initializer: Bias vector initializer.
        kernel_regularizer: Kernel regularizer.
        bias_regularizer: Bias regularizer.
        conv1_1: Initial 1×1 convolution layer.
        trans_block: Swin Transformer block for global modeling.
        conv_block: Sequential convolutional block for local modeling.
        conv1_2: Final 1×1 convolution for feature fusion.

    Example:
        ```python
        # Basic usage with balanced pathway split
        block = SwinConvBlock(conv_dim=48, trans_dim=48, head_dim=24)
        inputs = keras.Input(shape=(224, 224, 96))  # 48 + 48 = 96 channels
        outputs = block(inputs)
        print(outputs.shape)  # (None, 224, 224, 96)

        # With shifted window attention and stochastic depth
        block = SwinConvBlock(
            conv_dim=64,
            trans_dim=64,
            head_dim=32,
            window_size=7,
            block_type="SW",  # Shifted windows for cross-window connections
            drop_path_rate=0.1
        )
        inputs = keras.Input(shape=(56, 56, 128))  # 64 + 64 = 128 channels
        outputs = block(inputs)

        # Asymmetric split: more capacity for transformer path
        block = SwinConvBlock(
            conv_dim=96,
            trans_dim=192,  # 2× the conv capacity
            head_dim=48,
            window_size=8,
            mlp_ratio=6.0,
            input_resolution=56  # Hint for optimization
        )
        inputs = keras.Input(shape=(56, 56, 288))  # 96 + 192 = 288 channels
        outputs = block(inputs)

        # With regularization to prevent overfitting
        block = SwinConvBlock(
            conv_dim=128,
            trans_dim=128,
            head_dim=32,
            kernel_regularizer=keras.regularizers.L2(1e-4),
            bias_regularizer=keras.regularizers.L2(1e-5),
            activity_regularizer=keras.regularizers.L1(1e-6)
        )
        inputs = keras.Input(shape=(28, 28, 256))  # 128 + 128 = 256 channels
        outputs = block(inputs)

        # Integration in a larger model
        inputs = keras.Input(shape=(224, 224, 3))
        x = keras.layers.Conv2D(96, 4, strides=4)(inputs)  # Stem: (56, 56, 96)

        # Stack multiple SwinConvBlocks
        for i in range(4):
            x = SwinConvBlock(
                conv_dim=48,
                trans_dim=48,
                head_dim=24,
                window_size=7,
                block_type="W" if i % 2 == 0 else "SW",  # Alternate W and SW
                drop_path_rate=0.1 * (i / 4),  # Gradually increase
                name=f"swin_conv_block_{i}"
            )(x)

        outputs = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(1000, activation='softmax')(outputs)
        model = keras.Model(inputs, outputs)
        ```

    Raises:
        ValueError: If conv_dim <= 0, trans_dim <= 0, head_dim <= 0, window_size <= 0,
            trans_dim is not divisible by head_dim, mlp_ratio <= 0, drop_path_rate not
            in [0, 1), block_type is not "W" or "SW", or input_resolution <= 0 when
            provided.
        ValueError: In build(), if input_shape is not 4D or if input channels don't
            match conv_dim + trans_dim.
        ValueError: In call(), if input tensor is not 4D.

    Note:
        - The input tensor must have channels equal to conv_dim + trans_dim for proper
          pathway splitting. This is validated in the build() method.
        - When input_resolution <= window_size, the block automatically switches to
          regular window attention ("W") regardless of the block_type setting, since
          shifted windows provide no benefit for small feature maps.
        - The convolutional path uses a residual connection internally, and the entire
          block has a main residual connection from input to output, creating multiple
          gradient flow pathways for stable training.
        - For optimal attention computation, input spatial dimensions (height, width)
          should ideally be divisible by window_size, though the implementation handles
          arbitrary sizes.
        - The first 3×3 convolution in the conv_block intentionally has no bias (common
          practice when followed by activation), while the second has bias based on
          use_bias parameter.

    References:
        - Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
          (Liu et al., 2021): https://arxiv.org/abs/2103.14030
        - ConvNeXt: A ConvNet for the 2020s
          (Liu et al., 2022): https://arxiv.org/abs/2201.03545
        - Deep Residual Learning for Image Recognition
          (He et al., 2016): https://arxiv.org/abs/1512.03385
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
        """
        Build the layer and all its sub-layers.

        CRITICAL: This method explicitly builds each sub-layer to ensure all weight
        variables exist before weight restoration during model loading. This is
        essential for robust serialization and follows Keras 3 best practices.

        The build order follows the computational flow: initial conv → conv block
        and transformer block (parallel) → final conv. Each sub-layer is built with
        the appropriate input shape it will receive during forward pass.

        Args:
            input_shape: Shape tuple of the input tensor (B, H, W, C) where:
                - B: batch size (can be None)
                - H: height (can be None)
                - W: width (can be None)
                - C: channels (must equal conv_dim + trans_dim)

        Raises:
            ValueError: If input_shape is not 4D (missing batch, height, width, or
                channel dimensions).
            ValueError: If input channels don't match the expected total dimensions
                (conv_dim + trans_dim). This ensures proper pathway splitting.
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
        """
        Forward pass implementing the split-transform-merge paradigm with residuals.

        This method orchestrates the parallel processing through convolutional and
        transformer pathways, combining their complementary representations for
        enhanced feature learning. The computation follows this flow:

        1. **Initial Processing**: Apply 1×1 conv to prepare features
        2. **Split**: Separate channels into conv and transformer pathways
        3. **Transform (Parallel)**:
           - Conv path: Local feature extraction with residual
           - Trans path: Global attention-based modeling
        4. **Merge**: Concatenate pathway outputs
        5. **Fusion**: Mix features with 1×1 conv
        6. **Residual**: Add main skip connection from input

        Args:
            x: Input tensor of shape (B, H, W, C) where:
                - B: batch size
                - H: height dimension
                - W: width dimension
                - C: channels (must equal conv_dim + trans_dim)
            training: Boolean or None. When True, the layer behaves in training mode
                (e.g., applying dropout, stochastic depth). When False or None, the
                layer behaves in inference mode. Passed to all sub-layers.

        Returns:
            Output tensor of shape (B, H, W, C), matching the input shape exactly
            due to the main residual connection and shape-preserving operations.

        Raises:
            ValueError: If input tensor is not 4D, indicating incorrect input shape.
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
        """
        Compute the output shape of the layer.

        The SwinConvBlock preserves the input shape exactly due to the combination of:
        - Shape-preserving 1×1 and 3×3 convolutions with 'same' padding
        - Channel split and concatenation that maintain total channel count
        - Main residual connection that requires matching shapes

        Args:
            input_shape: Shape tuple of the input (B, H, W, C).

        Returns:
            Output shape tuple (B, H, W, C), identical to input_shape.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration dictionary for serialization.

        CRITICAL: This method must include ALL __init__ parameters to ensure proper
        reconstruction during model loading. The configuration is used by Keras to
        recreate the layer with identical settings.

        Returns:
            Dictionary containing the complete layer configuration, including all
            constructor parameters and their serialized representations. This dict
            is used to reconstruct the layer via __init__() during deserialization.
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