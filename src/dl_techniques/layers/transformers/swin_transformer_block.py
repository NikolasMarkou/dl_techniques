"""
Swin Transformer Block Implementation

This module implements the core SwinTransformerBlock layer, which forms the fundamental
building block of the Swin Transformer architecture - a hierarchical vision_heads transformer
that has revolutionized computer vision_heads by achieving state-of-the-art performance across
multiple tasks including image classification, object detection, and semantic segmentation.

Architecture Overview
--------------------

The Swin Transformer introduces a paradigm shift from traditional Vision Transformers (ViTs)
by replacing global self-attention with a more computationally efficient windowed attention
mechanism. The name "Swin" stands for "Shifted Windows", which refers to the key innovation
that enables the model to capture both local and global dependencies effectively.

Key Innovations:

1. **Windowed Multi-Head Self-Attention (W-MSA)**: Divides input feature maps into
   non-overlapping windows and computes self-attention within each window, reducing
   computational complexity from O((HW)²) to O(M²HW), where M is the fixed window size.

2. **Shifted Window Multi-Head Self-Attention (SW-MSA)**: Shifts windows by ⌊M/2⌋ pixels
   in both horizontal and vertical directions, enabling cross-window connections and
   information exchange between neighboring windows.

3. **Hierarchical Architecture**: Uses a pyramid-like structure with patch merging layers
   that progressively reduce spatial resolution while increasing feature dimensions,
   similar to CNNs but maintaining transformer capabilities.

4. **Linear Computational Complexity**: Achieves O(HW) complexity with respect to input
   size, making it suitable for high-resolution images and dense prediction tasks.

Mathematical Formulation
-----------------------

Each SwinTransformerBlock performs the following operations:

```
ẑˡ = W-MSA(LN(zˡ⁻¹)) + zˡ⁻¹                    (for regular blocks)
ẑˡ = SW-MSA(LN(zˡ⁻¹)) + zˡ⁻¹                   (for shifted blocks)
zˡ = MLP(LN(ẑˡ)) + ẑˡ
```

Where:
- zˡ represents the output features of block l
- LN denotes LayerNormalization
- W-MSA/SW-MSA are (shifted) windowed multi-head self-attention
- MLP is a two-layer feed-forward network with GELU activation

Window Partitioning and Merging
-------------------------------

The attention computation operates on windows of size M×M:
1. **Partition**: Input (H, W, C) → (⌈H/M⌉×⌈W/M⌉, M, M, C)
2. **Attention**: Applied within each M×M window independently
3. **Merge**: Reconstruct to original spatial dimensions (H, W, C)

For shifted windows, cyclic shifts are applied before partitioning and reversed after
attention computation, enabling cross-window communication.

Performance Characteristics
--------------------------

Computational Complexity:
- Traditional ViT: O(4hwC² + 2(hw)²C) per block
- Swin Transformer: O(4hwC² + 2M²hwC) per block
- Memory efficient for large images due to fixed window size M

Typical configurations:
- Swin-T: embed_dim=96, depths=[2,2,6,2], num_heads=[3,6,12,24]
- Swin-S: embed_dim=96, depths=[2,2,18,2], num_heads=[3,6,12,24]
- Swin-B: embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32]
- Swin-L: embed_dim=192, depths=[2,2,18,2], num_heads=[6,12,24,48]

Usage Example
------------

```python
# Basic Swin Transformer block
block = SwinTransformerBlock(
    dim=96,
    num_heads=3,
    window_size=7,
    shift_size=0,  # Regular window attention
    mlp_ratio=4.0
)

# Shifted window variant (typical for odd-numbered layers)
shifted_block = SwinTransformerBlock(
    dim=96,
    num_heads=3,
    window_size=7,
    shift_size=3,  # Shifted window attention
    drop_path=0.1  # Stochastic depth
)

# Process 4D image tensor
inputs = keras.Input(shape=(224, 224, 96))
outputs = block(inputs)
```

References
----------

- Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S. and Guo, B., 2021.
  "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
  arXiv preprint arXiv:2103.14030.
  https://arxiv.org/abs/2103.14030

- ICCV 2021 Best Paper Award Winner
"""

import keras
from keras import ops, initializers, regularizers
from typing import Tuple, Optional, Dict, Any, Union, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.tensors import window_reverse, window_partition

from ..ffn import SwinMLP
from ..stochastic_depth import StochasticDepth
from ..attention.window_attention import WindowAttention


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class SwinTransformerBlock(keras.layers.Layer):
    """
    Swin Transformer Block with windowed multi-head self-attention.

    This layer implements a Swin Transformer block that performs windowed multi-head
    self-attention followed by an MLP. It supports shifted windows for cross-window
    connections and includes optional stochastic depth regularization.

    **Intent**: Implement the core building block of Swin Transformer architecture
    for efficient vision_heads transformers with linear computational complexity relative
    to input size through windowed attention mechanisms.

    **Architecture**:
    ```
    Input(shape=[batch, height, width, channels])
           ↓
    LayerNorm → Window Partition → WindowAttention → Window Merge
           ↓                                              ↓
    StochasticDepth ←------------------------------------ ←
           ↓
    Residual Connection (+)
           ↓
    LayerNorm → SwinMLP
           ↓         ↓
    StochasticDepth ←
           ↓
    Residual Connection (+)
           ↓
    Output(shape=[batch, height, width, channels])
    ```

    **Key Features**:
    - Window-based attention for O(H*W) computational complexity
    - Optional shifted windows for cross-window connections
    - Stochastic depth regularization for training deep networks
    - Configurable MLP expansion ratio

    **Mathematical Operations**:
    1. **Attention Block**: x' = x + DropPath(WindowAttn(LN(x)))
    2. **MLP Block**: y = x' + DropPath(MLP(LN(x')))
    3. **Window Partition**: Splits (H, W) into non-overlapping windows
    4. **Shifted Windows**: Cyclically shifts windows by shift_size pixels

    Args:
        dim: Integer, number of input channels. Must be positive and should
            match the last dimension of input tensors.
        num_heads: Integer, number of attention heads. Must be positive and
            divide dim evenly for proper head dimension calculation.
        window_size: Integer, window size for attention computation. Must be positive.
            Common values are 7 or 8. Defaults to 8.
        shift_size: Integer, shift size for shifted window attention. Must be
            non-negative and less than window_size. Use window_size//2 for
            standard shifted windows. Defaults to 0 (no shifting).
        mlp_ratio: Float, ratio of MLP hidden dimension to embedding dimension.
            Must be positive. Standard value is 4.0. Defaults to 4.0.
        qkv_bias: Boolean, whether to add learnable bias to query, key, value
            projections in attention. Defaults to True.
        dropout_rate: Float, dropout rate for MLP and attention projection.
            Must be in [0, 1). Defaults to 0.0.
        attn_dropout_rate: Float, dropout rate for attention weights.
            Must be in [0, 1). Defaults to 0.0.
        drop_path: Float, stochastic depth rate for DropPath regularization.
            Must be in [0, 1). Higher values increase regularization. Defaults to 0.0.
        activation: String name or callable, activation function for MLP.
            Common choices: 'gelu', 'relu', 'swish'. Defaults to "gelu".
        use_bias: Boolean, whether to use bias in layer normalization and
            linear projections. Defaults to True.
        kernel_initializer: String name or Initializer instance for kernel weights.
            Defaults to "glorot_uniform".
        bias_initializer: String name or Initializer instance for bias vectors.
            Defaults to "zeros".
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias vectors.
        activity_regularizer: Optional regularizer for layer output activations.
        **kwargs: Additional keyword arguments for Layer base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`
        - height and width should be divisible by window_size for optimal performance
        - channels must equal the dim parameter

    Output shape:
        4D tensor with shape: `(batch_size, height, width, channels)`
        - Shape is preserved through the transformation

    Attributes:
        norm1: First LayerNormalization layer (pre-attention).
        attn: WindowAttention layer for windowed multi-head attention.
        drop_path_layer: StochasticDepth layer for regularization (if drop_path > 0).
        norm2: Second LayerNormalization layer (pre-MLP).
        mlp: SwinMLP layer for feed-forward processing.

    Example:
        ```python
        # Basic Swin Transformer block
        block = SwinTransformerBlock(dim=96, num_heads=3, window_size=7)
        inputs = keras.Input(shape=(224, 224, 96))
        outputs = block(inputs)

        # With shifted windows (typical for even-numbered layers)
        shifted_block = SwinTransformerBlock(
            dim=192,
            num_heads=6,
            window_size=7,
            shift_size=3,  # Half of window_size
            drop_path=0.1
        )

        # Deep network with regularization
        deep_block = SwinTransformerBlock(
            dim=384,
            num_heads=12,
            window_size=8,
            mlp_ratio=6.0,
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
            drop_path=0.2,
            activation='swish'
        )

        # In a complete Swin Transformer stage
        stage_blocks = []
        for i in range(6):  # 6 blocks per stage
            shift_size = 0 if i % 2 == 0 else window_size // 2
            block = SwinTransformerBlock(
                dim=192,
                num_heads=6,
                window_size=7,
                shift_size=shift_size,
                drop_path=0.1 * i / 6  # Increasing stochastic depth
            )
            stage_blocks.append(block)
        ```

    Raises:
        ValueError: If dim <= 0, num_heads <= 0, dim not divisible by num_heads,
            window_size <= 0, shift_size < 0, shift_size >= window_size,
            mlp_ratio <= 0, or dropout rates not in [0, 1).

    References:
        - Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
          (Liu et al., 2021): https://arxiv.org/abs/2103.14030
        - Official implementation: https://github.com/microsoft/Swin-Transformer

    Note:
        This implementation uses the window partitioning utilities from utils.tensors
        and integrates with the dl-techniques framework's SwinMLP and WindowAttention
        components for optimal performance and consistency.
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
        activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = "gelu",
        use_bias: bool = True,
        kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
        activity_regularizer: Optional[Union[str, regularizers.Regularizer]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(
            activity_regularizer=activity_regularizer,
            **kwargs
        )

        # Comprehensive input validation
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads}). "
                f"Got head_dim={dim // num_heads} with remainder {dim % num_heads}"
            )
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if shift_size < 0:
            raise ValueError(f"shift_size must be non-negative, got {shift_size}")
        if shift_size >= window_size:
            raise ValueError(
                f"shift_size ({shift_size}) must be less than window_size ({window_size})"
            )
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if not (0 <= dropout_rate < 1):
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
        if not (0 <= attn_dropout_rate < 1):
            raise ValueError(f"attn_dropout_rate must be in [0, 1), got {attn_dropout_rate}")
        if not (0 <= drop_path < 1):
            raise ValueError(f"drop_path must be in [0, 1), got {drop_path}")

        # Store ALL configuration parameters for serialization
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

        # Store and serialize initializers and regularizers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        # CREATE all sub-layers in __init__ (they are unbuilt)
        # Following Pattern 2: Composite Layer from the guide

        # Layer normalization layers
        self.norm1 = keras.layers.LayerNormalization(
            epsilon=1e-5,
            center=self.use_bias,
            scale=True,  # Always use scale parameter
            beta_initializer=self.bias_initializer if self.use_bias else "zeros",
            gamma_initializer="ones",
            beta_regularizer=self.bias_regularizer if self.use_bias else None,
            gamma_regularizer=None,  # Typically don't regularize scale parameters
            name="norm1"
        )

        self.norm2 = keras.layers.LayerNormalization(
            epsilon=1e-5,
            center=self.use_bias,
            scale=True,  # Always use scale parameter
            beta_initializer=self.bias_initializer if self.use_bias else "zeros",
            gamma_initializer="ones",
            beta_regularizer=self.bias_regularizer if self.use_bias else None,
            gamma_regularizer=None,  # Typically don't regularize scale parameters
            name="norm2"
        )

        # Window attention layer
        self.attn = WindowAttention(
            dim=self.dim,
            window_size=self.window_size,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            dropout_rate=self.attn_dropout_rate,
            proj_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            partition_mode="grid",
            normalization="softmax",
            attention_mode="linear",
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
            output_dim=self.dim,  # Explicit output dimension
            use_bias=self.use_bias,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="mlp"
        )

        logger.debug(
            f"Initialized SwinTransformerBlock: dim={dim}, num_heads={num_heads}, "
            f"window_size={window_size}, shift_size={shift_size}, "
            f"mlp_ratio={mlp_ratio}, drop_path={drop_path}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
        This ensures all weight variables exist before weight restoration during
        model loading. Without this, sub-layers would be unbuilt and weight
        loading would fail.

        Args:
            input_shape: Shape tuple of the input tensor (batch, height, width, channels).

        Raises:
            ValueError: If input_shape is not 4D or channels dimension doesn't match dim.
        """
        if len(input_shape) != 4:
            raise ValueError(
                f"SwinTransformerBlock expects 4D input (batch, height, width, channels), "
                f"got shape {input_shape}"
            )

        if input_shape[-1] is not None and input_shape[-1] != self.dim:
            raise ValueError(
                f"Input channels ({input_shape[-1]}) must match dim ({self.dim})"
            )

        # Build sub-layers in computational order following the forward pass

        # 1. Build normalization layers (operate on original input shape)
        self.norm1.build(input_shape)
        self.norm2.build(input_shape)

        # 2. Build attention layer with windowed input shape
        # After window partitioning: (batch * num_windows, window_size^2, channels)
        windowed_shape = (None, self.window_size * self.window_size, self.dim)
        self.attn.build(windowed_shape)

        # 3. Build stochastic depth layer if it exists (operates on original shape)
        if self.drop_path_layer is not None:
            self.drop_path_layer.build(input_shape)

        # 4. Build MLP layer (operates on original input shape)
        self.mlp.build(input_shape)

        # Always call parent build at the end
        super().build(input_shape)

        logger.debug(
            f"Built SwinTransformerBlock: input_shape={input_shape}, "
            f"windowed_shape={windowed_shape}"
        )

    def call(
        self,
        x: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the SwinTransformerBlock layer.

        Implements the standard Swin Transformer block computation:
        1. Attention block: x = x + DropPath(WindowAttn(LN(x)))
        2. MLP block: x = x + DropPath(MLP(LN(x)))

        Args:
            x: Input tensor of shape (batch_size, height, width, channels).
            training: Boolean indicating whether the layer should behave in
                training mode (apply dropout/stochastic depth) or inference mode.

        Returns:
            Output tensor of shape (batch_size, height, width, channels).
            Values represent the transformed features after attention and MLP processing.

        Raises:
            ValueError: If input tensor is not 4D.
        """
        input_shape = ops.shape(x)
        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D input shape, got shape {input_shape}"
            )

        B, H, W, C = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        shortcut = x

        # =============================================
        # Multi-head Self-Attention Block
        # =============================================

        # Layer Norm 1 (pre-attention normalization)
        x = self.norm1(x, training=training)

        # Apply cyclic shift for shifted window attention
        if self.shift_size > 0:
            # Shift windows by (-shift_size, -shift_size)
            shifted_x = ops.roll(
                x,
                shift=(-self.shift_size, -self.shift_size),
                axis=(1, 2)
            )
        else:
            shifted_x = x

        # Partition into windows: (B, H, W, C) -> (B*num_windows, window_size, window_size, C)
        x_windows = window_partition(shifted_x, self.window_size)

        # Reshape for attention: (B*num_windows, window_size*window_size, C)
        x_windows = ops.reshape(
            x_windows,
            (-1, self.window_size * self.window_size, C)
        )

        # Apply window-based multi-head self-attention
        attn_windows = self.attn(x_windows, training=training)

        # Reshape back to window format: (B*num_windows, window_size, window_size, C)
        attn_windows = ops.reshape(
            attn_windows,
            (-1, self.window_size, self.window_size, C)
        )

        # Merge windows back: (B*num_windows, window_size, window_size, C) -> (B, H, W, C)
        x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift if it was applied
        if self.shift_size > 0:
            x = ops.roll(
                x,
                shift=(self.shift_size, self.shift_size),
                axis=(1, 2)
            )

        # Apply stochastic depth and residual connection
        if self.drop_path_layer is not None:
            x = shortcut + self.drop_path_layer(x, training=training)
        else:
            x = shortcut + x

        # =============================================
        # MLP Block
        # =============================================

        shortcut = x

        # Layer Norm 2 (pre-MLP normalization)
        x = self.norm2(x, training=training)

        # Apply MLP transformation
        x = self.mlp(x, training=training)

        # Apply stochastic depth and residual connection
        if self.drop_path_layer is not None:
            x = shortcut + self.drop_path_layer(x, training=training)
        else:
            x = shortcut + x

        return x

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        SwinTransformerBlock preserves the input shape through all transformations.
        The window partitioning and attention operations maintain spatial dimensions.

        Args:
            input_shape: Shape of the input (batch, height, width, channels).

        Returns:
            Output shape tuple (batch, height, width, channels).
            Shape is identical to input_shape.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        CRITICAL: Must include ALL __init__ parameters for proper serialization.
        This ensures the layer can be reconstructed exactly when loading a saved model.

        Returns:
            Dictionary containing all layer configuration parameters.
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
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
        })
        return config

# ---------------------------------------------------------------------
