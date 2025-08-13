"""
# WindowAttention Layer

A Keras layer implementing windowed multi-head self-attention as described in the Swin Transformer
paper. This layer partitions input tokens into non-overlapping windows and computes self-attention
within each window, reducing computational complexity while maintaining effective modeling capacity.

## Conceptual Overview

Standard transformer attention computes relationships between all pairs of tokens in a sequence,
leading to quadratic complexity O(N²) where N is the sequence length. For vision tasks with
high-resolution images, this becomes computationally prohibitive as the number of patches grows.

WindowAttention addresses this by:

1. **Spatial Partitioning**: Dividing the input feature map into non-overlapping windows
2. **Local Attention**: Computing self-attention only within each window
3. **Relative Position Encoding**: Using learnable relative position biases to capture spatial relationships

This reduces complexity from O(N²) to O(W² × N) where W is the window size, making it linear
with respect to image size while quadratic only within small, fixed-size windows.

### Mathematical Description:

For an input tensor of shape (B, H×W, C) representing B images with H×W patches of C channels:

1. **Window Partitioning**: Reshape input into (B×num_windows, window_size², C)
2. **Standard Multi-Head Attention** within each window:
   * Query, Key, Value projections: `Q = XW_q`, `K = XW_k`, `V = XW_v`
   * Attention scores: `A = (QK^T) / √d_head`
   * **Relative Position Bias**: `A = A + B_rel`
   * Attention weights: `P = softmax(A)`
   * Output: `O = PV`

3. **Relative Position Bias Computation**:
   * For each window position pair (i,j), compute relative coordinates: `(Δi, Δj)`
   * Map to bias table index: `idx = (Δi + W-1) × (2W-1) + (Δj + W-1)`
   * Retrieve bias: `B_rel[i,j] = bias_table[idx]`

### Key Benefits:

1. **Linear Complexity**: O(W² × N) complexity scales linearly with image size
2. **Local Modeling**: Effectively captures local spatial relationships within windows
3. **Relative Position Awareness**: Learnable biases encode spatial structure without absolute positions
4. **Hardware Efficient**: Regular window structure enables efficient GPU/TPU computation
5. **Hierarchical Compatibility**: Works with shifted window schemes for cross-window connections

### Usage Example:
```python
# For a 7×7 window with 96 channels and 3 attention heads
window_attn = WindowAttention(dim=96, window_size=7, num_heads=3)

# Input: (batch_size, 49, 96) where 49 = 7×7 window
x = keras.random.normal((4, 49, 96))
output = window_attn(x)  # Shape: (4, 49, 96)
```
"""

import keras
from keras import ops
from typing import Tuple, Optional, Dict, Any, Union

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class WindowAttention(keras.layers.Layer):
    """Window Multi-head Self-Attention module for Swin Transformer.

    Implements windowed multi-head self-attention with relative position bias
    as described in the Swin Transformer paper. This layer follows modern Keras 3
    patterns for robust serialization and building.

    Args:
        dim: Input dimension/number of input channels. Must be positive.
        window_size: Size of attention window (both height and width). Must be positive.
        num_heads: Number of attention heads. Must be positive and divide dim evenly.
        qkv_bias: Whether to use bias in qkv projection. Defaults to True.
        qk_scale: Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
        attn_dropout_rate: Attention dropout rate. Must be between 0.0 and 1.0. Defaults to 0.0.
        proj_dropout_rate: Output projection dropout rate. Must be between 0.0 and 1.0. Defaults to 0.0.
        proj_bias: Whether to use bias in output projection. Defaults to True.
        kernel_initializer: Initializer for dense layer kernels. Defaults to "glorot_uniform".
        bias_initializer: Initializer for dense layer biases. Defaults to "zeros".
        kernel_regularizer: Optional regularizer for dense layer kernels. Defaults to None.
        bias_regularizer: Optional regularizer for dense layer biases. Defaults to None.
        **kwargs: Additional keyword arguments for Layer base class.

    Raises:
        ValueError: If dim is not positive.
        ValueError: If dim is not divisible by num_heads.
        ValueError: If window_size is not positive.
        ValueError: If num_heads is not positive.
        ValueError: If dropout rates are not between 0.0 and 1.0.

    Input shape:
        A 3D tensor with shape: `(batch_size, num_tokens_in_window, dim)`
        where num_tokens_in_window = window_size * window_size

    Output shape:
        A 3D tensor with shape: `(batch_size, num_tokens_in_window, dim)`

    Attributes:
        qkv: Dense layer for query, key, value projections.
        proj: Dense layer for output projection.
        attn_dropout: Dropout layer for attention weights (if dropout_rate > 0).
        proj_dropout: Dropout layer for output projection (if dropout_rate > 0).
        relative_position_bias_table: Learnable relative position bias parameters.
        relative_position_index: Non-trainable relative position indices.

    Example:
        ```python
        # Basic usage
        window_attn = WindowAttention(dim=96, window_size=7, num_heads=3)

        # Advanced configuration with regularization
        window_attn = WindowAttention(
            dim=384,
            window_size=7,
            num_heads=12,
            qkv_bias=True,
            attn_dropout_rate=0.1,
            proj_dropout_rate=0.1,
            kernel_regularizer=keras.regularizers.L2(1e-4)
        )

        # Input with batch_size=4, window_size=7x7=49, dim=96
        x = keras.random.normal((4, 49, 96))
        output = window_attn(x)  # Shape: (4, 49, 96)
        ```

    References:
        - Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.
          Liu et al., ICCV 2021.
        - https://arxiv.org/abs/2103.14030
    """

    def __init__(
            self,
            dim: int,
            window_size: int,
            num_heads: int,
            qkv_bias: bool = True,
            qk_scale: Optional[float] = None,
            attn_dropout_rate: float = 0.0,
            proj_dropout_rate: float = 0.0,
            proj_bias: bool = True,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validation
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if not (0.0 <= attn_dropout_rate <= 1.0):
            raise ValueError(f"attn_dropout_rate must be between 0.0 and 1.0, got {attn_dropout_rate}")
        if not (0.0 <= proj_dropout_rate <= 1.0):
            raise ValueError(f"proj_dropout_rate must be between 0.0 and 1.0, got {proj_dropout_rate}")

        # Store ALL configuration parameters
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_scale = qk_scale  # Store original parameter
        self.scale = qk_scale if qk_scale is not None else self.head_dim ** -0.5
        self.qkv_bias = qkv_bias
        self.proj_bias = proj_bias
        self.attn_dropout_rate = attn_dropout_rate
        self.proj_dropout_rate = proj_dropout_rate

        # Store initializers and regularizers
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # CREATE all sub-layers in __init__ (following modern Keras 3 pattern)
        self.qkv = keras.layers.Dense(
            self.dim * 3,
            use_bias=self.qkv_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="qkv"
        )

        self.proj = keras.layers.Dense(
            self.dim,
            use_bias=self.proj_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="proj"
        )

        # Create dropout layers if needed
        self.attn_dropout = None
        if self.attn_dropout_rate > 0.0:
            self.attn_dropout = keras.layers.Dropout(
                self.attn_dropout_rate,
                name="attn_dropout"
            )

        self.proj_dropout = None
        if self.proj_dropout_rate > 0.0:
            self.proj_dropout = keras.layers.Dropout(
                self.proj_dropout_rate,
                name="proj_dropout"
            )

        # Initialize weight attributes - created in build()
        self.relative_position_bias_table = None
        self.relative_position_index = None

    def _create_relative_position_index(self) -> None:
        """Create relative position index for relative position bias.

        This creates a lookup table that maps each pair of positions within
        a window to their relative position bias index.
        """
        # Get pair-wise relative position index
        coords_h = ops.arange(0, self.window_size, dtype="int32")
        coords_w = ops.arange(0, self.window_size, dtype="int32")
        coords = ops.stack(ops.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = ops.reshape(coords, (2, -1))  # 2, Wh*Ww

        # Compute relative coordinates
        relative_coords = (
            ops.expand_dims(coords_flatten, 2) -
            ops.expand_dims(coords_flatten, 1)
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = ops.transpose(relative_coords, (1, 2, 0))  # Wh*Ww, Wh*Ww, 2

        # Shift relative coordinates to make them positive
        relative_coords_h = relative_coords[:, :, 0] + self.window_size - 1
        relative_coords_w = relative_coords[:, :, 1] + self.window_size - 1
        relative_coords_h = relative_coords_h * (2 * self.window_size - 1)
        relative_position_index = relative_coords_h + relative_coords_w

        self.relative_position_index = relative_position_index

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and all its sub-layers.

        This method creates the layer's own weights (relative position bias table)
        and explicitly builds all sub-layers for robust serialization.

        Args:
            input_shape: Shape tuple of the input tensor.
        """
        # Build sub-layers explicitly (critical for serialization)
        self.qkv.build(input_shape)

        # Compute qkv output shape for proj layer building
        qkv_output_shape = self.qkv.compute_output_shape(input_shape)
        # proj takes the same input shape as original input (after attention computation)
        self.proj.build(input_shape)

        # Build dropout layers if they exist
        if self.attn_dropout is not None:
            # Attention dropout operates on attention weights: (B, num_heads, N, N)
            attn_shape = (input_shape[0], self.num_heads, input_shape[1], input_shape[1])
            self.attn_dropout.build(attn_shape)

        if self.proj_dropout is not None:
            # Projection dropout operates on the same shape as input
            self.proj_dropout.build(input_shape)

        # Create the layer's own weights (relative position bias table)
        num_relative_distance = (2 * self.window_size - 1) * (2 * self.window_size - 1)
        self.relative_position_bias_table = self.add_weight(
            name="relative_position_bias_table",
            shape=(num_relative_distance, self.num_heads),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True
        )

        # Create relative position index (non-trainable)
        self._create_relative_position_index()

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        x: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the WindowAttention layer.

        Args:
            x: Input tensor of shape (B, N, C) where N = window_size * window_size.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Output tensor of shape (B, N, C).

        Raises:
            ValueError: If input sequence length does not match window_size².
        """
        B = ops.shape(x)[0]
        N = ops.shape(x)[1]
        C = ops.shape(x)[2]

        # Validate that the input sequence length matches the window size squared
        expected_n = self.window_size * self.window_size
        if N != expected_n:
            raise ValueError(
                f"Input sequence length ({N}) does not match the square of the window_size "
                f"({self.window_size}^2 = {expected_n}). WindowAttention requires "
                f"the input to be a single window of tokens."
            )

        # Generate qkv matrices
        qkv = self.qkv(x, training=training)  # (B, N, 3*C)
        qkv = ops.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, num_heads, N, head_dim)

        # Separate q, k, v
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each has shape (B, num_heads, N, head_dim)

        # Scale query
        q = q * self.scale

        # Compute attention scores
        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))  # (B, num_heads, N, N)

        # Add relative position bias
        relative_position_bias = ops.take(
            self.relative_position_bias_table,
            ops.reshape(self.relative_position_index, (-1,)),
            axis=0
        )
        relative_position_bias = ops.reshape(
            relative_position_bias,
            (self.window_size * self.window_size, self.window_size * self.window_size, -1)
        )
        relative_position_bias = ops.transpose(
            relative_position_bias, (2, 0, 1)
        )  # (num_heads, N, N)
        attn = attn + ops.expand_dims(relative_position_bias, 0)  # (B, num_heads, N, N)

        # Apply softmax
        attn = ops.softmax(attn, axis=-1)

        # Apply attention dropout
        if self.attn_dropout is not None:
            attn = self.attn_dropout(attn, training=training)

        # Apply attention to values
        x = ops.matmul(attn, v)  # (B, num_heads, N, head_dim)
        x = ops.transpose(x, (0, 2, 1, 3))  # (B, N, num_heads, head_dim)
        x = ops.reshape(x, (B, N, C))  # (B, N, C)

        # Output projection
        x = self.proj(x, training=training)
        if self.proj_dropout is not None:
            x = self.proj_dropout(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple. Same as input shape for attention layers.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer configuration for serialization.

        This method returns ALL configuration parameters needed to recreate
        the layer, following modern Keras 3 serialization practices.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "window_size": self.window_size,
            "num_heads": self.num_heads,
            "qkv_bias": self.qkv_bias,
            "qk_scale": self.qk_scale,  # Store original parameter, not computed scale
            "attn_dropout_rate": self.attn_dropout_rate,
            "proj_dropout_rate": self.proj_dropout_rate,
            "proj_bias": self.proj_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------