"""
WindowAttention Layer

A Keras layer implementing windowed multi-head self-attention as described in the Swin Transformer
paper. This layer partitions input tokens into non-overlapping windows and computes self-attention
within each window, reducing computational complexity while maintaining effective modeling capacity.

This version is enhanced to support input sequences shorter than the window's capacity
(window_size * window_size) by automatically padding the sequence for computation and
removing the padding from the output.

## Conceptual Overview

Standard transformer attention computes relationships between all pairs of tokens in a sequence,
leading to quadratic complexity O(N²) where N is the sequence length. For vision tasks with
high-resolution images, this becomes computationally prohibitive as the number of patches grows.

WindowAttention addresses this by:

1. **Spatial Partitioning**: Dividing the input feature map into non-overlapping windows
2. **Local Attention**: Computing self-attention only within each window
3. **Relative Position Encoding**: Using learnable relative position biases to capture spatial relationships
4. **Flexible Input Handling**: Automatically pads sequences shorter than the window area to enable
   computation on non-square or incomplete windows, then un-pads the output.

This reduces complexity from O(N²) to O(W² × N) where W is the window size, making it linear
with respect to image size while quadratic only within small, fixed-size windows.

### Mathematical Description:

For an input tensor of shape (B, N_actual, C) where N_actual <= window_size²:

1. **Padding (if N_actual < window_size²)**: Pad input and create a mask.
2. **Window Partitioning**: Reshape input into (B×num_windows, window_size², C)
3. **Standard Multi-Head Attention** within each window:
   * Query, Key, Value projections: `Q = XW_q`, `K = XW_k`, `V = XW_v`
   * Attention scores: `A = (QK^T) / √d_head`
   * **Relative Position Bias**: `A = A + B_rel`
   * Attention weights: `P = softmax(A + Mask)`
   * Output: `O = PV`
4. **Un-padding**: Slice output back to original sequence length N_actual.

### Key Benefits:

1. **Linear Complexity**: O(W² × N) complexity scales linearly with image size
2. **Local Modeling**: Effectively captures local spatial relationships within windows
3. **Relative Position Awareness**: Learnable biases encode spatial structure without absolute positions
4. **Flexible**: Handles incomplete windows (e.g., at image edges) via padding.
5. **Hardware Efficient**: Regular window structure enables efficient GPU/TPU computation

### Usage Example:
```python
# For a 7×7 window with 96 channels and 3 attention heads
window_attn = WindowAttention(dim=96, window_size=7, num_heads=3)

# Input for a full window: (batch_size, 49, 96)
x_full = keras.random.normal((4, 49, 96))
output_full = window_attn(x_full)  # Shape: (4, 49, 96)

# Input for a partial window (e.g., 20 tokens): (batch_size, 20, 96)
x_partial = keras.random.normal((4, 20, 96))
output_partial = window_attn(x_partial) # Shape: (4, 20, 96)
```
"""

import keras
from typing import Tuple, Optional, Dict, Any, Union

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class WindowAttention(keras.layers.Layer):
    """Window Multi-head Self-Attention module for Swin Transformer.

    Implements windowed multi-head self-attention with relative position bias
    as described in the Swin Transformer paper. This layer follows modern Keras 3
    patterns for robust serialization and building.

    **Intent**: Provide efficient windowed self-attention that scales linearly with
    image size while maintaining local spatial modeling capacity through relative
    position encoding. This implementation robustly handles input sequences that are
    shorter than the window's capacity (`window_size**2`) by padding during computation.

    **Architecture**:
    ```
    Input(B, N_actual, C) where N_actual <= window_size²
           ↓
    Pad to (B, N_target, C) if needed, where N_target = window_size²
           ↓
    QKV Projection: Linear(C → 3C)
           ↓
    Reshape: (B, N_target, 3, num_heads, head_dim)
           ↓
    Multi-Head Attention with Relative Position Bias and Padding Mask
           ↓
    Output Projection: Linear(C → C)
           ↓
    Un-pad to (B, N_actual, C)
           ↓
    Output(B, N_actual, C)
    ```

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
        ValueError: In `call`, if input sequence length exceeds `window_size**2`.

    Input shape:
        A 3D tensor with shape: `(batch_size, num_tokens, dim)`
        where `num_tokens` must be less than or equal to `window_size * window_size`.

    Output shape:
        A 3D tensor with the same shape as the input: `(batch_size, num_tokens, dim)`.

    Attributes:
        qkv: Dense layer for query, key, value projections.
        proj: Dense layer for output projection.
        attn_dropout: Dropout layer for attention weights (if dropout_rate > 0).
        proj_dropout: Dropout layer for output projection (if dropout_rate > 0).
        relative_position_bias_table: Learnable relative position bias parameters.
        relative_position_index: Non-trainable, constant relative position indices.

    Example:
        ```python
        # Basic usage with a full window
        window_attn = WindowAttention(dim=96, window_size=7, num_heads=3)
        x_full = keras.random.normal((4, 49, 96))
        output_full = window_attn(x_full)  # Shape: (4, 49, 96)

        # Usage with a partial window (e.g., 30 tokens)
        x_partial = keras.random.normal((4, 30, 96))
        output_partial = window_attn(x_partial) # Shape: (4, 30, 96)
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

        # --- Validation ---
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

        # --- Store ALL configuration parameters for serialization ---
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_scale = qk_scale
        self.scale = qk_scale if qk_scale is not None else self.head_dim ** -0.5
        self.qkv_bias = qkv_bias
        self.proj_bias = proj_bias
        self.attn_dropout_rate = attn_dropout_rate
        self.proj_dropout_rate = proj_dropout_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # --- CREATE all sub-layers in __init__ (following modern Keras 3 pattern) ---
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

        # Create dropout layers only if needed
        self.attn_dropout = (
            keras.layers.Dropout(self.attn_dropout_rate, name="attn_dropout")
            if self.attn_dropout_rate > 0.0 else None
        )

        self.proj_dropout = (
            keras.layers.Dropout(self.proj_dropout_rate, name="proj_dropout")
            if self.proj_dropout_rate > 0.0 else None
        )

        # --- Pre-compute constant relative position indices in __init__ ---
        # This is constant state that doesn't depend on input shape
        coords_h = keras.ops.arange(self.window_size, dtype="int32")
        coords_w = keras.ops.arange(self.window_size, dtype="int32")
        coords = keras.ops.stack(keras.ops.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = keras.ops.reshape(coords, (2, -1))  # 2, Wh*Ww

        # Relative coordinates: 2, Wh*Ww, Wh*Ww
        relative_coords = keras.ops.expand_dims(coords_flatten, 2) - keras.ops.expand_dims(coords_flatten, 1)
        relative_coords = keras.ops.transpose(relative_coords, (1, 2, 0))  # Wh*Ww, Wh*Ww, 2

        # Shift to make coordinates positive and create unique indices
        relative_coords_h = relative_coords[:, :, 0] + self.window_size - 1  # 0 to 2*Wh-2
        relative_coords_w = relative_coords[:, :, 1] + self.window_size - 1  # 0 to 2*Ww-2
        relative_coords_h *= (2 * self.window_size - 1)

        # Store the computed indices as a constant tensor attribute
        self.relative_position_index = relative_coords_h + relative_coords_w

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer weights and sub-layers following modern Keras 3 patterns.

        CRITICAL: Explicitly build each sub-layer for robust serialization.
        This ensures all weight variables exist before weight restoration during
        model loading.
        """
        # --- CREATE the layer's own trainable weights ---
        num_relative_positions = (2 * self.window_size - 1) ** 2
        self.relative_position_bias_table = self.add_weight(
            name="relative_position_bias_table",
            shape=(num_relative_positions, self.num_heads),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            dtype=self.dtype
        )

        # --- Explicitly BUILD all sub-layers in computational order ---
        # This is CRITICAL for robust serialization in Keras 3
        # We build with the *padded* shape to ensure layers can handle full window size
        padded_shape = list(input_shape)
        padded_shape[1] = self.window_size * self.window_size

        # 1. Build QKV projection
        self.qkv.build(padded_shape)

        # 2. Build output projection
        self.proj.build(padded_shape)

        # 3. Build dropout layers if they exist
        if self.attn_dropout is not None:
            self.attn_dropout.build(None)  # Dropout doesn't need specific input shape

        if self.proj_dropout is not None:
            self.proj_dropout.build(padded_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass of the WindowAttention layer.

        Args:
            inputs: Input tensor of shape (B, N_actual, C) where N_actual <= window_size²
            attention_mask: Optional attention mask of shape (B, N_actual) where 1=attend, 0=mask.
                This mask is merged with the internal padding mask if padding is required.
            training: Whether in training mode (for dropout)

        Returns:
            Output tensor of shape (B, N_actual, C)
        """
        B_actual, N_actual, C_actual = keras.ops.shape(inputs)
        N_target = self.window_size * self.window_size

        if N_actual > N_target:
            raise ValueError(
                f"Input sequence length ({N_actual}) cannot be greater than the "
                f"window area ({N_target})."
            )

        # --- START PADDING LOGIC ---
        padded_inputs = inputs
        padding_mask = None
        if N_actual < N_target:
            padding_amount = N_target - N_actual
            # Pad inputs with zeros to match the window size
            padding_tensor = keras.ops.zeros((B_actual, padding_amount, C_actual), dtype=inputs.dtype)
            padded_inputs = keras.ops.concatenate([inputs, padding_tensor], axis=1)

            # Create an internal mask to ignore padded tokens
            padding_mask = keras.ops.concatenate([
                keras.ops.ones((B_actual, N_actual), dtype="int32"),
                keras.ops.zeros((B_actual, padding_amount), dtype="int32")
            ], axis=1)

        # Merge the internal padding mask with any user-provided mask
        if attention_mask is not None:
            # Pad user mask to target size if necessary
            if keras.ops.shape(attention_mask)[1] < N_target:
                 padding_amount_mask = N_target - keras.ops.shape(attention_mask)[1]
                 mask_padding = keras.ops.zeros((B_actual, padding_amount_mask), dtype=attention_mask.dtype)
                 attention_mask = keras.ops.concatenate([attention_mask, mask_padding], axis=1)

            if padding_mask is not None:
                # Combine masks using multiplication (logical AND)
                attention_mask = keras.ops.cast(attention_mask, dtype="int32") * padding_mask
            # If no padding_mask, user mask is used directly
        elif padding_mask is not None:
            attention_mask = padding_mask
        # If both are None, attention_mask remains None

        # --- END PADDING LOGIC ---

        B, N, C = keras.ops.shape(padded_inputs) # Now B, N_target, C

        # QKV projection and reshape
        qkv = self.qkv(padded_inputs, training=training)  # (B, N, 3*C)
        qkv = keras.ops.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim))  # (B, N, 3, num_heads, head_dim)
        qkv = keras.ops.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, N, head_dim)

        # Scale query
        q = q * self.scale

        # Compute attention scores
        attn = keras.ops.matmul(q, keras.ops.transpose(k, (0, 1, 3, 2)))  # (B, num_heads, N, N)

        # Add relative position bias
        relative_position_bias = keras.ops.take(
            self.relative_position_bias_table,
            keras.ops.reshape(self.relative_position_index, (-1,)),
            axis=0
        )  # (N*N, num_heads)

        relative_position_bias = keras.ops.reshape(
            relative_position_bias, (N, N, -1)
        )  # (N, N, num_heads)

        relative_position_bias = keras.ops.transpose(relative_position_bias, (2, 0, 1))  # (num_heads, N, N)
        attn = attn + keras.ops.expand_dims(relative_position_bias, 0)  # (B, num_heads, N, N)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Reshape mask for broadcasting: (B, 1, 1, N)
            broadcast_mask = keras.ops.reshape(attention_mask, (B, 1, 1, N))
            # Convert mask to additive form: 1 -> 0, 0 -> -inf
            inf_value = keras.ops.convert_to_tensor(-1e9, dtype=attn.dtype)
            additive_mask = (1.0 - keras.ops.cast(broadcast_mask, dtype=attn.dtype)) * inf_value
            attn = attn + additive_mask

        # Apply softmax
        attn = keras.ops.softmax(attn, axis=-1)  # (B, num_heads, N, N)

        # Apply attention dropout
        if self.attn_dropout is not None:
            attn = self.attn_dropout(attn, training=training)

        # Apply attention to values
        x = keras.ops.matmul(attn, v)  # (B, num_heads, N, head_dim)
        x = keras.ops.transpose(x, (0, 2, 1, 3))  # (B, N, num_heads, head_dim)
        x = keras.ops.reshape(x, (B, N, C))  # (B, N, C)

        # Output projection
        x = self.proj(x, training=training)

        # Apply projection dropout
        if self.proj_dropout is not None:
            x = self.proj_dropout(x, training=training)

        # --- START UN-PADDING LOGIC ---
        # Remove padding from the output if it was added
        if N_actual < N_target:
            x = x[:, :N_actual, :]
        # --- END UN-PADDING LOGIC ---

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape (identical to input shape)."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer configuration for serialization.

        CRITICAL: Must include ALL constructor parameters for proper serialization.
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "window_size": self.window_size,
            "num_heads": self.num_heads,
            "qkv_bias": self.qkv_bias,
            "qk_scale": self.qk_scale,
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
