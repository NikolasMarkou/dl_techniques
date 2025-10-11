"""
ZigzagWindowAttention Layer

A Keras layer implementing a variant of windowed multi-head self-attention.
This version arranges tokens within the window in a zigzag pattern, inspired by
the coefficient scanning order in JPEG compression.

This modification alters the locality assumption of the attention mechanism. Instead
of a standard row-by-row (raster) scan, the zigzag order prioritizes attention
among tokens that are close in the frequency domain (assuming a DCT-like transform
preceded this layer), grouping low-frequency components together.

## Conceptual Overview

Standard WindowAttention processes tokens in a raster-scan order:
(0,0), (0,1), (0,2), ..., (1,0), (1,1), ...

ZigzagWindowAttention processes tokens in a diagonal, zigzag order:
(0,0), (0,1), (1,0), (2,0), (1,1), (0,2), ...

This is achieved by re-calculating the relative position indices to reflect the
zigzag path. The core attention mechanism remains the same, but the learned
relative position biases will now capture relationships based on this new ordering.

### Key Differences & Benefits:

1.  **Frequency-Domain Locality**: If the input tokens represent frequency coefficients
    (like from a DCT), this ordering places perceptually important low-frequency
    coefficients at the beginning of the sequence, allowing them to attend to
    each other more directly.
2.  **Altered Inductive Bias**: It introduces a different inductive bias about which
    tokens are "close" or "related". This may be beneficial for tasks where
    diagonal or frequency-based relationships are more important than simple
    spatial proximity.
3.  **Drop-in Replacement**: It maintains the same interface as a standard
    `WindowAttention` layer and can be used as a drop-in replacement. It also
    retains the ability to handle partial windows via automatic padding.

### Usage Example:
```python
# The user is responsible for arranging the input tokens in zigzag order.
# For example, if you have a 7x7 grid of tokens, flatten it using a
# zigzag mapping before feeding it to the layer.

zigzag_attn = WindowZigZagAttention(dim=96, window_size=7, num_heads=3)

# Input for a full window (49 tokens already in zigzag order)
x_full = keras.random.normal((4, 49, 96))
output_full = zigzag_attn(x_full)  # Shape: (4, 49, 96)

# Input for a partial window (e.g., 20 tokens from the start of the zigzag path)
x_partial = keras.random.normal((4, 20, 96))
output_partial = zigzag_attn(x_partial) # Shape: (4, 20, 96)
```
"""

import keras
from typing import Tuple, Optional, Dict, Any, Union, List


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class WindowZigZagAttention(keras.layers.Layer):
    """Zigzag Window Multi-head Self-Attention.

    This layer implements windowed multi-head self-attention where tokens within
    the window are processed in a zigzag order. This is a variant of the standard
    Swin Transformer `WindowAttention` layer. The key difference lies in the
    calculation of relative position biases, which are indexed based on a
    zigzag path through the window grid.

    **Intent**: To provide an alternative to raster-scan window attention that
    prioritizes relationships between tokens based on a zigzag sequence. This is
    conceptually similar to how JPEG scans DCT coefficients and may be
    advantageous for inputs where frequency-domain locality is important.

    **Architecture**:
    The computational graph is identical to standard multi-head self-attention,
    but with a specialized relative position bias.
    ```
    Input(B, N_actual, C)
        ↓
    Pad to (B, N_target, C) if N_actual < N_target
        ↓
    QKV Projection: Linear(C, 3*C)
        ↓
    Reshape & Transpose Q, K, V for multi-head processing
        ↓
    Scaled Dot-Product: Attention = softmax(Q @ K^T / sqrt(d_k) + Bias) @ V
        ↓
    Relative Position Bias (Zigzag): Bias is added to Q @ K^T scores
        ↓
    Output Projection: Linear(C, C)
        ↓
    Un-pad to (B, N_actual, C)
        ↓
    Output(B, N_actual, C)
    ```

    **Mathematical Operation**:
        Attention(Q, K, V) = SoftMax( (Q K^T / sqrt(d_k)) + B ) V
    Where `B` is the relative position bias table, indexed by pre-computed
    zigzag relative coordinates.

    :param dim: Integer, dimensionality of the input feature space. Must be positive.
    :type dim: int
    :param window_size: Integer, the height and width of the attention window.
                        Must be positive.
    :type window_size: int
    :param num_heads: Integer, number of attention heads. Must be positive and
                      `dim` must be divisible by `num_heads`.
    :type num_heads: int
    :param qkv_bias: Boolean, whether to use a bias term in the QKV projection.
    :type qkv_bias: bool
    :param qk_scale: Optional float, override for the default query scaling factor
                     (1 / sqrt(head_dim)).
    :type qk_scale: Optional[float]
    :param attn_dropout_rate: Float between 0.0 and 1.0. Dropout rate for attention
                              probabilities.
    :type attn_dropout_rate: float
    :param proj_dropout_rate: Float between 0.0 and 1.0. Dropout rate for the final
                              output projection.
    :type proj_dropout_rate: float
    :param proj_bias: Boolean, whether to use a bias term in the output projection.
    :type proj_bias: bool
    :param kernel_initializer: Initializer for the kernel weights of dense layers.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for the bias vectors of dense layers.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Optional regularizer for bias vectors.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param kwargs: Additional keyword arguments for the `keras.layers.Layer` base class.

    :ivar relative_position_bias_table: Trainable weight of shape
        `((2 * window_size - 1) ** 2, num_heads)` storing relative position biases.
    :ivar relative_position_index: Non-trainable buffer storing the pre-computed
        indices for the bias table based on zigzag ordering.
    :ivar qkv: `keras.layers.Dense` layer for Q, K, V projection.
    :ivar proj: `keras.layers.Dense` layer for the output projection.
    :ivar attn_dropout: `keras.layers.Dropout` layer for attention scores.
    :ivar proj_dropout: `keras.layers.Dropout` layer for the final output.

    :raises ValueError: If `dim`, `window_size`, or `num_heads` are not positive.
    :raises ValueError: If `dim` is not divisible by `num_heads`.
    :raises ValueError: If dropout rates are outside the [0.0, 1.0] range.
    :raises ValueError: In `call`, if input `num_tokens` > `window_size**2`.
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
        """Initializes the WindowZigZagAttention layer."""
        super().__init__(**kwargs)

        # --- Configuration Validation ---
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
        # Use provided qk_scale or compute default; stored for get_config
        self.scale = qk_scale if qk_scale is not None else self.head_dim ** -0.5
        self.qkv_bias = qkv_bias
        self.proj_bias = proj_bias
        self.attn_dropout_rate = attn_dropout_rate
        self.proj_dropout_rate = proj_dropout_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # --- CREATE all sub-layers in __init__ (they remain unbuilt) ---
        # This follows the "Golden Rule" of Keras layer implementation.
        self.qkv = keras.layers.Dense(
            self.dim * 3, use_bias=self.qkv_bias,
            kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
            name="qkv"
        )
        self.proj = keras.layers.Dense(
            self.dim, use_bias=self.proj_bias,
            kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer, bias_regularizer=self.bias_regularizer,
            name="proj"
        )
        self.attn_dropout = (
            keras.layers.Dropout(self.attn_dropout_rate, name="attn_dropout")
            if self.attn_dropout_rate > 0.0 else None
        )
        self.proj_dropout = (
            keras.layers.Dropout(self.proj_dropout_rate, name="proj_dropout")
            if self.proj_dropout_rate > 0.0 else None
        )

        # --- Pre-compute ZIGZAG relative position indices ---
        # This is a non-trainable buffer that defines the layer's core logic.
        # 1. Generate 2D coordinates in zigzag order.
        zigzag_coords = self._generate_zigzag_coords(self.window_size)
        coords_flatten = keras.ops.transpose(
            keras.ops.convert_to_tensor(zigzag_coords, dtype="int32")
        )

        # 2. Calculate pairwise relative coordinates.
        # relative_coords[i, j] = coords[i] - coords[j]
        relative_coords = keras.ops.expand_dims(coords_flatten, 2) - keras.ops.expand_dims(coords_flatten, 1)
        relative_coords = keras.ops.transpose(relative_coords, (1, 2, 0))  # Shape: (N, N, 2)

        # 3. Shift coordinates to be non-negative.
        relative_coords_h = relative_coords[:, :, 0] + self.window_size - 1
        relative_coords_w = relative_coords[:, :, 1] + self.window_size - 1
        # 4. Flatten the 2D relative coordinates into a 1D index for table lookup.
        relative_coords_h *= (2 * self.window_size - 1)
        self.relative_position_index = relative_coords_h + relative_coords_w

    @staticmethod
    def _generate_zigzag_coords(size: int) -> List[Tuple[int, int]]:
        """Generates (row, col) coordinates for a zigzag scan of a square grid.

        :param size: The height and width of the square grid.
        :type size: int
        :return: A list of (row, col) tuples in zigzag order.
        :rtype: List[Tuple[int, int]]
        """
        coords = []
        r, c = 0, 0
        for _ in range(size * size):
            coords.append((r, c))
            if (r + c) % 2 == 0:  # Moving up-right
                if c == size - 1:
                    r += 1
                elif r == 0:
                    c += 1
                else:
                    r -= 1
                    c += 1
            else:  # Moving down-left
                if r == size - 1:
                    c += 1
                elif c == 0:
                    r += 1
                else:
                    r += 1
                    c -= 1
        return coords

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Creates the layer's weights and builds its sub-layers.

        This method is called once by the Keras framework when the layer is first
        used. It is responsible for creating all trainable variables and ensuring
        all sub-layers are also built, which is critical for correct weight
        loading during deserialization.

        :param input_shape: A tuple of integers representing the shape of the
                            input tensor. The last dimension is used to infer
                            weight shapes.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # --- CREATE the layer's own weights ---
        num_relative_positions = (2 * self.window_size - 1) ** 2
        self.relative_position_bias_table = self.add_weight(
            name="relative_position_bias_table",
            shape=(num_relative_positions, self.num_heads),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            dtype=self.dtype
        )

        # --- BUILD all sub-layers explicitly for robust serialization ---
        # We build with the maximum possible sequence length for this window.
        padded_shape = list(input_shape)
        padded_shape[1] = self.window_size * self.window_size
        padded_shape = tuple(padded_shape)

        self.qkv.build(padded_shape)
        # The projection layer takes the output of the attention mechanism,
        # which has the same shape as the padded input.
        self.proj.build(padded_shape)
        # Dropout layers are generally shape-agnostic but building them is good practice.
        if self.attn_dropout is not None:
            self.attn_dropout.build(None) # Attention scores are 4D, so pass None
        if self.proj_dropout is not None:
            self.proj_dropout.build(padded_shape)

        # Always call the parent's build() method at the end.
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Defines the forward pass of the ZigzagWindowAttention layer.

        :param inputs: The input tensor of shape `(batch_size, num_tokens, dim)`.
        :type inputs: keras.KerasTensor
        :param attention_mask: An optional mask of shape `(batch_size, num_tokens)`.
                               Masked positions (to be ignored) are indicated by a 0.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: A boolean indicating whether the layer should behave in
                         training mode (e.g., applying dropout).
        :type training: Optional[bool]
        :return: The output tensor with the same shape as the input.
        :rtype: keras.KerasTensor
        """
        B_actual, N_actual, C_actual = keras.ops.shape(inputs)
        N_target = self.window_size * self.window_size

        if N_actual > N_target:
            raise ValueError(
                f"Input sequence length ({N_actual}) cannot be greater than the "
                f"window area ({N_target})."
            )

        # --- START PADDING LOGIC (handles partial windows) ---
        # This allows the layer to process sequences shorter than the full window
        # by temporarily padding them to the full window size.
        padded_inputs = inputs
        padding_mask = None
        if N_actual < N_target:
            padding_amount = N_target - N_actual
            padding_tensor = keras.ops.zeros((B_actual, padding_amount, C_actual), dtype=inputs.dtype)
            padded_inputs = keras.ops.concatenate([inputs, padding_tensor], axis=1)
            # Create a mask to ignore the padded tokens during attention.
            padding_mask = keras.ops.concatenate([
                keras.ops.ones((B_actual, N_actual), dtype="int32"),
                keras.ops.zeros((B_actual, padding_amount), dtype="int32")
            ], axis=1)

        # Merge the internal padding mask with any user-provided attention mask.
        if attention_mask is not None:
            if keras.ops.shape(attention_mask)[1] < N_target:
                padding_amount_mask = N_target - keras.ops.shape(attention_mask)[1]
                mask_padding = keras.ops.zeros((B_actual, padding_amount_mask), dtype=attention_mask.dtype)
                attention_mask = keras.ops.concatenate([attention_mask, mask_padding], axis=1)
            # Combine masks by multiplication (logical AND).
            final_mask_base = keras.ops.cast(attention_mask, dtype="int32")
            if padding_mask is not None:
                final_mask_base = final_mask_base * padding_mask
            attention_mask = final_mask_base
        elif padding_mask is not None:
            attention_mask = padding_mask
        # --- END PADDING LOGIC ---

        B, N, C = keras.ops.shape(padded_inputs)

        # Project to Q, K, V and reshape for multi-head attention.
        qkv = self.qkv(padded_inputs, training=training)
        qkv = keras.ops.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim))
        qkv = keras.ops.transpose(qkv, (2, 0, 3, 1, 4)) # 3, B, num_heads, N, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Calculate scaled attention scores.
        q = q * self.scale
        attn = keras.ops.matmul(q, keras.ops.transpose(k, (0, 1, 3, 2))) # B, num_heads, N, N

        # Add zigzag relative position bias.
        relative_position_bias = keras.ops.take(
            self.relative_position_bias_table,
            keras.ops.reshape(self.relative_position_index, (-1,)),
            axis=0
        )
        relative_position_bias = keras.ops.reshape(relative_position_bias, (N, N, self.num_heads))
        relative_position_bias = keras.ops.transpose(relative_position_bias, (2, 0, 1)) # num_heads, N, N
        attn = attn + keras.ops.expand_dims(relative_position_bias, 0)

        # Apply attention mask if it exists.
        if attention_mask is not None:
            broadcast_mask = keras.ops.reshape(attention_mask, (B, 1, 1, N))
            inf_value = keras.ops.convert_to_tensor(-1e9, dtype=attn.dtype)
            # Convert mask (0s and 1s) to an additive mask (-inf and 0s).
            additive_mask = (1.0 - keras.ops.cast(broadcast_mask, dtype=attn.dtype)) * inf_value
            attn = attn + additive_mask

        attn = keras.ops.softmax(attn, axis=-1)
        if self.attn_dropout is not None:
            attn = self.attn_dropout(attn, training=training)

        # Compute final output.
        x = keras.ops.matmul(attn, v)
        x = keras.ops.transpose(x, (0, 2, 1, 3)) # B, N, num_heads, head_dim
        x = keras.ops.reshape(x, (B, N, C))

        x = self.proj(x, training=training)
        if self.proj_dropout is not None:
            x = self.proj_dropout(x, training=training)

        # --- START UN-PADDING LOGIC ---
        # Remove the padded tokens to return a tensor of the original length.
        if N_actual < N_target:
            x = x[:, :N_actual, :]
        # --- END UN-PADDING LOGIC ---

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Computes the output shape of the layer.

        For this layer, the output shape is identical to the input shape.

        :param input_shape: The shape of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The shape of the output tensor.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer's configuration as a serializable dictionary.

        This method is used by Keras to save and load the model, allowing it to
        be re-instantiated from this configuration.

        :return: A dictionary containing the layer's configuration.
        :rtype: Dict[str, Any]
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
