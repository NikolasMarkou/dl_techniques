"""
Windowed multi-head self-attention for sequence processing.

This module implements windowed multi-head self-attention as described in the
Swin Transformer paper. The layer takes a 1D sequence of tokens, internally
reshapes it into a 2D grid, partitions the grid into non-overlapping windows,
and computes self-attention within each window.

This implementation is fully self-contained and does not require pre-partitioned
inputs. It automatically handles padding for sequences that do not form perfect
squares or are not perfectly divisible by the window size.

Complete Architecture Flow::

    INPUT: 1D Sequence [batch, N, dim]
      │
      │  Step 1: Grid Formation
      ├─────────────────────────────────────────────┐
      │  Calculate smallest square H×W >= N         │
      │  Pad sequence: N → H×W                      │
      └─────────────────────────────────────────────┘
      │
      ↓  2D Grid [batch, H, W, dim]
      │
      │  Step 2: Window Padding
      ├─────────────────────────────────────────────┐
      │  Pad grid to be divisible by window_size    │
      │  H×W → H_pad×W_pad                          │
      └─────────────────────────────────────────────┘
      │
      ↓  Padded Grid [batch, H_pad, W_pad, dim]
      │
      │  Step 3: Window Partitioning
      ├─────────────────────────────────────────────┐
      │  Split into non-overlapping windows         │
      │  Each window: [window_size × window_size]   │
      └─────────────────────────────────────────────┘
      │
      ↓  Windows [batch×num_windows, ws², dim]
      │
      │  Step 4: Local Self-Attention
      ├─────────────────────────────────────────────┐
      │  Multi-head attention within each window    │
      │  With relative position bias                │
      └─────────────────────────────────────────────┘
      │
      ↓  Attended Windows [batch×num_windows, ws², dim]
      │
      │  Step 5: Window Merging
      ├─────────────────────────────────────────────┐
      │  Reverse partition to grid                  │
      │  H_pad×W_pad grid restored                  │
      └─────────────────────────────────────────────┘
      │
      ↓  Grid [batch, H_pad, W_pad, dim]
      │
      │  Step 6: Unpad & Flatten
      ├─────────────────────────────────────────────┐
      │  Remove padding: H_pad×W_pad → H×W          │
      │  Flatten to sequence: H×W → N               │
      └─────────────────────────────────────────────┘
      │
      ↓
    OUTPUT: 1D Sequence [batch, N, dim]

Grid Formation Example::

    Sequence length N=150 tokens
      ↓
    Calculate grid: H = W = ceil(√150) = 13
    Grid size: 13×13 = 169 tokens
      ↓
    Pad sequence: 150 → 169 (pad 19 tokens)
      ↓
    Reshape to grid: (batch, 169, dim) → (batch, 13, 13, dim)

Window Partitioning Example::

    Grid: 13×13, Window size: 7×7
      ↓
    Pad grid for windowing:
      H: 13 → 14 (pad 1 row)
      W: 13 → 14 (pad 1 col)
      ↓
    Padded grid: 14×14
      ↓
    Number of windows: (14/7) × (14/7) = 2×2 = 4 windows
      ↓
    Each window: 7×7 = 49 tokens
      ↓
    Total shape: (batch×4, 49, dim)

Window Pattern Visualization::

    14×14 Grid with 7×7 windows (4 windows total):

    ┌─────────┬─────────┐
    │ Window  │ Window  │
    │    0    │    1    │
    │ (7×7)   │ (7×7)   │
    ├─────────┼─────────┤
    │ Window  │ Window  │
    │    2    │    3    │
    │ (7×7)   │ (7×7)   │
    └─────────┴─────────┘

    Attention is computed independently within each window:
      - Window 0: Self-attention on 49 tokens
      - Window 1: Self-attention on 49 tokens
      - Window 2: Self-attention on 49 tokens
      - Window 3: Self-attention on 49 tokens
      - No cross-window attention

Conceptual Overview
-------------------

Standard transformer attention computes relationships between all pairs of
tokens, leading to O(N²) complexity. WindowAttention reduces this by applying
attention only within local windows.

Key Benefits
~~~~~~~~~~~~

1. **Linear Complexity**: Scales efficiently with the number of tokens.
   Complexity: O(N × window_size²) vs O(N²) for standard attention.

2. **General Applicability**: Works on any 1D sequence without assuming a 2D
   input structure.

3. **End-to-End Logic**: Handles all padding, partitioning, and merging
   internally.

4. **Relative Position Awareness**: Learns spatial relationships within each
   window via learned position biases.

Complexity Analysis
-------------------

============== ================= ====================================
Operation      Complexity        Notes
============== ================= ====================================
Grid formation O(N)              Linear padding and reshape
Window split   O(N)              Linear partitioning
Attention      O(N × ws²)        ws = window_size, independent windows
Window merge   O(N)              Linear reverse operation
Total          O(N × ws²)        Linear in sequence length N
============== ================= ====================================

Compare to standard attention: O(N²)

Example: N=1024, window_size=8
  - Standard attention: 1,048,576 operations
  - Window attention: 65,536 operations
  - Speedup: **16x faster**

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

    import keras

    # Input sequence of 150 tokens
    # Layer will form 13×13 grid, pad to 14×14
    # Partition into 2×2 windows of 7×7 each
    window_attn = WindowAttention(dim=96, window_size=7, num_heads=4)

    # Input: (batch_size, sequence_length, channels)
    x = keras.random.normal((2, 150, 96))
    output = window_attn(x)  # Output shape: (2, 150, 96)

    # Example with attention mask
    # Mask shape: (batch_size, sequence_length)
    # 1 = valid token, 0 = masked token
    mask = keras.ops.ones((2, 150))
    output_masked = window_attn(x, attention_mask=mask)

References
----------
- Liu, Z., et al. (2021). "Swin Transformer: Hierarchical Vision
  Transformer using Shifted Windows". ICCV 2021.
  https://arxiv.org/abs/2103.14030
"""

import keras
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SingleWindowAttention(keras.layers.Layer):
    """
    Multi-head self-attention for a single window (internal use).

    This is the core attention module that operates on an input tensor which
    is assumed to be a single window or a batch of windows. It includes
    relative position bias and is designed to be used by the public
    WindowAttention layer.

    This layer handles inputs shorter than window_size² via internal padding.

    **Architecture**::

        Input (B, N, C) -- N <= ws²
               │
               ▼
        ┌──────────────────────┐
        │   Pad to ws²         │
        └──────────────────────┘
               │
        (B, ws², C)
               │
               ▼
        ┌──────────────────────┐
        │   QKV Projection     │  Dense(3×C)
        └──────────────────────┘
               │
        (B, ws², 3×C)
               │
        ┌──────┴──────┬──────┐
        ▼      ▼      ▼      ▼
      Query   Key    Value
     (B,H,ws²,D) (B,H,ws²,D) (B,H,ws²,D)
        │      │             │
        └──►MatMul◄──────────┘
              │
      (B, H, ws², ws²)
              │
        ┌─────┴─────────────────────┐
        │ + Relative Position Bias  │
        │ + Attention Mask          │
        └───────────────────────────┘
              │
              ▼
        ┌──────────────┐
        │   Softmax    │
        └──────────────┘
              │
      Attention Scores
      (B, H, ws², ws²)
              │
              └────►MatMul◄────────┐
                      │            │
              (B, H, ws², D)     Value
                      │
                      ▼
              ┌───────────────┐
              │ Reshape & Cat │
              └───────────────┘
                      │
              (B, ws², C)
                      │
                      ▼
              ┌───────────────┐
              │  Output Proj  │
              └───────────────┘
                      │
              (B, ws², C)
                      │
                      ▼
              ┌───────────────┐
              │   Unpad to N  │
              └───────────────┘
                      │
              Output (B, N, C)

    Where:
      - B: Number of windows (batch dimension)
      - N: Original tokens in window (N ≤ ws²)
      - C: Channel dimension (dim)
      - ws: window_size
      - H: num_heads
      - D: head_dim (C / H)

    **Relative Position Bias**::

        For a 3×3 window, the relative position indices:

        Position coordinates:
        ┌───────┬───────┬───────┐
        │ (0,0) │ (0,1) │ (0,2) │
        ├───────┼───────┼───────┤
        │ (1,0) │ (1,1) │ (1,2) │
        ├───────┼───────┼───────┤
        │ (2,0) │ (2,1) │ (2,2) │
        └───────┴───────┴───────┘

        Relative position from (1,1) to all positions:
        ┌────────┬────────┬────────┐
        │ (-1,-1)│ (-1,0) │ (-1,+1)│
        ├────────┼────────┼────────┤
        │ (0,-1) │ (0,0)  │ (0,+1) │
        ├────────┼────────┼────────┤
        │ (+1,-1)│ (+1,0) │ (+1,+1)│
        └────────┴────────┴────────┘

        Each unique relative position has a learned bias
        Bias table size: (2×ws-1)² entries

    :param dim: Dimension of the input tokens (channels).
    :type dim: int
    :param window_size: The height and width of the attention window.
    :type window_size: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param qkv_bias: If True, add learnable bias to query, key, value.
        Default: True.
    :type qkv_bias: bool
    :param qk_scale: Override for query-key scaling factor. Defaults to
        head_dim ** -0.5.
    :type qk_scale: Optional[float]
    :param dropout_rate: Dropout rate for attention scores. Default: 0.0.
    :type dropout_rate: float
    :param proj_bias: If True, add learnable bias to output projection.
        Default: True.
    :type proj_bias: bool
    :param kernel_initializer: Initializer for kernel weights.
        Default: 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for bias weights. Default: 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for kernel weights. Default: None.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Regularizer for bias weights. Default: None.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param kwargs: Other keyword arguments for base Layer.
    :type kwargs: Any

    :raises ValueError: If dim is not positive.
    :raises ValueError: If window_size is not positive.
    :raises ValueError: If num_heads is not positive.
    :raises ValueError: If dim is not divisible by num_heads.
    :raises ValueError: If dropout_rate is not in [0, 1].
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        dropout_rate: float = 0.0,
        proj_bias: bool = True,
        kernel_initializer: Union[
            str, keras.initializers.Initializer
        ] = "glorot_uniform",
        bias_initializer: Union[
            str, keras.initializers.Initializer
        ] = "zeros",
        kernel_regularizer: Optional[
            Union[str, keras.regularizers.Regularizer]
        ] = None,
        bias_regularizer: Optional[
            Union[str, keras.regularizers.Regularizer]
        ] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the single window attention layer.

        :param dim: Token dimension.
        :type dim: int
        :param window_size: Window height and width.
        :type window_size: int
        :param num_heads: Number of attention heads.
        :type num_heads: int
        :param qkv_bias: Use bias in QKV projection.
        :type qkv_bias: bool
        :param qk_scale: Optional attention scaling.
        :type qk_scale: Optional[float]
        :param dropout_rate: Attention dropout rate.
        :type dropout_rate: float
        :param proj_bias: Use bias in output projection.
        :type proj_bias: bool
        :param kernel_initializer: Weight initializer.
        :type kernel_initializer: Union[str, keras.initializers.Initializer]
        :param bias_initializer: Bias initializer.
        :type bias_initializer: Union[str, keras.initializers.Initializer]
        :param kernel_regularizer: Weight regularizer.
        :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
        :param bias_regularizer: Bias regularizer.
        :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
        :param kwargs: Additional layer arguments.
        :type kwargs: Any
        """
        super().__init__(**kwargs)

        # Validate parameters
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if window_size <= 0:
            raise ValueError(
                f"window_size must be positive, got {window_size}"
            )
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(
                "dropout_rate must be between 0.0 and 1.0, "
                f"got {dropout_rate}"
            )

        # Store ALL configuration parameters for serialization
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_scale = qk_scale
        self.scale = (
            qk_scale if qk_scale is not None else self.head_dim**-0.5
        )
        self.qkv_bias = qkv_bias
        self.proj_bias = proj_bias
        self.dropout_rate = dropout_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # CREATE all sub-layers in __init__
        self.qkv = keras.layers.Dense(
            self.dim * 3,
            use_bias=self.qkv_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="qkv",
        )
        self.proj = keras.layers.Dense(
            self.dim,
            use_bias=self.proj_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="proj",
        )
        self.attn_dropout = (
            keras.layers.Dropout(self.dropout_rate, name="attn_dropout")
            if self.dropout_rate > 0.0
            else None
        )

        # Pre-compute constant relative position indices
        # Creates indices for looking up learned position biases
        #
        # Relative Position Index Computation
        # ====================================
        # For each pair of positions (i,j) in the window, compute their
        # relative position (Δrow, Δcol). Map this to a unique index
        # for looking up the learned bias.
        #
        # Example 3×3 window:
        #   Positions: 0-8
        #   Relative range: [-2, -1, 0, +1, +2] for each dimension
        #   Index range: [0, 24] for (2×3-1)² = 25 combinations
        #
        coords_h = keras.ops.arange(self.window_size, dtype="int32")
        coords_w = keras.ops.arange(self.window_size, dtype="int32")
        coords = keras.ops.stack(
            keras.ops.meshgrid(coords_h, coords_w, indexing="ij")
        )
        coords_flatten = keras.ops.reshape(coords, (2, -1))
        relative_coords = keras.ops.expand_dims(
            coords_flatten, 2
        ) - keras.ops.expand_dims(coords_flatten, 1)
        relative_coords = keras.ops.transpose(relative_coords, (1, 2, 0))
        relative_coords_h = relative_coords[:, :, 0] + self.window_size - 1
        relative_coords_w = relative_coords[:, :, 1] + self.window_size - 1
        relative_coords_h *= 2 * self.window_size - 1
        self.relative_position_index = relative_coords_h + relative_coords_w

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer's weights.

        Creates the relative position bias table and builds sub-layers.

        **Relative Position Bias Table**::

            For window_size = ws:
              - Number of unique relative positions: (2×ws-1)²
              - Each relative position has num_heads bias values
              - Table shape: [(2×ws-1)², num_heads]

            Example: ws=3, num_heads=4
              - Unique positions: (2×3-1)² = 25
              - Table shape: [25, 4]
              - Each head learns different spatial biases

        :param input_shape: Shape tuple of input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Create learnable relative position bias table
        num_relative_positions = (2 * self.window_size - 1) ** 2
        self.relative_position_bias_table = self.add_weight(
            name="relative_position_bias_table",
            shape=(num_relative_positions, self.num_heads),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            dtype=self.dtype,
        )

        # Build QKV and projection layers with padded shape
        padded_shape = list(input_shape)
        padded_shape[1] = self.window_size * self.window_size
        self.qkv.build(padded_shape)
        self.proj.build(padded_shape)

        # Build dropout
        if self.attn_dropout is not None:
            self.attn_dropout.build(None)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """
        Forward pass for single window attention.

        **Processing Steps**::

            1. Pad input to window_size²
            2. Create padding mask
            3. Combine with user mask
            4. QKV projection and reshape
            5. Compute attention scores
            6. Add relative position bias
            7. Apply attention mask
            8. Softmax normalization
            9. Apply attention to values
            10. Output projection
            11. Remove padding

        **Attention Mask Handling**::

            Internal Padding Mask:
              [1, 1, ..., 1, 0, 0, 0]  ← 1=valid, 0=padded
              └────N────┘ └──pad──┘

            Combined Mask:
              final_mask = user_mask × padding_mask

            Applied as Additive Mask:
              scores = scores + (1 - mask) × (-inf)

        :param inputs: Input tensor of shape (B, N, C) where N ≤ ws².
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional mask of shape (B, N). 1=valid,
            0=masked. Default: None.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Whether in training mode. Used for dropout.
            Default: None.
        :type training: Optional[bool]
        :return: Output tensor of shape (B, N, C).
        :rtype: keras.KerasTensor
        """
        # Get actual input dimensions
        input_shape = keras.ops.shape(inputs)
        B_actual, N_actual, C_actual = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )
        N_target = self.window_size * self.window_size

        # PADDING & MASKING LOGIC
        # ========================
        # Pad input to full window size (ws²)
        # Note: N_actual <= N_target by layer contract
        # padding_amount is guaranteed non-negative
        # Unconditional padding is graph-compatible (no-op if pad=0)
        padding_amount = N_target - N_actual
        padded_inputs = keras.ops.pad(
            inputs, [[0, 0], [0, padding_amount], [0, 0]]
        )

        # Create internal padding mask
        # 1 = valid token, 0 = padded token
        internal_padding_mask = keras.ops.concatenate(
            [
                keras.ops.ones((B_actual, N_actual), dtype="int32"),
                keras.ops.zeros((B_actual, padding_amount), dtype="int32"),
            ],
            axis=1,
        )

        # Combine with user-provided mask if present
        final_attention_mask = internal_padding_mask
        if attention_mask is not None:
            padded_user_mask = keras.ops.pad(
                attention_mask, [[0, 0], [0, padding_amount]]
            )
            final_attention_mask = (
                keras.ops.cast(padded_user_mask, "int32")
                * internal_padding_mask
            )

        # ATTENTION LOGIC
        # ===============
        # Get padded dimensions
        B, N, C = keras.ops.shape(padded_inputs)

        # QKV projection
        # Shape: (B, N, C) → (B, N, 3×C)
        qkv = self.qkv(padded_inputs, training=training)

        # Reshape and split into Q, K, V
        # Shape: (B, N, 3×C) → (B, N, 3, num_heads, head_dim)
        qkv = keras.ops.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim))

        # Transpose to (3, B, num_heads, N, head_dim)
        qkv = keras.ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scale queries for attention computation
        # Scaled dot-product: (Q @ K^T) / √d
        q = q * self.scale

        # Compute attention scores
        # Shape: (B, num_heads, N, N)
        attn = keras.ops.matmul(q, keras.ops.transpose(k, (0, 1, 3, 2)))

        # Add relative position bias
        # Lookup learned biases for all position pairs
        #
        # Relative Position Bias Addition
        # ================================
        # For each (i,j) position pair, add learned bias b[relative_pos(i,j)]
        # Different heads learn different spatial biases
        #
        # Example: Position (1,1) to (2,2) in 3×3 window
        #   Relative position: (+1, +1)
        #   Index in bias table: unique index for (+1, +1)
        #   Add bias[index, :] to attention scores for all heads
        #
        relative_position_bias = keras.ops.take(
            self.relative_position_bias_table,
            keras.ops.reshape(self.relative_position_index, (-1,)),
            axis=0,
        )
        relative_position_bias = keras.ops.reshape(
            relative_position_bias, (N_target, N_target, -1)
        )
        relative_position_bias = keras.ops.transpose(
            relative_position_bias, (2, 0, 1)
        )
        attn = attn + keras.ops.expand_dims(relative_position_bias, 0)

        # Apply attention mask
        # Convert mask to additive form: 0 for valid, -inf for masked
        broadcast_mask = keras.ops.reshape(final_attention_mask, (B, 1, 1, N))

        inf_value = keras.ops.convert_to_tensor(-1e9, dtype=attn.dtype)
        additive_mask = (
            1.0 - keras.ops.cast(broadcast_mask, dtype=attn.dtype)
        ) * inf_value
        attn = attn + additive_mask

        # Clip attention logits before softmax
        # Prevents exp() overflow which causes NaNs
        # Safe range: [-30, 30] for float32 and float16
        attn = keras.ops.clip(attn, -30.0, 30.0)

        # Normalize attention scores
        attn = keras.ops.softmax(attn, axis=-1)

        # Apply dropout to attention weights
        if self.attn_dropout is not None:
            attn = self.attn_dropout(attn, training=training)

        # Apply attention to values
        # Shape: (B, num_heads, N, head_dim)
        x = keras.ops.matmul(attn, v)

        # Transpose and reshape back to (B, N, C)
        x = keras.ops.transpose(x, (0, 2, 1, 3))
        x = keras.ops.reshape(x, (B, N, C))

        # Output projection
        x = self.proj(x, training=training)

        # Remove padding to restore original sequence length
        output = x[:, :N_actual, :]
        return output

    def get_config(self) -> Dict[str, Any]:
        """
        Serialize the layer's configuration.

        :return: Dictionary containing layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
                "qkv_bias": self.qkv_bias,
                "qk_scale": self.qk_scale,
                "dropout_rate": self.dropout_rate,
                "proj_bias": self.proj_bias,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": keras.regularizers.serialize(
                    self.bias_regularizer
                ),
            }
        )
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class WindowAttention(keras.layers.Layer):
    """
    Window-based multi-head self-attention layer.

    This layer implements the complete windowed multi-head self-attention
    mechanism. It takes a 1D sequence, reshapes it into a 2D grid, partitions
    the grid into windows, applies attention within each window, and merges
    the result back into a 1D sequence of the original length.

    **Architecture**::

        Input (B, N, C)
               │
               ▼
        ┌────────────────────────┐
        │  Pad to Square Grid    │  N → H×W
        └────────────────────────┘
               │
        (B, N_grid, C)
               │
               ▼
        ┌────────────────────────┐
        │  Reshape to 2D Grid    │
        └────────────────────────┘
               │
        (B, H, W, C)
               │
               ▼
        ┌────────────────────────┐
        │  Pad for Windowing     │  Make divisible by ws
        └────────────────────────┘
               │
        (B, H_pad, W_pad, C)
               │
               ▼
        ┌────────────────────────┐
        │  Partition → Windows   │
        └────────────────────────┘
               │
        (B×num_win, ws², C)
               │
               ▼
        ┌────────────────────────┐
        │ SingleWindowAttention  │  With relative position bias
        └────────────────────────┘
               │
        (B×num_win, ws², C)
               │
               ▼
        ┌────────────────────────┐
        │  Reverse Partition     │
        └────────────────────────┘
               │
        (B, H_pad, W_pad, C)
               │
               ▼
        ┌────────────────────────┐
        │  Unpad Grid            │  H_pad×W_pad → H×W
        └────────────────────────┘
               │
        (B, H, W, C)
               │
               ▼
        ┌────────────────────────┐
        │  Flatten & Unpad       │  H×W → N
        └────────────────────────┘
               │
        Output (B, N, C)

    **Example Processing** (N=150, window_size=7)::

        Input: 150 tokens
          ↓
        Grid formation: H=W=13 (169 tokens, pad 19)
          ↓
        Window padding: 13×13 → 14×14 (pad 1 row, 1 col)
          ↓
        Partitioning: 14×14 → 4 windows of 7×7
          ↓
        Windows shape: (batch×4, 49, dim)
          ↓
        Attention in each window independently
          ↓
        Merge: 4 windows → 14×14 grid
          ↓
        Unpad: 14×14 → 13×13 → 150 tokens
          ↓
        Output: 150 tokens

    :param dim: Dimension of the input tokens (channels).
    :type dim: int
    :param window_size: The height and width of the attention window.
    :type window_size: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param kwargs: Keyword arguments for the internal SingleWindowAttention
        layer (qkv_bias, dropout_rate, etc.).
    :type kwargs: Any

    Examples::

        >>> import keras
        >>> # Basic usage
        >>> layer = WindowAttention(dim=96, window_size=7, num_heads=4)
        >>> x = keras.random.normal((2, 150, 96))
        >>> output = layer(x)  # Shape: (2, 150, 96)
        >>>
        >>> # With attention mask
        >>> mask = keras.ops.ones((2, 150))
        >>> output = layer(x, attention_mask=mask)
        >>>
        >>> # With custom parameters
        >>> layer = WindowAttention(
        ...     dim=96,
        ...     window_size=7,
        ...     num_heads=4,
        ...     qkv_bias=True,
        ...     dropout_rate=0.1
        ... )
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        **kwargs: Any,
    ):
        """
        Initialize the window attention layer.

        :param dim: Token dimension.
        :type dim: int
        :param window_size: Window height and width.
        :type window_size: int
        :param num_heads: Number of attention heads.
        :type num_heads: int
        :param kwargs: Additional arguments for SingleWindowAttention.
        :type kwargs: Any
        """
        super().__init__(name=kwargs.pop("name", "window_attention"))
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        # All other configuration arguments are passed to inner attention
        self.kwargs = kwargs

        # Create core attention mechanism for single window
        self.attention = SingleWindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            **kwargs,
        )

    def _window_partition(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """
        Partition a 4D grid tensor into windows.

        Splits the H×W grid into non-overlapping windows of size ws×ws.

        **Partitioning Process** (H=14, W=14, ws=7)::

            Input Grid (14×14):
            ┌─────────────────────────┐
            │         H=14            │
            │  ┌──────────┬──────────┐│
            │  │          │          ││
            │  │ Window 0 │ Window 1 ││
            │  │  (7×7)   │  (7×7)   ││
            │  │          │          ││
            │W=├──────────┼──────────┤│
            │14│          │          ││
            │  │ Window 2 │ Window 3 ││
            │  │  (7×7)   │  (7×7)   ││
            │  │          │          ││
            │  └──────────┴──────────┘│
            └─────────────────────────┘

            Output: 4 windows, each 7×7

            Reshape sequence:
              (B, 14, 14, C)
              → (B, 2, 7, 2, 7, C)  # Separate window dimensions
              → (B, 2, 2, 7, 7, C)  # Reorder
              → (B×4, 7, 7, C)      # Flatten batch and windows

        :param x: 4D grid tensor of shape (B, H, W, C).
        :type x: keras.KerasTensor
        :return: Partitioned windows of shape (B×num_windows, ws, ws, C).
        :rtype: keras.KerasTensor
        """
        B, H, W, C = keras.ops.shape(x)
        ws = self.window_size

        # Reshape to separate window dimensions
        # (B, H, W, C) → (B, H//ws, ws, W//ws, ws, C)
        x = keras.ops.reshape(x, (B, H // ws, ws, W // ws, ws, C))

        # Reorder dimensions to group windows together
        # (B, H//ws, ws, W//ws, ws, C) → (B, H//ws, W//ws, ws, ws, C)
        x = keras.ops.transpose(x, (0, 1, 3, 2, 4, 5))

        # Flatten batch and window dimensions
        # (B, H//ws, W//ws, ws, ws, C) → (B×num_windows, ws, ws, C)
        windows = keras.ops.reshape(x, (-1, ws, ws, C))
        return windows

    def _window_reverse(
        self, windows: keras.KerasTensor, H: int, W: int
    ) -> keras.KerasTensor:
        """
        Merge windows back into a 4D grid tensor.

        Reverses the window partitioning operation.

        **Merging Process** (4 windows 7×7 → 14×14 grid)::

            Input: 4 windows
            ┌──────┐ ┌──────┐
            │Win 0 │ │Win 1 │
            │ 7×7  │ │ 7×7  │
            └──────┘ └──────┘
            ┌──────┐ ┌──────┐
            │Win 2 │ │Win 3 │
            │ 7×7  │ │ 7×7  │
            └──────┘ └──────┘

            Output Grid:
            ┌─────────────────┐
            │    14×14        │
            │  ┌─────┬─────┐  │
            │  │Win0 │Win1 │  │
            │  ├─────┼─────┤  │
            │  │Win2 │Win3 │  │
            │  └─────┴─────┘  │
            └─────────────────┘

            Reshape sequence (reverse of partition):
              (B×4, 7, 7, C)
              → (B, 2, 2, 7, 7, C)  # Unflatten
              → (B, 2, 7, 2, 7, C)  # Reorder
              → (B, 14, 14, C)      # Merge dimensions

        :param windows: Partitioned windows of shape
            (B×num_windows, ws, ws, C).
        :type windows: keras.KerasTensor
        :param H: Original grid height.
        :type H: int
        :param W: Original grid width.
        :type W: int
        :return: Reconstructed 4D grid of shape (B, H, W, C).
        :rtype: keras.KerasTensor
        """
        ws = self.window_size
        num_windows_h = H // ws
        num_windows_w = W // ws
        num_windows_total = keras.ops.shape(windows)[0]
        B = num_windows_total // (num_windows_h * num_windows_w)

        # Unflatten batch and window dimensions
        # (B×num_windows, ws, ws, C) → (B, num_h, num_w, ws, ws, C)
        x = keras.ops.reshape(
            windows, (B, num_windows_h, num_windows_w, ws, ws, -1)
        )

        # Reorder dimensions
        # (B, num_h, num_w, ws, ws, C) → (B, num_h, ws, num_w, ws, C)
        x = keras.ops.transpose(x, (0, 1, 3, 2, 4, 5))

        # Merge window dimensions back to grid
        # (B, num_h, ws, num_w, ws, C) → (B, H, W, C)
        x = keras.ops.reshape(x, (B, H, W, -1))
        return x

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer's weights.

        Builds the internal SingleWindowAttention layer with appropriate
        input shape.

        :param input_shape: Shape tuple of input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Build internal attention layer
        # It expects input of shape (batch, window_area, dim)
        # where window_area = window_size²
        self.attention.build(
            (input_shape[0], self.window_size * self.window_size, self.dim)
        )
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """
        Forward pass for the window attention layer.

        **Processing Steps**::

            1. Calculate grid dimensions (H×W)
            2. Pad sequence to grid size
            3. Reshape to 2D grid
            4. Pad grid for window divisibility
            5. Partition into windows
            6. Apply attention per window
            7. Merge windows back to grid
            8. Unpad grid
            9. Flatten and unpad sequence

        **Shape Transformations** (N=150, ws=7)::

            Input:        (B, 150, C)
            ↓ pad
            Grid seq:     (B, 169, C)  # 13×13 grid
            ↓ reshape
            Grid:         (B, 13, 13, C)
            ↓ pad
            Grid padded:  (B, 14, 14, C)
            ↓ partition
            Windows:      (B×4, 49, C)  # 4 windows, 49=7²
            ↓ attention
            Attended:     (B×4, 49, C)
            ↓ reverse
            Grid padded:  (B, 14, 14, C)
            ↓ unpad
            Grid:         (B, 13, 13, C)
            ↓ flatten
            Grid seq:     (B, 169, C)
            ↓ unpad
            Output:       (B, 150, C)

        :param inputs: Input tensor of shape (B, N, C). B=batch size,
            N=sequence length, C=channels.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional mask of shape (B, N). 1=valid token,
            0=masked token. Default: None.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Whether in training mode. Used for dropout.
            Default: None.
        :type training: Optional[bool]
        :return: Output tensor of shape (B, N, C).
        :rtype: keras.KerasTensor
        """
        # Get input dimensions
        input_shape = keras.ops.shape(inputs)
        B, N_actual, C = input_shape[0], input_shape[1], input_shape[2]
        ws = self.window_size

        # Step 1: Grid Formation
        # ======================
        # Calculate smallest square grid that fits the sequence
        # H = W = ceil(√N)
        #
        # Example: N=150
        #   √150 ≈ 12.25
        #   ceil(12.25) = 13
        #   Grid: 13×13 = 169 tokens
        #   Padding needed: 169 - 150 = 19 tokens
        #
        H = W = keras.ops.cast(
            keras.ops.ceil(
                keras.ops.sqrt(keras.ops.cast(N_actual, "float32"))
            ),
            "int32",
        )
        N_grid = H * W

        # Pad sequence to form perfect grid
        # Use maximum to ensure padding is non-negative (graph-compatible)
        pad_amount_seq = N_grid - N_actual
        safe_pad_amount_seq = keras.ops.maximum(0, pad_amount_seq)

        x = keras.ops.pad(inputs, [[0, 0], [0, safe_pad_amount_seq], [0, 0]])

        # Reshape to 2D grid
        x = keras.ops.reshape(x, (B, H, W, C))

        # Step 2: Window Padding
        # ======================
        # Pad grid to be divisible by window_size
        # This ensures we can partition into complete windows
        #
        # Example: H=W=13, ws=7
        #   H % ws = 13 % 7 = 6
        #   pad_h = (7 - 6) % 7 = 1
        #   W % ws = 13 % 7 = 6
        #   pad_w = (7 - 6) % 7 = 1
        #   Padded grid: 14×14
        #
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = keras.ops.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
        H_pad, W_pad = H + pad_h, W + pad_w

        # Step 3: Window Partitioning
        # ===========================
        # Split grid into non-overlapping windows
        #
        # Example: 14×14 grid, 7×7 windows
        #   Number of windows: (14/7) × (14/7) = 2×2 = 4 windows
        #   Each window: 7×7 = 49 tokens
        #   Shape: (B×4, 49, C)
        #
        windows = self._window_partition(x)
        windows = keras.ops.reshape(windows, (-1, ws * ws, C))

        # Step 3a: Attention Mask Handling
        # =================================
        # If user provides mask, process it through same transformations
        window_mask = None
        if attention_mask is not None:
            mask = attention_mask

            # Pad mask to grid size
            mask = keras.ops.pad(mask, [[0, 0], [0, safe_pad_amount_seq]])

            # Reshape to 2D grid
            mask = keras.ops.reshape(mask, (B, H, W))

            # Pad mask for windowing
            mask = keras.ops.pad(mask, [[0, 0], [0, pad_h], [0, pad_w]])

            # Partition mask into windows
            mask = keras.ops.expand_dims(mask, axis=-1)
            mask_windows = self._window_partition(mask)
            window_mask = keras.ops.reshape(mask_windows, (-1, ws * ws))

        # Step 4: Local Attention
        # =======================
        # Apply self-attention independently within each window
        # Each window attends only to itself (no cross-window attention)
        attn_windows = self.attention(
            windows, attention_mask=window_mask, training=training
        )

        # Step 5: Window Merging
        # ======================
        # Reverse the partitioning to reconstruct the grid
        #
        # Example: 4 windows of 7×7 → 14×14 grid
        #   Shape: (B×4, 49, C) → (B, 14, 14, C)
        #
        attn_windows = keras.ops.reshape(attn_windows, (-1, ws, ws, C))
        x = self._window_reverse(attn_windows, H_pad, W_pad)

        # Step 6: Output Unpadding
        # ========================
        # Remove padding added for windowing
        #
        # Example:
        #   Grid: 14×14 → 13×13 (remove 1 row, 1 col)
        #   Sequence: 169 → 150 (remove 19 tokens)
        #
        # Slicing is graph-compatible
        x = x[:, :H, :W, :]

        # Flatten grid back to sequence
        x = keras.ops.reshape(x, (B, N_grid, C))

        # Remove sequence padding to restore original length
        x = x[:, :N_actual, :]

        return x

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Output shape is identical to input shape.

        :param input_shape: Shape tuple of input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple (same as input).
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Serialize the layer's configuration.

        :return: Dictionary containing layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
            }
        )
        # Add all kwargs passed to inner layer for complete serialization
        config.update(self.kwargs)
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "WindowAttention":
        """
        Create a layer from its configuration.

        :param config: Dictionary containing layer configuration.
        :type config: Dict[str, Any]
        :return: New instance of WindowAttention.
        :rtype: WindowAttention
        """
        # Create copy to avoid mutating original config
        # (often used for comparison in tests after deserialization)
        config_copy = config.copy()

        # Pop main args and pass rest as kwargs to __init__
        dim = config_copy.pop("dim")
        window_size = config_copy.pop("window_size")
        num_heads = config_copy.pop("num_heads")
        return cls(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            **config_copy,
        )