"""WindowAttention Layer.

A Keras layer implementing windowed multi-head self-attention as described in
the Swin Transformer paper. This layer takes a 1D sequence of tokens,
internally reshapes it into a 2D grid, partitions the grid into
non-overlapping windows, and computes self-attention within each window.

This implementation is fully self-contained and does not require pre-partitioned
inputs. It automatically handles padding for sequences that do not form perfect
squares or are not perfectly divisible by the window size.

Conceptual Overview
-------------------

Standard transformer attention computes relationships between all pairs of
tokens, leading to O(N²) complexity. WindowAttention reduces this by applying
attention only within local windows.

Workflow:
~~~~~~~~~

1. **Input**: A tensor of shape ``(B, N, C)``.
2. **Grid Formation**: The sequence of length `N` is conceptually arranged into
   the smallest possible square grid (`H x W`), padding the sequence if `N` is
   not a perfect square.
3. **Window Partitioning**: This `H x W` grid is further padded to be divisible
   by `window_size` and then partitioned into non-overlapping windows.
4. **Local Attention**: Multi-head self-attention with relative position
   biases is computed independently within each window.
5. **Window Merging**: The processed windows are stitched back together to form
   the grid.
6. **Output**: The grid is un-padded and flattened back to a sequence of the
   original length `N`, resulting in a tensor of shape ``(B, N, C)``.

Key Benefits:
~~~~~~~~~~~~~

1. **Linear Complexity**: Scales efficiently with the number of tokens.
2. **General Applicability**: Works on any 1D sequence without assuming a 2D
   input structure.
3. **End-to-End Logic**: Handles all padding, partitioning, and merging
   internally.
4. **Relative Position Awareness**: Learns spatial relationships within each
   window.

Usage Example:
~~~~~~~~~~~~~~
.. code-block:: python

    import keras

    # Input sequence of 150 tokens. The layer will internally form a 13x13 grid.
    # The grid will be partitioned into 7x7 windows.
    window_attn = WindowAttention(dim=96, window_size=7, num_heads=4)

    # Input: (batch_size, sequence_length, channels)
    x = keras.random.normal((2, 150, 96))
    output = window_attn(x)  # Output shape: (2, 150, 96)

    # Example with an attention mask
    mask = keras.ops.ones((2, 150)) # Mask is shape (batch_size, sequence_length)
    output_masked = window_attn(x, attention_mask=mask) # Shape: (2, 150, 96)

"""

import keras
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SingleWindowAttention(keras.layers.Layer):
    """Multi-head Self-Attention for a single window.

    This is the core attention module that operates on an input tensor which is
    assumed to be a single window or a batch of windows. It includes relative
    position bias and is designed to be used by the public `WindowAttention`
    layer.

    This layer handles inputs shorter than ``window_size**2`` via internal
    padding.

    Architecture:
    ~~~~~~~~~~~~~
    .. code-block::

        Input (B, N, C) -- N <= ws*ws
               │
               ▼
        Pad to Window Area
               │
        (B, ws*ws, C)
               │
               ▼
            QKV Dense Layer
               │
        (B, ws*ws, 3*C)
               │
               └───────────┬───────────┐
               ▼           ▼           ▼
             Query       Key         Value
        (B, H, ws*ws, D) (B, H, ws*ws, D) (B, H, ws*ws, D)
               │           │                 ▲
               └─────► MatMul◄───────────────┘
                         │
               (B, H, ws*ws, ws*ws)
                         │
                         ├─ Add Relative Position Bias
                         ├─ Add Attention Mask (from user + padding)
                         │
                         ▼
                      Softmax
                         │
                         ▼
                   Attention Scores
               (B, H, ws*ws, ws*ws)
                         │
                         └─────► MatMul ◄──────────────
                                   │
                           (B, H, ws*ws, D)
                                   │
                                   ▼
                            Reshape & Concat
                                   │
                             (B, ws*ws, C)
                                   │
                                   ▼
                              Projection Layer
                                   │
                             (B, ws*ws, C)
                                   │
                                   ▼
                           Un-pad to Original Length
                                   │
                            Output (B, N, C)

    Where:
      - ``B``: Number of windows (batch dimension for this layer).
      - ``N``: Original number of tokens in the window.
      - ``C``: Channel dimension (``dim``).
      - ``ws``: ``window_size``.
      - ``H``: ``num_heads``.
      - ``D``: ``head_dim`` (``C / H``).

    Args:
        dim: Dimension of the input tokens (channels).
        window_size: The height and width of the attention window.
        num_heads: Number of attention heads.
        qkv_bias: If ``True``, add a learnable bias to query, key, value.
        qk_scale: Override for query-key scaling factor. Defaults to
            ``head_dim ** -0.5``.
        attn_dropout_rate: Dropout rate for the attention scores.
        proj_dropout_rate: Dropout rate for the final output projection.
        proj_bias: If ``True``, add a learnable bias to the output projection.
        kernel_initializer: Initializer for kernel weights.
        bias_initializer: Initializer for bias weights.
        kernel_regularizer: Regularizer for kernel weights.
        bias_regularizer: Regularizer for bias weights.
        **kwargs: Other keyword arguments for the base ``keras.layers.Layer``.
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
        super().__init__(**kwargs)

        # --- Validation ---
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
        if not (0.0 <= attn_dropout_rate <= 1.0):
            raise ValueError(
                "attn_dropout_rate must be between 0.0 and 1.0, "
                f"got {attn_dropout_rate}"
            )
        if not (0.0 <= proj_dropout_rate <= 1.0):
            raise ValueError(
                "proj_dropout_rate must be between 0.0 and 1.0, "
                f"got {proj_dropout_rate}"
            )

        # --- Store ALL configuration parameters for serialization ---
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
        self.attn_dropout_rate = attn_dropout_rate
        self.proj_dropout_rate = proj_dropout_rate
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # --- CREATE all sub-layers in __init__ ---
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
            keras.layers.Dropout(self.attn_dropout_rate, name="attn_dropout")
            if self.attn_dropout_rate > 0.0
            else None
        )
        self.proj_dropout = (
            keras.layers.Dropout(self.proj_dropout_rate, name="proj_dropout")
            if self.proj_dropout_rate > 0.0
            else None
        )

        # --- Pre-compute constant relative position indices ---
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
        """Build the layer's weights.

        Args:
            input_shape: Shape of the input tensor.
        """
        num_relative_positions = (2 * self.window_size - 1) ** 2
        self.relative_position_bias_table = self.add_weight(
            name="relative_position_bias_table",
            shape=(num_relative_positions, self.num_heads),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            dtype=self.dtype,
        )
        padded_shape = list(input_shape)
        padded_shape[1] = self.window_size * self.window_size
        self.qkv.build(padded_shape)
        self.proj.build(padded_shape)
        if self.attn_dropout is not None:
            self.attn_dropout.build(None)
        if self.proj_dropout is not None:
            self.proj_dropout.build(padded_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass for the single window attention.

        Args:
            inputs: Input tensor of shape ``(B, N, C)``, where ``N`` must be
                less than or equal to ``window_size**2``.
            attention_mask: An optional mask of shape ``(B, N)`` to prevent
                attention to certain positions.
            training: Boolean indicating if the layer is in training mode.

        Returns:
            The output tensor after applying attention, with shape ``(B, N, C)``.
        """
        input_shape = keras.ops.shape(inputs)
        B_actual, N_actual, C_actual = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )
        N_target = self.window_size * self.window_size

        # --- PADDING & MASKING LOGIC ---
        padding_amount = N_target - N_actual
        padded_inputs = keras.ops.pad(
            inputs, [[0, 0], [0, padding_amount], [0, 0]]
        )

        internal_padding_mask = keras.ops.concatenate(
            [
                keras.ops.ones((B_actual, N_actual), dtype="int32"),
                keras.ops.zeros((B_actual, padding_amount), dtype="int32"),
            ],
            axis=1,
        )

        final_attention_mask = internal_padding_mask
        if attention_mask is not None:
            mask_padding = keras.ops.zeros(
                (B_actual, padding_amount), dtype=attention_mask.dtype
            )
            padded_user_mask = keras.ops.concatenate(
                [attention_mask, mask_padding], axis=1
            )
            final_attention_mask = (
                keras.ops.cast(padded_user_mask, "int32")
                * internal_padding_mask
            )

        # --- ATTENTION LOGIC ---
        B, N, C = keras.ops.shape(padded_inputs)
        qkv = self.qkv(padded_inputs, training=training)
        qkv = keras.ops.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim))
        qkv = keras.ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = keras.ops.matmul(q, keras.ops.transpose(k, (0, 1, 3, 2)))

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

        broadcast_mask = keras.ops.reshape(final_attention_mask, (B, 1, 1, N))
        inf_value = keras.ops.convert_to_tensor(-1e9, dtype=attn.dtype)
        additive_mask = (
            1.0 - keras.ops.cast(broadcast_mask, dtype=attn.dtype)
        ) * inf_value
        attn = attn + additive_mask
        attn = keras.ops.softmax(attn, axis=-1)

        if self.attn_dropout is not None:
            attn = self.attn_dropout(attn, training=training)

        x = keras.ops.matmul(attn, v)
        x = keras.ops.transpose(x, (0, 2, 1, 3))
        x = keras.ops.reshape(x, (B, N, C))
        x = self.proj(x, training=training)

        if self.proj_dropout is not None:
            x = self.proj_dropout(x, training=training)

        output = x[:, :N_actual, :]
        return output

    def get_config(self) -> Dict[str, Any]:
        """Serialize the layer's configuration.

        Returns:
            A dictionary containing the layer's configuration.
        """
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
                "qkv_bias": self.qkv_bias,
                "qk_scale": self.qk_scale,
                "attn_dropout_rate": self.attn_dropout_rate,
                "proj_dropout_rate": self.proj_dropout_rate,
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
    """Window Attention Layer.

    This layer implements the complete windowed multi-head self-attention
    mechanism. It takes a 1D sequence, reshapes it into a 2D grid, partitions
    the grid into windows, applies attention within each window, and merges
    the result back into a 1D sequence of the original length.

    Architecture:
    ~~~~~~~~~~~~~
    .. code-block::

        Input(B, N, C)
               ↓
        Pad sequence to form a square grid -> (B, N_grid, C)
               ↓
        Reshape to grid -> (B, H, W, C)
               ↓
        Pad grid for windowing -> (B, H_pad, W_pad, C)
               ↓
        Partition into windows -> (B*num_windows, win_size*win_size, C)
               ↓
        SingleWindowAttention (Core attention logic with relative position bias)
               ↓
        Reverse window partition -> (B, H_pad, W_pad, C)
               ↓
        Un-pad grid -> (B, H, W, C)
               ↓
        Flatten and un-pad sequence -> (B, N, C)
               ↓
        Output(B, N, C)

    Args:
        dim: Dimension of the input tokens (channels).
        window_size: The height and width of the attention window.
        num_heads: Number of attention heads.
        **kwargs: Keyword arguments for the internal
            :class:`SingleWindowAttention` layer, such as ``qkv_bias``,
            ``proj_dropout_rate``, etc.

    Input shape:
        A 3D tensor with shape ``(batch_size, sequence_length, dim)``.

    Output shape:
        A 3D tensor with the same shape as the input:
        ``(batch_size, sequence_length, dim)``.
    """

    def __init__(
        self, dim: int, window_size: int, num_heads: int, **kwargs: Any
    ):
        super().__init__(name=kwargs.pop("name", "window_attention"))
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        # All other configuration arguments are passed to the inner attention
        # layer.
        self.kwargs = kwargs

        # The core attention mechanism for a single window.
        self.attention = SingleWindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            **kwargs,
        )

    def _window_partition(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Partition a 4D grid tensor into windows.

        Args:
            x: A 4D grid tensor of shape ``(B, H, W, C)``.

        Returns:
            A tensor of partitioned windows with shape
            ``(B*num_windows, ws*ws, C)``.
        """
        B, H, W, C = keras.ops.shape(x)
        ws = self.window_size
        x = keras.ops.reshape(x, (B, H // ws, ws, W // ws, ws, C))
        x = keras.ops.transpose(x, (0, 1, 3, 2, 4, 5))
        windows = keras.ops.reshape(x, (-1, ws, ws, C))
        return windows

    def _window_reverse(
        self, windows: keras.KerasTensor, H: int, W: int
    ) -> keras.KerasTensor:
        """Merge windows back into a 4D grid tensor.

        Args:
            windows: A tensor of partitioned windows, shape
                ``(B*num_windows, ws, ws, C)``.
            H: The original height of the grid.
            W: The original width of the grid.

        Returns:
            The reconstructed 4D grid tensor of shape ``(B, H, W, C)``.
        """
        B = keras.ops.shape(windows)[0] // (
            (H // self.window_size) * (W // self.window_size)
        )
        ws = self.window_size
        x = keras.ops.reshape(windows, (B, H // ws, W // ws, ws, ws, -1))
        x = keras.ops.transpose(x, (0, 1, 3, 2, 4, 5))
        x = keras.ops.reshape(x, (B, H, W, -1))
        return x

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer's weights.

        Args:
            input_shape: Shape of the input tensor.
        """
        # Build the internal attention layer. It expects an input of shape
        # (batch, window_area, dim), where window_area = window_size**2
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
        """Forward pass for the window attention layer.

        Args:
            inputs: Input tensor of shape ``(B, N, C)``.
            attention_mask: An optional mask of shape ``(B, N)`` to prevent
                attention to certain sequence elements.
            training: Boolean indicating if the layer is in training mode.

        Returns:
            The output tensor with the same shape as the input.
        """
        input_shape = keras.ops.shape(inputs)
        B, N_actual, C = input_shape[0], input_shape[1], input_shape[2]
        ws = self.window_size

        # --- 1. Pad sequence to form a square grid ---
        H = W = keras.ops.cast(
            keras.ops.ceil(keras.ops.sqrt(keras.ops.cast(N_actual, "float32"))),
            "int32",
        )
        N_grid = H * W
        pad_len_seq = N_grid - N_actual

        padded_inputs = keras.ops.pad(
            inputs, [[0, 0], [0, pad_len_seq], [0, 0]]
        )
        grid = keras.ops.reshape(padded_inputs, (B, H, W, C))

        # Also pad the attention mask if it exists
        padded_mask = None
        if attention_mask is not None:
            padded_mask = keras.ops.pad(
                attention_mask, [[0, 0], [0, pad_len_seq]]
            )

        # --- 2. Pad grid so its dimensions are divisible by window_size ---
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws

        padded_grid = keras.ops.pad(
            grid, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]]
        )
        H_pad, W_pad = H + pad_h, W + pad_w

        # --- 3. Partition into windows ---
        windows = self._window_partition(padded_grid)
        windows = keras.ops.reshape(windows, (-1, ws * ws, C))

        # Partition the mask as well
        attn_mask_for_windows = None
        if padded_mask is not None:
            mask_grid = keras.ops.reshape(padded_mask, (B, H, W, 1))
            padded_mask_grid = keras.ops.pad(
                mask_grid,
                [[0, 0], [0, pad_h], [0, pad_w], [0, 0]],
                constant_values=0,
            )
            mask_windows = self._window_partition(padded_mask_grid)
            attn_mask_for_windows = keras.ops.reshape(
                mask_windows, (-1, ws * ws)
            )

        # --- 4. Apply attention within windows ---
        attn_windows = self.attention(
            windows, attention_mask=attn_mask_for_windows, training=training
        )

        # --- 5. Reverse window partition ---
        attn_windows = keras.ops.reshape(attn_windows, (-1, ws, ws, C))
        reconstructed_grid = self._window_reverse(attn_windows, H_pad, W_pad)

        # --- 6. Un-pad grid and flatten back to sequence ---
        grid_unpadded = reconstructed_grid[:, :H, :W, :]
        sequence_unpadded = keras.ops.reshape(grid_unpadded, (B, N_grid, C))

        # --- 7. Un-pad sequence to original length ---
        output = sequence_unpadded[:, :N_actual, :]

        return output

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: The shape of the input tensor.

        Returns:
            The shape of the output tensor, which is the same as input_shape.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Serialize the layer's configuration.

        Returns:
            A dictionary containing the layer's configuration.
        """
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
            }
        )
        # Add all kwargs passed to the inner layer for complete serialization.
        config.update(self.kwargs)
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "WindowAttention":
        """Create a layer from its config.

        Args:
            config: A dictionary containing the layer's configuration.

        Returns:
            A new instance of the layer.
        """
        # Create a copy to avoid mutating the original config dict, which is
        # often used for comparison in tests after deserialization.
        config_copy = config.copy()

        # Pop the main args from the copy and pass the rest as kwargs to
        # __init__.
        dim = config_copy.pop("dim")
        window_size = config_copy.pop("window_size")
        num_heads = config_copy.pop("num_heads")
        return cls(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            **config_copy,
        )


# ---------------------------------------------------------------------