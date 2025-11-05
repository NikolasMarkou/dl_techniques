"""
KAN Window Attention Layer.

This module introduces `WindowAttentionKAN`, a novel attention mechanism that
integrates principles of Kolmogorov-Arnold Networks (KAN) into the efficient
windowed attention framework.

This implementation enhances the standard attention mechanism by replacing the
linear projection for the Key (K) with a `KANLinear` layer. This allows the
model to learn a complex, non-linear transformation from a token's content to
its "key" representation. The Query (Q) and Value (V) projections remain
standard linear transformations, creating a powerful hybrid attention mechanism.

The core idea is to give the model more expressive power in determining token
similarity. Instead of relying on a simple linear dot product, the attention
score is computed between a linear query and a non-linearly transformed key,
enabling the capture of more intricate relationships within the data.
"""

import keras
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..kan_linear import KANLinear

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SingleWindowAttentionKAN(keras.layers.Layer):
    """
    Multi-head self-attention with a KAN projection for the Key.

    This layer implements the core attention mechanism for a single window,
    replacing the standard Key (K) dense projection with a learnable KAN
    (Kolmogorov-Arnold Network) transformation. The Query (Q) and Value (V)
    projections remain as standard linear layers. This hybrid approach allows
    for a more expressive calculation of attention scores by learning a
    non-linear, data-driven mapping for the key.

    **Intent**: Enhance windowed self-attention by replacing the fixed linear
    projection for the Key with an adaptive, non-linear transformation via a
    KAN layer. This enables the model to learn more complex similarity metrics
    between tokens, while maintaining the efficiency of local attention.

    **Architecture**:
    ```
    Input(shape=[batch * num_windows, window_area, dim])
              ↓
    ┌─────────────────────────────────────────┐
    │  Hybrid Projections:                    │
    │  Q = Dense_q(input)         (Linear)    │
    │  K = KANLinear_k(input)     (Non-linear)│
    │  V = Dense_v(input)         (Linear)    │
    └─────────────────────────────────────────┘
              ↓
    Reshape & Transpose to (batch, heads, seq, head_dim)
              ↓
    Attention Scores = (Q @ K^T) / √d_k + RelativeBias
              ↓
    Apply Attention Mask (if provided)
              ↓
    Attention Weights = Softmax(Scores) → Dropout
              ↓
    Context = Attention_Weights @ V
              ↓
    Reshape & Project: Dense(dim)
              ↓
    Output(shape=[batch * num_windows, window_area, dim])
    ```

    **Note on KANLinear Implementation**:
    The `KANLinear` layer used here learns its non-linear functions by
    parameterizing them as a linear combination of basis functions (e.g.,
    Gaussian RBFs), rather than piecewise B-splines.

    Args:
        dim: Integer, dimension of input tokens (channels). Must be positive and
            divisible by num_heads.
        window_size: Integer, height and width of the square attention window.
            Must be positive.
        num_heads: Integer, number of parallel attention heads. Must be positive
            and evenly divide dim.
        kan_grid_size: Integer, size of the grid for KAN basis functions.
            Defaults to 5.
        kan_spline_order: Integer, order of the basis functions. Defaults to 3.
        kan_activation: String, base activation function for the KAN layer.
            Defaults to 'swish'.
        qk_scale: Optional float, override for query-key scaling factor.
            If None, uses `head_dim ** -0.5`. Defaults to None.
        dropout_rate: Float between 0 and 1, dropout rate for attention
            weights. Defaults to 0.0.
        proj_dropout_rate: Float between 0 and 1, dropout rate for the final
            output projection. Defaults to 0.0.
        proj_bias: Boolean, if True, add a learnable bias to the output
            projection layer. Defaults to True.
        kernel_initializer: Initializer for the output projection kernel weights.
            Defaults to 'glorot_uniform'.
        bias_initializer: Initializer for the output projection bias weights.
            Defaults to 'zeros'.
        kernel_regularizer: Optional regularizer for output projection kernel.
            Defaults to None.
        bias_regularizer: Optional regularizer for output projection bias.
            Defaults to None.
        **kwargs: Additional keyword arguments for base Layer class.

    Input shape:
        3D tensor: `(batch_size, sequence_length, dim)`, where
        `sequence_length <= window_size**2`.

    Output shape:
        3D tensor with same shape as input: `(batch_size, sequence_length, dim)`.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        kan_grid_size: int = 5,
        kan_spline_order: int = 3,
        kan_activation: str = "swish",
        qk_scale: Optional[float] = None,
        dropout_rate: float = 0.0,
        proj_bias: bool = True,
        kernel_initializer: Union[
            str, keras.initializers.Initializer
        ] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
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
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(
                "attn_dropout_rate must be between 0.0 and 1.0, "
                f"got {dropout_rate}"
            )

        if kan_grid_size <= 0:
            raise ValueError(
                f"kan_grid_size must be positive, got {kan_grid_size}"
            )
        if kan_spline_order <= 0:
            raise ValueError(
                f"kan_spline_order must be positive, got {kan_spline_order}"
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
        self.dropout_rate = dropout_rate
        self.proj_bias = proj_bias
        self.kan_grid_size = kan_grid_size
        self.kan_spline_order = kan_spline_order
        self.kan_activation = kan_activation
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

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

        # --- Create sub-layers ---
        # Query and Value use standard linear projections.
        self.query = keras.layers.Dense(
            self.dim,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="query",
        )
        self.value = keras.layers.Dense(
            self.dim,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="value",
        )
        # Key uses a KANLinear layer for a non-linear projection.
        self.key = KANLinear(
            features=self.dim,
            grid_size=self.kan_grid_size,
            spline_order=self.kan_spline_order,
            activation=self.kan_activation,
            name="key",  # Corrected typo: was "value"
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

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer's weights and sub-layers."""
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
        self.query.build(padded_shape)
        self.key.build(padded_shape)
        self.value.build(padded_shape)
        self.proj.build(padded_shape)

        if self.attn_dropout is not None:
            self.attn_dropout.build(None)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass for single KAN window attention."""
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

        # --- ATTENTION COMPUTATION ---
        B, N, C = keras.ops.shape(padded_inputs)

        # Q, K, V projections: Q/V are linear, K is non-linear via KAN.
        q_proj = self.query(padded_inputs, training=training)
        k_proj = self.key(padded_inputs, training=training)
        v_proj = self.value(padded_inputs, training=training)

        # Reshape for multi-head attention
        q = keras.ops.reshape(q_proj, (B, N, self.num_heads, self.head_dim))
        q = keras.ops.transpose(q, (0, 2, 1, 3))
        k = keras.ops.reshape(k_proj, (B, N, self.num_heads, self.head_dim))
        k = keras.ops.transpose(k, (0, 2, 1, 3))
        v = keras.ops.reshape(v_proj, (B, N, self.num_heads, self.head_dim))
        v = keras.ops.transpose(v, (0, 2, 1, 3))

        # Scaled dot-product attention
        q = q * self.scale
        attn = keras.ops.matmul(q, keras.ops.transpose(k, (0, 1, 3, 2)))

        # Add learned relative position bias
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
        broadcast_mask = keras.ops.reshape(final_attention_mask, (B, 1, 1, N))
        inf_value = keras.ops.convert_to_tensor(-1e9, dtype=attn.dtype)
        additive_mask = (
            1.0 - keras.ops.cast(broadcast_mask, dtype=attn.dtype)
        ) * inf_value
        attn = attn + additive_mask

        # Softmax and optional dropout
        attn = keras.ops.softmax(attn, axis=-1)
        if self.attn_dropout is not None:
            attn = self.attn_dropout(attn, training=training)

        # Aggregate values and project
        x = keras.ops.matmul(attn, v)
        x = keras.ops.transpose(x, (0, 2, 1, 3))
        x = keras.ops.reshape(x, (B, N, C))
        x = self.proj(x, training=training)

        # Remove padding
        output = x[:, :N_actual, :]
        return output

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape, which is identical to input shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for layer serialization."""
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
                "kan_grid_size": self.kan_grid_size,
                "kan_spline_order": self.kan_spline_order,
                "kan_activation": self.kan_activation,
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
class WindowAttentionKAN(keras.layers.Layer):
    """
    KAN-enhanced windowed multi-head self-attention for sequences.

    This layer implements efficient windowed attention by applying a hybrid
    attention mechanism within local windows. It takes a 1D sequence, reshapes
    it into a 2D grid, partitions the grid, applies attention using a non-linear
    KAN projection for the Key, and merges the results back into a 1D sequence.

    **Intent**: Provide an efficient, expressive attention mechanism for long
    sequences by combining the computational benefits of local windowed
    attention with the enhanced representational power of a KAN-based Key
    projection.

    **Architecture**:
    ```
    Input: 1D Sequence (batch, seq_len, dim)
              ↓
    Reshape to 2D Grid & Pad for Windowing
              ↓
    Partition into Windows
              ↓
    ┌──────────────────────────────────────┐
    │  For Each Window (in parallel):      │
    │  • Q = Linear Projection             │
    │  • K = Non-linear KAN Projection     │
    │  • V = Linear Projection             │
    │  • Multi-Head Attention + Rel. Bias  │
    └──────────────────────────────────────┘
              ↓
    Merge Windows Back to Grid & Un-pad
              ↓
    Reshape to 1D Sequence
              ↓
    Output: (batch, seq_len, dim)
    ```

    **Compared to Standard Window Attention**:
    - Traditional: `Q, K, V = Dense(X)` (all linear projections)
    - KAN Version: `Q, V = Dense(X)`, `K = KANLinear(X)`
    - Benefit: More expressive projections can capture complex content-based
      relationships for similarity scoring.

    Args:
        dim: Integer, dimension of input tokens (channels).
        window_size: Integer, height and width of square attention windows.
        num_heads: Integer, number of parallel attention heads.
        **kwargs: All other keyword arguments are passed to the internal
            `SingleWindowAttentionKAN` layer (e.g., `kan_grid_size`,
            `attn_dropout_rate`).

    Input shape:
        3D tensor: `(batch_size, sequence_length, dim)`.

    Output shape:
        3D tensor with same shape as input: `(batch_size, sequence_length, dim)`.
    """

    def __init__(
        self, dim: int, window_size: int, num_heads: int, **kwargs: Any
    ) -> None:
        layer_name = kwargs.pop("name", "kan_window_attention")
        super().__init__(name=layer_name)

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.kwargs = kwargs

        self.attention = SingleWindowAttentionKAN(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            **kwargs,
        )

    def _window_partition(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Partition a 4D grid tensor into non-overlapping windows."""
        B, H, W, C = keras.ops.shape(x)
        ws = self.window_size
        x = keras.ops.reshape(x, (B, H // ws, ws, W // ws, ws, C))
        x = keras.ops.transpose(x, (0, 1, 3, 2, 4, 5))
        windows = keras.ops.reshape(x, (-1, ws, ws, C))
        return windows

    def _window_reverse(
        self, windows: keras.KerasTensor, H: int, W: int
    ) -> keras.KerasTensor:
        """Merge windows back into a 4D grid tensor."""
        B = keras.ops.shape(windows)[0] // (
            (H // self.window_size) * (W // self.window_size)
        )
        ws = self.window_size
        x = keras.ops.reshape(windows, (B, H // ws, W // ws, ws, ws, -1))
        x = keras.ops.transpose(x, (0, 1, 3, 2, 4, 5))
        x = keras.ops.reshape(x, (B, H, W, -1))
        return x

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer's sub-layers."""
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
        """Forward pass for KAN window attention."""
        input_shape = keras.ops.shape(inputs)
        B, N_actual, C = input_shape[0], input_shape[1], input_shape[2]
        ws = self.window_size

        # --- SEQUENCE TO GRID CONVERSION ---
        H = W = keras.ops.cast(
            keras.ops.ceil(
                keras.ops.sqrt(keras.ops.cast(N_actual, "float32"))
            ),
            "int32",
        )
        N_grid = H * W
        pad_len_seq = N_grid - N_actual

        padded_inputs = keras.ops.pad(
            inputs, [[0, 0], [0, pad_len_seq], [0, 0]]
        )
        grid = keras.ops.reshape(padded_inputs, (B, H, W, C))

        padded_mask = None
        if attention_mask is not None:
            padded_mask = keras.ops.pad(
                attention_mask, [[0, 0], [0, pad_len_seq]]
            )

        # --- GRID PADDING FOR WINDOWING ---
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        padded_grid = keras.ops.pad(
            grid, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]]
        )
        H_pad, W_pad = H + pad_h, W + pad_w

        # --- WINDOW PARTITIONING ---
        windows = self._window_partition(padded_grid)
        windows = keras.ops.reshape(windows, (-1, ws * ws, C))

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

        # --- APPLY KAN ATTENTION WITHIN WINDOWS ---
        attn_windows = self.attention(
            windows, attention_mask=attn_mask_for_windows, training=training
        )

        # --- WINDOW MERGING AND RECONSTRUCTION ---
        attn_windows = keras.ops.reshape(attn_windows, (-1, ws, ws, C))
        reconstructed_grid = self._window_reverse(attn_windows, H_pad, W_pad)

        grid_unpadded = reconstructed_grid[:, :H, :W, :]
        sequence_unpadded = keras.ops.reshape(grid_unpadded, (B, N_grid, C))
        output = sequence_unpadded[:, :N_actual, :]

        return output

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape, which is identical to input shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for layer serialization."""
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
            }
        )
        config.update(self.kwargs)
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "WindowAttentionKAN":
        """Create layer from configuration dictionary."""
        config_copy = config.copy()
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