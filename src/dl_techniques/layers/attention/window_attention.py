"""
Unified windowed multi-head self-attention for sequence processing.

This module provides a highly configurable windowed multi-head self-attention
layer. It unifies two distinct partitioning strategies—standard grid-based
windowing and frequency-proximate zigzag windowing—into a single interface,
controlled by the `partition_mode` parameter.

The layer takes a 1D sequence, internally reshapes it, partitions it
according to the chosen mode, and computes self-attention within each local
window. This approach maintains linear complexity with respect to sequence
length while offering different locality biases.

================================================
Partition Mode 1: 'grid' (Swin Transformer-style)
================================================

Complete Architecture Flow (`partition_mode='grid'`)::

    INPUT: 1D Sequence [batch, N, dim]
      │
      ├─ Step 1: Grid Formation (Pad N → H×W, Reshape)
      ↓
    2D Grid [batch, H, W, dim]
      │
      ├─ Step 2: Window Padding (Pad grid to be divisible by window_size)
      ↓
    Padded Grid [batch, H_pad, W_pad, dim]
      │
      ├─ Step 3: Window Partitioning (Split into non-overlapping windows)
      ↓
    Windows [batch×num_windows, ws², dim]
      │
      ├─ Step 4: Local Self-Attention (With relative position bias)
      ↓
    Attended Windows [batch×num_windows, ws², dim]
      │
      ├─ Step 5: Window Merging (Reverse partition)
      ↓
    Grid [batch, H_pad, W_pad, dim]
      │
      ├─ Step 6: Unpad & Flatten (Restore original sequence)
      ↓
    OUTPUT: 1D Sequence [batch, N, dim]

================================================
Partition Mode 2: 'zigzag' (Frequency Locality)
================================================

Complete Architecture Flow (`partition_mode='zigzag'`)::

    INPUT: 1D Sequence [batch, N, dim]
      │
      ├─ Step 1: Grid Formation (Pad N → H×W)
      ↓
    Grid Sequence [batch, H×W, dim]
      │
      ├─ Step 2: Zigzag Reordering (Group frequency-proximate tokens)
      ↓
    Zigzag Sequence [batch, H×W, dim]
      │
      ├─ Step 3: Window Partitioning (Split reordered sequence into windows)
      ↓
    Windows [batch×num_windows, win_size², dim]
      │
      ├─ Step 4: Local Self-Attention (Relative position bias often disabled)
      ↓
    Attended Windows [batch×num_windows, win_size², dim]
      │
      ├─ Step 5: Inverse Zigzag (Restore grid order)
      ↓
    Grid Sequence [batch, H×W, dim]
      │
      ├─ Step 6: Unpad (Restore original sequence length)
      ↓
    OUTPUT: 1D Sequence [batch, N, dim]

Complexity Analysis
-------------------

============== ============== =================================
Operation      Complexity     Notes
============== ============== =================================
Partitioning   O(N)           Linear reshape/reorder
Attention      O(N × k²)      k = window_size, N = seq_len
Total          O(N × k²)      Linear in sequence length
============== ============== =================================
"""

import math
import keras
from typing import Any, Dict, Literal, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .single_window_attention import SingleWindowAttention

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class WindowAttention(keras.layers.Layer):
    """
    Unified window-based multi-head self-attention layer.

    This layer implements an efficient self-attention mechanism by partitioning
    an input sequence into local windows and applying attention within them.
    It supports two partitioning strategies:
    1.  `'grid'`: Standard spatial windowing inspired by Swin Transformer.
    2.  `'zigzag'`: Reorders the sequence based on a 2D zigzag scan to group
        frequency-proximate tokens before windowing.

    All internal padding, reshaping, partitioning, and merging operations are
    handled automatically.

    **Core Features (Configurable)**:

    1.  **Partitioning Mode (`partition_mode`)**:
        -   `'grid'` (default): Reshapes the sequence into a 2D grid and
            partitions it into non-overlapping spatial windows. Best for
            tasks where 2D spatial locality is important.
        -   `'zigzag'`: Reorders the sequence based on a 2D zigzag scan,
            grouping tokens with similar frequency-domain characteristics
            (inspired by JPEG). This induces a frequency-based locality bias.

    2.  **Attention Mechanism**: All attention-related parameters are exposed
        directly from the underlying `SingleWindowAttention` layer, allowing
        fine-grained control over the attention mechanism (e.g., `attention_mode`,
        `normalization`, `use_relative_position_bias`).

    :param dim: Dimension of the input tokens (channels).
    :type dim: int
    :param window_size: The height and width of the attention window.
    :type window_size: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param partition_mode: The partitioning strategy. One of `'grid'` or
        `'zigzag'`. Default: 'grid'.
    :type partition_mode: Literal["grid", "zigzag"]
    :param attention_mode: The type of attention projection in each window. One
        of `'linear'` or `'kan_key'`. Default: 'linear'.
    :type attention_mode: Literal["linear", "kan_key"]
    :param normalization: The normalization method for attention scores. One of
        `'softmax'`, `'adaptive_softmax'`, or `'hierarchical_routing'`.
        Default: 'softmax'.
    :type normalization: Literal["softmax", "adaptive_softmax", "hierarchical_routing"]
    :param use_relative_position_bias: If True, add a learnable relative
        position bias to the attention scores. Recommended for `'grid'` mode.
        For `'zigzag'` mode, this is often set to `False` as the spatial
        relationship is already altered. Default: True.
    :type use_relative_position_bias: bool
    :param qkv_bias: If True, add a learnable bias to the QKV projection.
        Only used when `attention_mode` is `'linear'`. Default: True.
    :type qkv_bias: bool
    :param qk_scale: Override for query-key scaling factor. Defaults to
        `head_dim ** -0.5`.
    :type qk_scale: Optional[float]
    :param dropout_rate: Dropout rate for attention scores. Default: 0.0.
    :type dropout_rate: float
    :param proj_bias: If True, add a learnable bias to the output projection.
        Default: True.
    :type proj_bias: bool
    :param kan_grid_size: Grid size for the KAN layer. Only used when
        `attention_mode` is `'kan_key'`. Default: 5.
    :type kan_grid_size: int
    :param kan_spline_order: Spline order for the KAN layer. Only used when
        `attention_mode` is `'kan_key'`. Default: 3.
    :type kan_spline_order: int
    :param kan_activation: Activation for the KAN layer. Only used when
        `attention_mode` is `'kan_key'`. Default: 'swish'.
    :type kan_activation: str
    :param adaptive_softmax_config: Configuration for adaptive softmax. Only
        used when `normalization` is `'adaptive_softmax'`. Default: None.
    :type adaptive_softmax_config: Optional[Dict[str, Any]]
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
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        partition_mode: Literal["grid", "zigzag"] = "grid",
        attention_mode: Literal["linear", "kan_key"] = "linear",
        normalization: Literal[
            "softmax", "adaptive_softmax", "hierarchical_routing"
        ] = "softmax",
        use_relative_position_bias: bool = True,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        dropout_rate: float = 0.0,
        proj_bias: bool = True,
        kan_grid_size: int = 5,
        kan_spline_order: int = 3,
        kan_activation: str = "swish",
        adaptive_softmax_config: Optional[Dict[str, Any]] = None,
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
    ):
        super().__init__(**kwargs)

        # Store all parameters for get_config()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.partition_mode = partition_mode
        self.attention_mode = attention_mode
        self.normalization = normalization
        self.use_relative_position_bias = use_relative_position_bias
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.dropout_rate = dropout_rate
        self.proj_bias = proj_bias
        self.kan_grid_size = kan_grid_size
        self.kan_spline_order = kan_spline_order
        self.kan_activation = kan_activation
        self.adaptive_softmax_config = adaptive_softmax_config
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        # placeholder
        self._call_internal = None
        # Create the core attention layer that operates on a single window
        self.attention = SingleWindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            attention_mode=attention_mode,
            normalization=normalization,
            use_relative_position_bias=use_relative_position_bias,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            dropout_rate=dropout_rate,
            proj_bias=proj_bias,
            kan_grid_size=kan_grid_size,
            kan_spline_order=kan_spline_order,
            kan_activation=kan_activation,
            adaptive_softmax_config=adaptive_softmax_config,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )

        # Attributes for zigzag mode, computed in build()
        if self.partition_mode == "zigzag":
            self.H = None
            self.W = None
            self.N_grid = None
            self.pad_len_seq = None
            self.zigzag_indices = None
            self.inverse_zigzag_indices = None

    @staticmethod
    def _generate_zigzag_indices(
        H: keras.KerasTensor, W: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Generate zigzag scan indices for an HxW grid."""
        r_coords = keras.ops.arange(0, H, dtype="int32")
        c_coords = keras.ops.arange(0, W, dtype="int32")
        r_grid, c_grid = keras.ops.meshgrid(r_coords, c_coords, indexing="ij")

        r_flat = keras.ops.reshape(r_grid, (-1,))
        c_flat = keras.ops.reshape(c_grid, (-1,))

        s = r_flat + c_flat
        secondary_key = keras.ops.where(s % 2 == 1, r_flat, H - 1 - r_flat)
        combined_key = s * H + secondary_key

        return keras.ops.argsort(combined_key)

    def _window_partition(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Partition a 4D grid tensor into windows."""
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
        ws = self.window_size
        num_windows_h = H // ws
        num_windows_w = W // ws
        num_windows_total = keras.ops.shape(windows)[0]
        B = num_windows_total // (num_windows_h * num_windows_w)
        x = keras.ops.reshape(
            windows, (B, num_windows_h, num_windows_w, ws, ws, -1)
        )
        x = keras.ops.transpose(x, (0, 1, 3, 2, 4, 5))
        x = keras.ops.reshape(x, (B, H, W, -1))
        return x

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build layer and precompute zigzag indices if needed."""
        if self.partition_mode == "zigzag":
            N_actual = input_shape[1]
            if N_actual is None:
                raise ValueError(
                    "WindowAttention with partition_mode='zigzag' requires a "
                    "fixed sequence length for XLA compatibility."
                )

            self.H = int(math.ceil(math.sqrt(N_actual)))
            self.W = self.H
            self.N_grid = self.H * self.W
            self.pad_len_seq = self.N_grid - N_actual

            H_tensor = keras.ops.convert_to_tensor(self.H, dtype="int32")
            W_tensor = keras.ops.convert_to_tensor(self.W, dtype="int32")
            self.zigzag_indices = self._generate_zigzag_indices(
                H_tensor, W_tensor
            )
            self.inverse_zigzag_indices = keras.ops.argsort(
                self.zigzag_indices
            )

        self.attention.build(
            (None, self.window_size * self.window_size, self.dim)
        )

        if self.partition_mode == "grid":
            self._call_internal = self._call_grid
        elif self.partition_mode == "zigzag":
            self._call_internal = self._call_zigzag
        else:
            # Should not be reachable due to __init__ validation
            raise RuntimeError(f"Invalid partition mode: {self.partition_mode}")
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        return self._call_internal(inputs, attention_mask, training)

    def _call_grid(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass for 'grid' partitioning."""
        input_shape = keras.ops.shape(inputs)
        B, N_actual, C = input_shape[0], input_shape[1], input_shape[2]
        ws = self.window_size

        H = W = keras.ops.cast(
            keras.ops.ceil(
                keras.ops.sqrt(keras.ops.cast(N_actual, "float32"))
            ),
            "int32",
        )
        N_grid = H * W
        pad_amount_seq = keras.ops.maximum(0, N_grid - N_actual)

        x = keras.ops.pad(inputs, [[0, 0], [0, pad_amount_seq], [0, 0]])
        x = keras.ops.reshape(x, (B, H, W, C))

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = keras.ops.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
        H_pad, W_pad = H + pad_h, W + pad_w

        windows = self._window_partition(x)
        windows = keras.ops.reshape(windows, (-1, ws * ws, C))

        window_mask = None
        if attention_mask is not None:
            mask = keras.ops.pad(attention_mask, [[0, 0], [0, pad_amount_seq]])
            mask = keras.ops.reshape(mask, (B, H, W))
            mask = keras.ops.pad(mask, [[0, 0], [0, pad_h], [0, pad_w]])
            mask = keras.ops.expand_dims(mask, axis=-1)
            mask_windows = self._window_partition(mask)
            window_mask = keras.ops.reshape(mask_windows, (-1, ws * ws))

        attn_windows = self.attention(
            windows, attention_mask=window_mask, training=training
        )
        attn_windows = keras.ops.reshape(attn_windows, (-1, ws, ws, C))
        x = self._window_reverse(attn_windows, H_pad, W_pad)

        x = x[:, :H, :W, :]
        x = keras.ops.reshape(x, (B, N_grid, C))
        x = x[:, :N_actual, :]
        return x

    def _call_zigzag(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass for 'zigzag' partitioning."""
        input_shape = keras.ops.shape(inputs)
        B, N_actual, C = input_shape[0], input_shape[1], input_shape[2]
        win_len = self.window_size * self.window_size

        padded_inputs = keras.ops.pad(
            inputs, [[0, 0], [0, self.pad_len_seq], [0, 0]]
        )

        zigzag_sequence = keras.ops.take(
            padded_inputs, self.zigzag_indices, axis=1
        )

        zigzag_mask = None
        if attention_mask is not None:
            padded_mask = keras.ops.pad(
                attention_mask, [[0, 0], [0, self.pad_len_seq]]
            )
            zigzag_mask = keras.ops.take(
                padded_mask, self.zigzag_indices, axis=1
            )

        pad_len_win = (win_len - (self.N_grid % win_len)) % win_len
        padded_zigzag_seq = keras.ops.pad(
            zigzag_sequence, [[0, 0], [0, pad_len_win], [0, 0]]
        )

        num_windows = (self.N_grid + pad_len_win) // win_len
        windows = keras.ops.reshape(
            padded_zigzag_seq, (B * num_windows, win_len, C)
        )

        attn_mask_for_windows = None
        if zigzag_mask is not None:
            padded_zigzag_mask = keras.ops.pad(
                zigzag_mask, [[0, 0], [0, pad_len_win]], constant_values=0
            )
            attn_mask_for_windows = keras.ops.reshape(
                padded_zigzag_mask, (B * num_windows, win_len)
            )

        attn_windows = self.attention(
            windows, attention_mask=attn_mask_for_windows, training=training
        )

        merged_zigzag_seq = keras.ops.reshape(
            attn_windows, (B, num_windows * win_len, C)
        )
        unpadded_zigzag_seq = merged_zigzag_seq[:, : self.N_grid, :]

        sequence_unpadded = keras.ops.take(
            unpadded_zigzag_seq, self.inverse_zigzag_indices, axis=1
        )

        output = sequence_unpadded[:, :N_actual, :]
        return output

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape, which is identical to the input shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Serialize the layer's configuration."""
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
                "partition_mode": self.partition_mode,
                "attention_mode": self.attention_mode,
                "normalization": self.normalization,
                "use_relative_position_bias": self.use_relative_position_bias,
                "qkv_bias": self.qkv_bias,
                "qk_scale": self.qk_scale,
                "dropout_rate": self.dropout_rate,
                "proj_bias": self.proj_bias,
                "kan_grid_size": self.kan_grid_size,
                "kan_spline_order": self.kan_spline_order,
                "kan_activation": self.kan_activation,
                "adaptive_softmax_config": self.adaptive_softmax_config,
                "kernel_initializer": keras.initializers.serialize(
                    keras.initializers.get(self.kernel_initializer)
                ),
                "bias_initializer": keras.initializers.serialize(
                    keras.initializers.get(self.bias_initializer)
                ),
                "kernel_regularizer": keras.regularizers.serialize(
                    keras.regularizers.get(self.kernel_regularizer)
                ),
                "bias_regularizer": keras.regularizers.serialize(
                    keras.regularizers.get(self.bias_regularizer)
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "WindowAttention":
        """Create a layer from its configuration."""
        return cls(**config)

# ---------------------------------------------------------------------

"""
Utility functions for creating common variations of the WindowAttention layer.

This module provides factory functions that simplify the creation of specialized
WindowAttention layers by pre-configuring them for common use cases. This
promotes code readability and reduces boilerplate when experimenting with
different attention architectures.

Each function wraps the core `WindowAttention` layer, setting specific
parameters like `partition_mode`, `attention_mode`, or `normalization`
to sensible defaults for that particular variant. All other parameters can
still be overridden via keyword arguments.

Available Factories:
--------------------
- `create_grid_window_attention`:
    Standard Swin Transformer-style spatial windowing.

- `create_zigzag_window_attention`:
    Windowing on a zigzag-reordered sequence for frequency locality.

- `create_kan_key_window_attention`:
    Window attention using a non-linear KAN layer for the Key projection.

- `create_adaptive_softmax_window_attention`:
    Window attention with adaptive temperature softmax for better calibration.
"""

# ---------------------------------------------------------------------


def create_grid_window_attention(
    dim: int, window_size: int, num_heads: int, **kwargs: Any
) -> WindowAttention:
    """
    Creates a standard spatial window attention layer (Swin-style).

    This factory configures `WindowAttention` for grid-based partitioning,
    which is ideal for tasks where 2D spatial locality is important.
    It defaults to using relative position bias, as is standard for this
    architecture.

    :param dim: Dimension of the input tokens.
    :type dim: int
    :param window_size: The height and width of the attention window.
    :type window_size: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param kwargs: Additional keyword arguments to pass to the `WindowAttention`
        constructor (e.g., `dropout_rate`, `qkv_bias`).
    :type kwargs: Any
    :return: A `WindowAttention` layer configured for grid partitioning.
    :rtype: WindowAttention
    """
    # Default to using relative position bias, but allow override.
    kwargs.setdefault("use_relative_position_bias", True)

    return WindowAttention(
        dim=dim,
        window_size=window_size,
        num_heads=num_heads,
        partition_mode="grid",
        **kwargs,
    )

# ---------------------------------------------------------------------


def create_zigzag_window_attention(
    dim: int, window_size: int, num_heads: int, **kwargs: Any
) -> WindowAttention:
    """
    Creates a window attention layer with zigzag partitioning.

    This factory configures `WindowAttention` to reorder the sequence
    along a 2D zigzag path before windowing. This groups frequency-proximate
    tokens, inducing a locality bias sensitive to frequency bands. It defaults
    to *disabling* relative position bias, as the original spatial grid is
    intentionally broken by the zigzag scan.

    :param dim: Dimension of the input tokens.
    :type dim: int
    :param window_size: The height and width of the attention window.
    :type window_size: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param kwargs: Additional keyword arguments to pass to the `WindowAttention`
        constructor (e.g., `dropout_rate`, `proj_bias`).
    :type kwargs: Any
    :return: A `WindowAttention` layer configured for zigzag partitioning.
    :rtype: WindowAttention
    """
    # Default to disabling relative position bias, but allow override.
    kwargs.setdefault("use_relative_position_bias", False)

    return WindowAttention(
        dim=dim,
        window_size=window_size,
        num_heads=num_heads,
        partition_mode="zigzag",
        **kwargs,
    )

# ---------------------------------------------------------------------


def create_kan_key_window_attention(
    dim: int,
    window_size: int,
    num_heads: int,
    partition_mode: Literal["grid", "zigzag"] = "grid",
    **kwargs: Any,
) -> WindowAttention:
    """
    Creates a window attention layer with a non-linear KAN Key projection.

    This factory configures `WindowAttention` to use a `KANLinear` layer for
    projecting the Key tensor. This allows for more expressive similarity
    matching compared to a standard linear projection.

    :param dim: Dimension of the input tokens.
    :type dim: int
    :param window_size: The height and width of the attention window.
    :type window_size: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param partition_mode: The partitioning strategy (`'grid'` or `'zigzag'`).
        Default: 'grid'.
    :type partition_mode: Literal["grid", "zigzag"]
    :param kwargs: Additional keyword arguments to pass to `WindowAttention`,
        especially KAN-specific ones like `kan_grid_size`,
        `kan_spline_order`.
    :type kwargs: Any
    :return: A `WindowAttention` layer with a KAN-based Key projection.
    :rtype: WindowAttention
    """
    return WindowAttention(
        dim=dim,
        window_size=window_size,
        num_heads=num_heads,
        partition_mode=partition_mode,
        attention_mode="kan_key",
        **kwargs,
    )

# ---------------------------------------------------------------------


def create_adaptive_softmax_window_attention(
    dim: int,
    window_size: int,
    num_heads: int,
    partition_mode: Literal["grid", "zigzag"] = "grid",
    **kwargs: Any,
) -> WindowAttention:
    """
    Creates a window attention layer with adaptive temperature softmax.

    This factory configures `WindowAttention` to use `AdaptiveTemperatureSoftmax`
    for normalization. This can improve model calibration and performance by
    dynamically adjusting the sharpness of the attention distribution based on
    model confidence.

    :param dim: Dimension of the input tokens.
    :type dim: int
    :param window_size: The height and width of the attention window.
    :type window_size: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param partition_mode: The partitioning strategy (`'grid'` or `'zigzag'`).
        Default: 'grid'.
    :type partition_mode: Literal["grid", "zigzag"]
    :param kwargs: Additional keyword arguments to pass to `WindowAttention`,
        especially `adaptive_softmax_config`.
    :type kwargs: Any
    :return: A `WindowAttention` layer with adaptive softmax normalization.
    :rtype: WindowAttention
    """
    return WindowAttention(
        dim=dim,
        window_size=window_size,
        num_heads=num_heads,
        partition_mode=partition_mode,
        normalization="adaptive_softmax",
        **kwargs,
    )

# ---------------------------------------------------------------------
