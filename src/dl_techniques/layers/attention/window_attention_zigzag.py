"""WindowZigZagAttention Layer.

A Keras layer implementing a configurable windowed multi-head self-attention.
This layer takes a 1D sequence of tokens, internally reshapes it into a 2D
grid, partitions the grid into windows, and computes self-attention within each
window.

The key distinction of this layer is its use of a zigzag pattern for calculating
relative position biases, inspired by the coefficient scanning order in JPEG
compression. This prioritizes relationships between tokens that are close in
the frequency domain.

It also extends the standard attention mechanism with two advanced, optional
normalization strategies as alternatives to the traditional softmax function:

1.  **Adaptive Temperature Softmax**: Dynamically adjusts the sharpness of the
    attention distribution based on its entropy, helping to prevent
    over-confidence and improve model calibration.
2.  **Hierarchical Routing**: A parameter-free, deterministic alternative that
    computes attention probabilities by routing mass through a fixed binary
    tree.

Conceptual Overview
-------------------

The layer's operational flow mirrors the standard WindowAttention but with a
specialized core:

1.  **Input**: A tensor of shape ``(B, N, C)``.
2.  **Grid Formation**: The sequence of length `N` is arranged into the smallest
    possible square grid (`H x W`), padding if necessary.
3.  **Window Partitioning**: The grid is padded to be divisible by
    `window_size` and then partitioned into non-overlapping windows.
4.  **Local Zigzag Attention**: Within each window, multi-head self-attention
    is computed by the `SingleWindowZigZagAttention` core, which uses a
    zigzag-ordered relative position bias and one of the configurable
    normalization methods.
5.  **Window Merging**: The processed windows are stitched back together.
6.  **Output**: The grid is un-padded and flattened back to a sequence of the
    original length `N`, resulting in a tensor of shape ``(B, N, C)``.

Key Features & Benefits:
~~~~~~~~~~~~~~~~~~~~~~~~

1.  **Frequency-Domain Locality**: The zigzag ordering of relative positions
    can be beneficial for image or signal processing tasks.
2.  **Advanced Normalization**: Offers state-of-the-art alternatives to
    softmax that can improve performance and calibration.
3.  **End-to-End Logic**: Handles all padding, grid formation, and windowing
    internally, accepting any 1D sequence as input.

Usage Example:
~~~~~~~~~~~~~~
.. code-block:: python

    import keras

    # Input sequence of 150 tokens. The layer will form a 13x13 grid and
    # partition it into 7x7 windows.
    x = keras.random.normal((2, 150, 96))

    # Standard usage with zigzag bias and softmax
    zigzag_attn = WindowZigZagAttention(dim=96, window_size=7, num_heads=4)
    output_softmax = zigzag_attn(x) # Shape: (2, 150, 96)

    # With hierarchical routing instead of softmax
    routing_attn = WindowZigZagAttention(
        dim=96, window_size=7, num_heads=4, use_hierarchical_routing=True
    )
    output_routing = routing_attn(x) # Shape: (2, 150, 96)

    # With adaptive temperature softmax for better calibration
    adaptive_attn = WindowZigZagAttention(
        dim=96, window_size=7, num_heads=4,
        use_adaptive_softmax=True,
        adaptive_softmax_config={"min_temp": 0.1, "max_temp": 2.0}
    )
    output_adaptive = adaptive_attn(x) # Shape: (2, 150, 96)

"""

import keras
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..activations.adaptive_softmax import AdaptiveTemperatureSoftmax
from ..activations.routing_probabilities import RoutingProbabilitiesLayer


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SingleWindowZigZagAttention(keras.layers.Layer):
    """Core multi-head self-attention for a single zigzag-ordered window.

    This layer implements the core logic for windowed multi-head self-attention
    using a zigzag-ordered relative position bias. It is designed to be used
    internally by the public `WindowZigZagAttention` layer.

    It can be configured to use standard softmax, adaptive temperature softmax,
    or hierarchical routing for attention normalization. It handles partial
    windows by internally padding inputs to the full window size.

    Args:
        dim: Integer, dimensionality of the input feature space.
        window_size: Integer, the height and width of the attention window.
        num_heads: Integer, number of attention heads.
        qkv_bias: Boolean, whether to use a bias term in the QKV projection.
        qk_scale: Optional float, override for query scaling factor.
        attn_dropout_rate: Float, dropout rate for attention probabilities.
        proj_dropout_rate: Float, dropout rate for the final output projection.
        proj_bias: Boolean, whether to use a bias term in the output projection.
        use_hierarchical_routing: Boolean, if True, uses hierarchical routing
            instead of softmax. Cannot be True if `use_adaptive_softmax` is True.
        use_adaptive_softmax: Boolean, if True, uses adaptive temperature softmax
            instead of standard softmax. Cannot be True if
            `use_hierarchical_routing` is True.
        adaptive_softmax_config: Optional dictionary of arguments for
            AdaptiveTemperatureSoftmax. Used only when `use_adaptive_softmax=True`.
        kernel_initializer: Initializer for kernel weights.
        bias_initializer: Initializer for bias vectors.
        kernel_regularizer: Optional regularizer for kernel weights.
        bias_regularizer: Optional regularizer for bias weights.
        **kwargs: Additional keyword arguments for the base Layer class.
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
        use_hierarchical_routing: bool = False,
        use_adaptive_softmax: bool = False,
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
    ) -> None:
        super().__init__(**kwargs)

        # --- Configuration Validation ---
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        if not (0.0 <= attn_dropout_rate <= 1.0):
            raise ValueError(
                f"attn_dropout_rate must be in [0, 1], got {attn_dropout_rate}"
            )
        if not (0.0 <= proj_dropout_rate <= 1.0):
            raise ValueError(
                f"proj_dropout_rate must be in [0, 1], got {proj_dropout_rate}"
            )
        # Enforce mutual exclusivity of custom normalization methods.
        if use_adaptive_softmax and use_hierarchical_routing:
            raise ValueError(
                "Only one of `use_adaptive_softmax` or "
                "`use_hierarchical_routing` can be True."
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
        self.use_hierarchical_routing = use_hierarchical_routing
        self.use_adaptive_softmax = use_adaptive_softmax
        self.adaptive_softmax_config = adaptive_softmax_config
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # --- Adaptive Softmax Configuration and Validation ---
        if self.use_adaptive_softmax:
            config = adaptive_softmax_config or {}
            min_temp = config.setdefault("min_temp", 0.1)
            max_temp = config.setdefault("max_temp", 1.0)
            config.setdefault("entropy_threshold", 0.5)
            if min_temp <= 0:
                raise ValueError(f"min_temp must be positive, got {min_temp}")
            if max_temp <= min_temp:
                raise ValueError(
                    f"max_temp ({max_temp}) must be > min_temp ({min_temp})"
                )
            self.adaptive_softmax_config = config
        else:
            self.adaptive_softmax_config = None

        # --- CREATE all sub-layers in __init__ ---
        dense_kwargs = {
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
        }
        self.qkv = keras.layers.Dense(
            self.dim * 3, use_bias=self.qkv_bias, name="qkv", **dense_kwargs
        )
        self.proj = keras.layers.Dense(
            self.dim, use_bias=self.proj_bias, name="proj", **dense_kwargs
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

        # Conditionally create normalization layers
        self.hierarchical_routing = (
            RoutingProbabilitiesLayer(axis=-1, name="routing_probs")
            if self.use_hierarchical_routing
            else None
        )
        self.adaptive_softmax = (
            AdaptiveTemperatureSoftmax(
                name="adaptive_softmax", **self.adaptive_softmax_config
            )
            if self.use_adaptive_softmax
            else None
        )

        # --- Pre-compute ZIGZAG relative position indices ---
        zigzag_coords = self._generate_zigzag_coords(self.window_size)
        coords = keras.ops.convert_to_tensor(zigzag_coords, dtype="int32")
        relative_coords = keras.ops.expand_dims(
            coords, 1
        ) - keras.ops.expand_dims(coords, 0)
        relative_coords += self.window_size - 1
        num_cols = 2 * self.window_size - 1
        self.relative_position_index = (
            relative_coords[:, :, 0] * num_cols + relative_coords[:, :, 1]
        )

    @staticmethod
    def _generate_zigzag_coords(size: int) -> List[Tuple[int, int]]:
        """Generates (row, col) coordinates for a zigzag scan."""
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
        """Creates the layer's weights and builds its sub-layers."""
        num_bias_entries = (2 * self.window_size - 1) ** 2
        self.relative_position_bias_table = self.add_weight(
            name="relative_position_bias_table",
            shape=(num_bias_entries, self.num_heads),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            dtype=self.dtype,
        )

        padded_shape = list(input_shape)
        padded_shape[1] = self.window_size * self.window_size
        padded_shape = tuple(padded_shape)

        self.qkv.build(padded_shape)
        self.proj.build(padded_shape)
        if self.attn_dropout is not None:
            self.attn_dropout.build(None)
        if self.proj_dropout is not None:
            self.proj_dropout.build(padded_shape)

        attn_scores_shape = (
            input_shape[0],
            self.num_heads,
            padded_shape[1],
            padded_shape[1],
        )
        if self.hierarchical_routing is not None:
            self.hierarchical_routing.build(attn_scores_shape)
        if self.adaptive_softmax is not None:
            self.adaptive_softmax.build(attn_scores_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Defines the forward pass of the attention layer."""
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
        attn_scores = keras.ops.matmul(q, keras.ops.transpose(k, (0, 1, 3, 2)))

        bias = keras.ops.take(
            self.relative_position_bias_table,
            self.relative_position_index,
            axis=0,
        )
        bias = keras.ops.transpose(bias, (2, 0, 1))
        attn_scores = attn_scores + keras.ops.expand_dims(bias, 0)

        if final_attention_mask is not None:
            broadcast_mask = keras.ops.reshape(
                final_attention_mask, (B, 1, 1, N)
            )
            inf_value = keras.ops.convert_to_tensor(
                -1e9, dtype=attn_scores.dtype
            )
            additive_mask = (
                1.0 - keras.ops.cast(broadcast_mask, dtype=attn_scores.dtype)
            ) * inf_value
            attn_scores = attn_scores + additive_mask

        # --- NORMALIZATION STEP ---
        if self.use_adaptive_softmax:
            attn_weights = self.adaptive_softmax(attn_scores, training=training)
        elif self.use_hierarchical_routing:
            attn_weights = self.hierarchical_routing(
                attn_scores, training=training
            )
        else:
            attn_weights = keras.ops.softmax(attn_scores, axis=-1)

        if self.attn_dropout is not None:
            attn_weights = self.attn_dropout(attn_weights, training=training)

        # --- OUTPUT COMPUTATION ---
        x = keras.ops.matmul(attn_weights, v)
        x = keras.ops.transpose(x, (0, 2, 1, 3))
        x = keras.ops.reshape(x, (B, N, C))
        x = self.proj(x, training=training)
        if self.proj_dropout is not None:
            x = self.proj_dropout(x, training=training)

        # --- UN-PADDING ---
        output = x[:, :N_actual, :]
        return output

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """The output shape is identical to the input shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer's configuration for serialization."""
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
                "use_hierarchical_routing": self.use_hierarchical_routing,
                "use_adaptive_softmax": self.use_adaptive_softmax,
                "adaptive_softmax_config": self.adaptive_softmax_config,
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
class WindowZigZagAttention(keras.layers.Layer):
    """Windowed Zigzag Attention Layer.

    This layer implements the complete windowed multi-head self-attention
    mechanism using a zigzag relative position bias. It takes a 1D sequence,
    reshapes it into a 2D grid, partitions the grid into windows, applies
    attention within each window via the `SingleWindowZigZagAttention` core,
    and merges the result back into a 1D sequence of the original length.

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
        SingleWindowZigZagAttention (Core zigzag attention logic)
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
            :class:`SingleWindowZigZagAttention` layer, such as ``qkv_bias``,
            ``use_hierarchical_routing``, etc.

    Input shape:
        A 3D tensor with shape ``(batch_size, sequence_length, dim)``.

    Output shape:
        A 3D tensor with the same shape as the input:
        ``(batch_size, sequence_length, dim)``.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int, **kwargs: Any
    ):
        super().__init__(name=kwargs.pop("name", "window_zigzag_attention"))
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        # Store the kwargs to pass to the inner layer and for serialization.
        # This will be replaced by a more robust get_config method.
        self.kwargs = kwargs

        # The core attention mechanism for a single window.
        self.attention = SingleWindowZigZagAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            **kwargs,
        )

    def _window_partition(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Partition a 4D grid tensor into windows."""
        B, H, W, C = keras.ops.shape(x)
        ws = self.window_size
        x = keras.ops.reshape(x, (B, H // ws, ws, W // ws, ws, C))
        x = keras.ops.transpose(x, (0, 1, 3, 2, 4, 5))
        windows = keras.ops.reshape(x, (-1, ws * ws, C))
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
        """Build the layer's weights."""
        self.attention.build(
            (None, self.window_size * self.window_size, self.dim)
        )
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass for the window attention layer."""
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
        """Compute the output shape of the layer."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Serialize the layer's configuration."""
        config = super().get_config()
        # Add the primary parameters managed by this layer.
        config.update(
            {
                "dim": self.dim,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
            }
        )
        # Get the configuration of the inner attention layer, which holds all
        # the detailed kwargs. This is the source of truth.
        inner_config = self.attention.get_config()

        # Remove keys that are already present in the outer layer's config to
        # avoid duplication.
        for key in [
            "name",
            "trainable",
            "dtype",
            "dim",
            "window_size",
            "num_heads",
        ]:
            inner_config.pop(key, None)

        # Update the main config with the remaining inner layer-specific kwargs.
        # This correctly captures any default values set by the inner layer,
        # such as in `adaptive_softmax_config`.
        config.update(inner_config)
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "WindowZigZagAttention":
        """Create a layer from its config."""
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