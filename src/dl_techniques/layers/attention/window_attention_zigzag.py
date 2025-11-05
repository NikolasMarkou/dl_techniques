"""
A windowed multi-head attention on a zigzag-reordered sequence.

This layer provides a specialized form of self-attention that first
reorders a sequence based on a two-dimensional zigzag scan before applying
local, windowed attention. The primary motivation is to induce a locality
bias that is sensitive to frequency-domain proximity, a concept borrowed
from classical image compression algorithms like JPEG.

Architectural Overview:
    The layer operates through a series of transformations designed to
    restructure the input sequence for its specialized attention mechanism:
    1.  Grid Reshaping: The input 1D sequence of length `N` is first
        conceptually arranged into the smallest possible square 2D grid
        (H x W), with zero-padding applied if `N` is not a perfect square.
    2.  Zigzag Reordering: The tokens in this 2D grid are then physically
        reordered into a new 1D sequence by following a zigzag scan path.
        This scan traverses the grid along its anti-diagonals, grouping
        tokens that are proximate in the 2D frequency domain.
    3.  Window Partitioning: This new zigzag-ordered 1D sequence is
        partitioned into non-overlapping, fixed-size windows.
    4.  Local Attention: Standard multi-head self-attention is computed
        independently within each 1D window. This step is computationally
        efficient, scaling linearly with the sequence length.
    5.  Inverse Reordering: The outputs from the attention windows are
        merged and then reordered back into their original 2D grid
        positions using an inverse zigzag transformation.
    6.  Sequence Flattening: Finally, the 2D grid is flattened back into a
        1D sequence, and the initial padding is removed to match the
        original input length.

Foundational Mathematics and Intuition:
    The core of this layer lies in its fusion of windowed attention with
    zigzag scanning.

    -   Zigzag Scan: This technique linearizes a 2D data grid by tracing a
        path that prioritizes elements based on the sum of their
        coordinates (i.e., `row + col`). In signal processing, particularly
        with transformations like the Discrete Cosine Transform (DCT), this
        ordering groups low-frequency coefficients (top-left of the grid)
        before high-frequency coefficients (bottom-right). By applying
        attention to windows of this reordered sequence, the model can
        focus on relationships between components of similar frequency bands.

    -   Windowed Attention: To maintain computational tractability, attention
        is not computed across the entire sequence. Instead, following the
        paradigm of models like the Swin Transformer, the attention
        mechanism is constrained to local windows. This reduces the
        complexity from O(N^2) to O(N * k^2), where `k` is the window size.

    -   Advanced Normalization (Optional): The layer can replace the
        standard softmax function for attention weight computation with two
        alternatives:
        1.  Adaptive Temperature Softmax: This method addresses potential
            model over-confidence by dynamically scaling the logits before
            the softmax operation. The temperature `τ` is adjusted based on
            the entropy of the attention distribution, leading to better
            calibrated models. A higher temperature (softer distribution)
            is used for uncertain, high-entropy scores, while a lower
            temperature (sharper distribution) is used for confident,
            low-entropy scores.
        2.  Hierarchical Routing: As a parameter-free alternative to softmax,
            this deterministic method computes attention probabilities by
            routing a total probability mass of 1.0 through a fixed binary
            tree. At each node, incoming mass is split between its two
            children based on their relative activation scores. The final
            attention weights are the accumulated mass at the leaf nodes,
            providing a sparse and computationally distinct alternative to
            the standard exponential function.

References:
    -   Windowed Attention: Liu et al., "Swin Transformer: Hierarchical
        Vision Transformer using Shifted Windows" (2021).
        https://arxiv.org/abs/2103.14030
    -   Zigzag Scan: A foundational concept in the JPEG image compression
        standard.
    -   Adaptive Temperature Softmax: Ding et al., "Adaptive Temperature
        Scaling for Better Calibrated and More Accurate Models" (2021).
        https://arxiv.org/abs/2102.10599
    -   Hierarchical Routing: Hassani, "NOMAD: N-th Order-of-Magnitude
        -Attention" (2022). https://arxiv.org/abs/2205.13233
"""

import math
import keras
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..activations.adaptive_softmax import AdaptiveTemperatureSoftmax
from ..activations.routing_probabilities import RoutingProbabilitiesLayer


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class _CoreAttention(keras.layers.Layer):
    """Core multi-head self-attention for a single window. (Internal use)

    This layer implements the core logic for multi-head self-attention. It is
    designed to be used internally by the public `WindowZigZagAttention` layer
    to process a single window of tokens.

    It can be configured to use standard softmax, adaptive temperature softmax,
    or hierarchical routing for attention normalization. It handles partial
    windows by internally padding inputs to the full window size.

    Args:
        dim: Integer, dimensionality of the input feature space.
        window_size: Integer, the height and width of the attention window.
            The number of tokens in a window is `window_size * window_size`.
        num_heads: Integer, number of attention heads.
        qkv_bias: Boolean, whether to use a bias term in the QKV projection.
        qk_scale: Optional float, override for query scaling factor.
        dropout_rate: Float, dropout rate for attention probabilities.
        proj_dropout_rate: Float, dropout rate for the final output projection.
        proj_bias: Boolean, whether to use a bias term in the output projection.
        use_hierarchical_routing: Boolean, if True, uses hierarchical routing
            instead of softmax.
        use_adaptive_softmax: Boolean, if True, uses adaptive temperature softmax
            instead of standard softmax.
        adaptive_softmax_config: Optional dictionary of arguments for
            AdaptiveTemperatureSoftmax.
        **kwargs: Additional keyword arguments for the base Layer class.
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
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(
                f"attn_dropout_rate must be in [0, 1], got {dropout_rate}"
            )
        if use_adaptive_softmax and use_hierarchical_routing:
            raise ValueError(
                "Only one of `use_adaptive_softmax` or "
                "`use_hierarchical_routing` can be True."
            )

        # --- Store configuration parameters ---
        self.dim = dim
        self.window_size = window_size
        self.num_tokens_in_window = self.window_size * self.window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (
            qk_scale if qk_scale is not None else self.head_dim**-0.5
        )
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.proj_bias = proj_bias
        self.dropout_rate = dropout_rate
        self.use_hierarchical_routing = use_hierarchical_routing
        self.use_adaptive_softmax = use_adaptive_softmax
        self.adaptive_softmax_config = adaptive_softmax_config
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        if self.use_adaptive_softmax:
            config = adaptive_softmax_config or {}
            min_temp = config.setdefault("min_temp", 0.1)
            max_temp = config.setdefault("max_temp", 1.0)
            if min_temp <= 0 or max_temp <= min_temp:
                raise ValueError("Invalid adaptive softmax temperature range.")
            self.adaptive_softmax_config = config
        else:
            self.adaptive_softmax_config = None

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
            keras.layers.Dropout(self.dropout_rate, name="attn_dropout")
            if self.dropout_rate > 0.0
            else None
        )
        self.hierarchical_routing = (
            RoutingProbabilitiesLayer(axis=-1, name="routing_probs")
            if self.use_hierarchical_routing
            else None
        )
        self.adaptive_softmax = (
            AdaptiveTemperatureSoftmax(
                name="adaptive_softmax", **(self.adaptive_softmax_config or {})
            )
            if self.use_adaptive_softmax
            else None
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Builds the sub-layers."""
        padded_shape = list(input_shape)
        padded_shape[1] = self.num_tokens_in_window
        padded_shape = tuple(padded_shape)

        self.qkv.build(padded_shape)
        self.proj.build(padded_shape)
        if self.attn_dropout is not None:
            self.attn_dropout.build(None)

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

        # --- PADDING & MASKING LOGIC for partial windows ---
        padding_amount = self.num_tokens_in_window - N_actual
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

        # --- UN-PADDING ---
        return x[:, :N_actual, :]

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "dim": self.dim, "window_size": self.window_size,
            "num_heads": self.num_heads, "qkv_bias": self.qkv_bias,
            "qk_scale": self.qk_scale,
            "dropout_rate": self.dropout_rate,
            "proj_bias": self.proj_bias,
            "use_hierarchical_routing": self.use_hierarchical_routing,
            "use_adaptive_softmax": self.use_adaptive_softmax,
            "adaptive_softmax_config": self.adaptive_softmax_config,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class WindowZigZagAttention(keras.layers.Layer):
    """Windowed Zigzag Attention Layer.

    This layer implements windowed multi-head self-attention on a sequence
    that has been physically reordered according to a zigzag scan.

    Architecture:
    ~~~~~~~~~~~~~
    .. code-block::

        Input(B, N, C)
               ↓
        Pad sequence to form a square grid -> (B, N_grid, C)
               ↓
        Reshape to grid -> (B, H, W, C)
               ↓
        Flatten grid with zigzag scan -> (B, N_grid, C) [zigzag order]
               ↓
        Pad sequence for windowing -> (B, N_padded, C)
               ↓
        Partition into 1D windows -> (B*num_windows, win_len, C)
               ↓
        _CoreAttention (Standard MHSA + advanced normalization)
               ↓
        Merge windows -> (B, N_padded, C) [zigzag order]
               ↓
        Un-pad sequence -> (B, N_grid, C) [zigzag order]
               ↓
        Apply inverse zigzag reordering -> (B, N_grid, C) [original grid order]
               ↓
        Un-pad sequence to original length -> (B, N, C)
               ↓
        Output(B, N, C)

    Args:
        dim: Dimension of the input tokens (channels).
        window_size: The height and width of the conceptual attention window.
            The number of tokens per window is `window_size * window_size`.
        num_heads: Number of attention heads.
        **kwargs: Keyword arguments for the internal :class:`_CoreAttention`
            layer, such as `qkv_bias`, `use_hierarchical_routing`, etc.

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
        self.num_tokens_in_window = window_size * window_size

        self.kwargs = kwargs
        self.attention = _CoreAttention(
            dim=dim, window_size=window_size, num_heads=num_heads, **kwargs
        )
        # Attributes to be computed in build() for XLA compatibility
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
        """Generates flat indices for a zigzag scan of an HxW grid using keras ops."""
        # Create coordinate grids
        r_coords = keras.ops.arange(0, H, dtype="int32")
        c_coords = keras.ops.arange(0, W, dtype="int32")
        r_grid, c_grid = keras.ops.meshgrid(r_coords, c_coords, indexing="ij")

        # Flatten to 1D coordinate vectors
        r_flat = keras.ops.reshape(r_grid, (-1,))
        c_flat = keras.ops.reshape(c_grid, (-1,))

        # Primary sorting key: anti-diagonal index (r + c)
        s = r_flat + c_flat

        # Secondary sorting key: depends on the anti-diagonal's parity
        # On odd anti-diagonals (s % 2 == 1), r increases (ascending order)
        # On even anti-diagonals (s % 2 == 0), r decreases (descending order)
        # We simulate descending order by sorting on (H - 1 - r)
        secondary_key = keras.ops.where(
            s % 2 == 1, r_flat, H - 1 - r_flat
        )

        # Combine keys for a stable lexicographical sort simulation.
        # The primary key is scaled by a value larger than the max secondary key.
        combined_key = s * H + secondary_key

        # The sorted indices of the combined key give the zigzag order
        zigzag_indices = keras.ops.argsort(combined_key)
        return zigzag_indices

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        N_actual = input_shape[1]
        if N_actual is None:
            raise ValueError(
                "WindowZigZagAttention requires a fixed sequence length for "
                "XLA compatibility when generating zigzag indices. Received "
                "`None` for the sequence dimension in input_shape."
            )

        # Calculate grid dimensions and padding as static values
        self.H = self.W = int(math.ceil(math.sqrt(N_actual)))
        self.N_grid = self.H * self.W
        self.pad_len_seq = self.N_grid - N_actual

        # Generate indices as constant tensors.
        H_tensor = keras.ops.convert_to_tensor(self.H, dtype="int32")
        W_tensor = keras.ops.convert_to_tensor(self.W, dtype="int32")
        self.zigzag_indices = self._generate_zigzag_indices(H_tensor, W_tensor)
        self.inverse_zigzag_indices = keras.ops.argsort(self.zigzag_indices)

        # Build the internal attention layer
        self.attention.build(
            (None, self.num_tokens_in_window, self.dim)
        )
        super().build(input_shape)


    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        input_shape = keras.ops.shape(inputs)
        B, N_actual, C = input_shape[0], input_shape[1], input_shape[2]
        win_len = self.num_tokens_in_window

        # --- 1. Pad sequence and form a square grid ---
        padded_inputs = keras.ops.pad(
            inputs, [[0, 0], [0, self.pad_len_seq], [0, 0]]
        )

        # --- 2. Generate zigzag indices and reorder sequence ---
        expanded_indices = keras.ops.expand_dims(
            keras.ops.expand_dims(self.zigzag_indices, 0), 2
        )
        broadcasted_indices = keras.ops.repeat(
            expanded_indices, repeats=B, axis=0
        )
        zigzag_sequence = keras.ops.take_along_axis(
            padded_inputs, broadcasted_indices, axis=1
        )

        # --- 3. Handle attention mask through reordering ---
        zigzag_mask = None
        if attention_mask is not None:
            padded_mask = keras.ops.pad(
                attention_mask, [[0, 0], [0, self.pad_len_seq]]
            )
            broadcastable_mask_indices = keras.ops.expand_dims(
                self.zigzag_indices, 0
            )
            zigzag_mask = keras.ops.take_along_axis(
                padded_mask, broadcastable_mask_indices, axis=1
            )

        # --- 4. Pad zigzag sequence and partition into 1D windows ---
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

        # --- 5. Apply attention within windows ---
        attn_windows = self.attention(
            windows, attention_mask=attn_mask_for_windows, training=training
        )

        # --- 6. Merge windows and un-pad back to N_grid ---
        merged_zigzag_seq = keras.ops.reshape(
            attn_windows, (B, num_windows * win_len, C)
        )
        unpadded_zigzag_seq = merged_zigzag_seq[:, :self.N_grid, :]

        # --- 7. Apply inverse zigzag reordering ---
        expanded_inverse_indices = keras.ops.expand_dims(
            keras.ops.expand_dims(self.inverse_zigzag_indices, 0), 2
        )
        broadcasted_inverse_indices = keras.ops.repeat(
            expanded_inverse_indices, repeats=B, axis=0
        )
        sequence_unpadded = keras.ops.take_along_axis(
            unpadded_zigzag_seq, broadcasted_inverse_indices, axis=1
        )

        # --- 8. Un-pad sequence to original length ---
        output = sequence_unpadded[:, :N_actual, :]
        return output

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "window_size": self.window_size,
            "num_heads": self.num_heads,
        })
        inner_config = self.attention.get_config()
        for key in ["name", "trainable", "dtype", "dim", "window_size", "num_heads"]:
            inner_config.pop(key, None)
        config.update(inner_config)
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "WindowZigZagAttention":
        config_copy = config.copy()
        dim = config_copy.pop("dim")
        window_size = config_copy.pop("window_size")
        num_heads = config_copy.pop("num_heads")
        return cls(
            dim=dim, window_size=window_size, num_heads=num_heads, **config_copy
        )

# ---------------------------------------------------------------------