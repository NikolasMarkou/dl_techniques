"""
A windowed multi-head attention on a zigzag-reordered sequence.

This layer provides a specialized form of self-attention that first
reorders a sequence based on a two-dimensional zigzag scan before applying
local, windowed attention. The primary motivation is to induce a locality
bias that is sensitive to frequency-domain proximity, a concept borrowed
from classical image compression algorithms like JPEG.
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
    """Core multi-head self-attention for a single window. (Internal use)"""
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
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"attn_dropout_rate must be in [0, 1], got {dropout_rate}")
        if use_adaptive_softmax and use_hierarchical_routing:
            raise ValueError("Only one of `use_adaptive_softmax` or `use_hierarchical_routing` can be True.")
        self.dim = dim
        self.window_size = window_size
        self.num_tokens_in_window = self.window_size * self.window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (qk_scale if qk_scale is not None else self.head_dim**-0.5)
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
        self.qkv = keras.layers.Dense(self.dim * 3, use_bias=self.qkv_bias, name="qkv", **dense_kwargs)
        self.proj = keras.layers.Dense(self.dim, use_bias=self.proj_bias, name="proj", **dense_kwargs)
        self.attn_dropout = (keras.layers.Dropout(self.dropout_rate, name="attn_dropout") if self.dropout_rate > 0.0 else None)
        self.hierarchical_routing = (RoutingProbabilitiesLayer(axis=-1, name="routing_probs") if self.use_hierarchical_routing else None)
        self.adaptive_softmax = (AdaptiveTemperatureSoftmax(name="adaptive_softmax", **(self.adaptive_softmax_config or {})) if self.use_adaptive_softmax else None)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        padded_shape = list(input_shape)
        padded_shape[1] = self.num_tokens_in_window
        padded_shape = tuple(padded_shape)
        self.qkv.build(padded_shape)
        self.proj.build(padded_shape)
        if self.attn_dropout is not None:
            self.attn_dropout.build(None)
        attn_scores_shape = (input_shape[0], self.num_heads, padded_shape[1], padded_shape[1])
        if self.hierarchical_routing is not None:
            self.hierarchical_routing.build(attn_scores_shape)
        if self.adaptive_softmax is not None:
            self.adaptive_softmax.build(attn_scores_shape)
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, attention_mask: Optional[keras.KerasTensor] = None, training: Optional[bool] = None) -> keras.KerasTensor:
        input_shape = keras.ops.shape(inputs)
        B_actual, N_actual, C_actual = (input_shape[0], input_shape[1], input_shape[2])
        padding_amount = self.num_tokens_in_window - N_actual
        padded_inputs = keras.ops.pad(inputs, [[0, 0], [0, padding_amount], [0, 0]])
        internal_padding_mask = keras.ops.concatenate([keras.ops.ones((B_actual, N_actual), dtype="int32"), keras.ops.zeros((B_actual, padding_amount), dtype="int32")], axis=1)
        final_attention_mask = internal_padding_mask
        if attention_mask is not None:
            mask_padding = keras.ops.zeros((B_actual, padding_amount), dtype=attention_mask.dtype)
            padded_user_mask = keras.ops.concatenate([attention_mask, mask_padding], axis=1)
            final_attention_mask = (keras.ops.cast(padded_user_mask, "int32") * internal_padding_mask)
        B, N, C = keras.ops.shape(padded_inputs)
        qkv = self.qkv(padded_inputs, training=training)
        qkv = keras.ops.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim))
        qkv = keras.ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn_scores = keras.ops.matmul(q, keras.ops.transpose(k, (0, 1, 3, 2)))
        if final_attention_mask is not None:
            broadcast_mask = keras.ops.reshape(final_attention_mask, (B, 1, 1, N))
            inf_value = keras.ops.convert_to_tensor(-1e9, dtype=attn_scores.dtype)
            additive_mask = (1.0 - keras.ops.cast(broadcast_mask, dtype=attn_scores.dtype)) * inf_value
            attn_scores = attn_scores + additive_mask
        if self.use_adaptive_softmax:
            attn_weights = self.adaptive_softmax(attn_scores, training=training)
        elif self.use_hierarchical_routing:
            attn_weights = self.hierarchical_routing(attn_scores, training=training)
        else:
            attn_weights = keras.ops.softmax(attn_scores, axis=-1)
        if self.attn_dropout is not None:
            attn_weights = self.attn_dropout(attn_weights, training=training)
        x = keras.ops.matmul(attn_weights, v)
        x = keras.ops.transpose(x, (0, 2, 1, 3))
        x = keras.ops.reshape(x, (B, N, C))
        x = self.proj(x, training=training)
        return x[:, :N_actual, :]

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "dim": self.dim, "window_size": self.window_size, "num_heads": self.num_heads,
            "qkv_bias": self.qkv_bias, "qk_scale": self.qk_scale, "dropout_rate": self.dropout_rate,
            "proj_bias": self.proj_bias, "use_hierarchical_routing": self.use_hierarchical_routing,
            "use_adaptive_softmax": self.use_adaptive_softmax, "adaptive_softmax_config": self.adaptive_softmax_config,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

@keras.saving.register_keras_serializable()
class WindowZigZagAttention(keras.layers.Layer):
    """Windowed Zigzag Attention Layer."""

    def __init__(self, dim: int, window_size: int, num_heads: int, **kwargs: Any):
        super().__init__(name=kwargs.pop("name", "window_zigzag_attention"))
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.num_tokens_in_window = window_size * window_size
        self.kwargs = kwargs
        self.attention = _CoreAttention(dim=dim, window_size=window_size, num_heads=num_heads, **kwargs)
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
        r_coords = keras.ops.arange(0, H, dtype="int32")
        c_coords = keras.ops.arange(0, W, dtype="int32")
        r_grid, c_grid = keras.ops.meshgrid(r_coords, c_coords, indexing="ij")
        r_flat = keras.ops.reshape(r_grid, (-1,))
        c_flat = keras.ops.reshape(c_grid, (-1,))
        s = r_flat + c_flat
        secondary_key = keras.ops.where(s % 2 == 1, r_flat, H - 1 - r_flat)
        combined_key = s * H + secondary_key
        zigzag_indices = keras.ops.argsort(combined_key)
        return zigzag_indices

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        N_actual = input_shape[1]
        if N_actual is None:
            raise ValueError(
                "WindowZigZagAttention requires a fixed sequence length for XLA "
                "compatibility. Received `None` for the sequence dimension."
            )
        self.H = int(math.ceil(math.sqrt(N_actual)))
        self.W = self.H
        self.N_grid = self.H * self.W
        self.pad_len_seq = self.N_grid - N_actual
        H_tensor = keras.ops.convert_to_tensor(self.H, dtype="int32")
        W_tensor = keras.ops.convert_to_tensor(self.W, dtype="int32")
        self.zigzag_indices = self._generate_zigzag_indices(H_tensor, W_tensor)
        self.inverse_zigzag_indices = keras.ops.argsort(self.zigzag_indices)
        self.attention.build((None, self.num_tokens_in_window, self.dim))
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, attention_mask: Optional[keras.KerasTensor] = None, training: Optional[bool] = None) -> keras.KerasTensor:
        input_shape = keras.ops.shape(inputs)
        B, N_actual, C = input_shape[0], input_shape[1], input_shape[2]
        win_len = self.num_tokens_in_window

        padded_inputs = keras.ops.pad(inputs, [[0, 0], [0, self.pad_len_seq], [0, 0]])

        # Use keras.ops.take for direct, XLA-compatible gathering
        zigzag_sequence = keras.ops.take(padded_inputs, self.zigzag_indices, axis=1)

        zigzag_mask = None
        if attention_mask is not None:
            padded_mask = keras.ops.pad(attention_mask, [[0, 0], [0, self.pad_len_seq]])
            zigzag_mask = keras.ops.take(padded_mask, self.zigzag_indices, axis=1)

        pad_len_win = (win_len - (self.N_grid % win_len)) % win_len
        padded_zigzag_seq = keras.ops.pad(zigzag_sequence, [[0, 0], [0, pad_len_win], [0, 0]])
        num_windows = (self.N_grid + pad_len_win) // win_len
        windows = keras.ops.reshape(padded_zigzag_seq, (B * num_windows, win_len, C))

        attn_mask_for_windows = None
        if zigzag_mask is not None:
            padded_zigzag_mask = keras.ops.pad(zigzag_mask, [[0, 0], [0, pad_len_win]], constant_values=0)
            attn_mask_for_windows = keras.ops.reshape(padded_zigzag_mask, (B * num_windows, win_len))

        attn_windows = self.attention(windows, attention_mask=attn_mask_for_windows, training=training)

        merged_zigzag_seq = keras.ops.reshape(attn_windows, (B, num_windows * win_len, C))
        unpadded_zigzag_seq = merged_zigzag_seq[:, :self.N_grid, :]

        # Use keras.ops.take for the inverse operation
        sequence_unpadded = keras.ops.take(unpadded_zigzag_seq, self.inverse_zigzag_indices, axis=1)

        output = sequence_unpadded[:, :N_actual, :]
        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "dim": self.dim, "window_size": self.window_size, "num_heads": self.num_heads,
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
        return cls(dim=dim, window_size=window_size, num_heads=num_heads, **config_copy)