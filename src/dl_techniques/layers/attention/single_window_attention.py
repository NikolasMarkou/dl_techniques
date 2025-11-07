import keras
from typing import Any, Dict, Optional, Tuple, Union

from ..kan_linear import KANLinear
from ..activations.adaptive_softmax import AdaptiveTemperatureSoftmax
from ..activations.routing_probabilities import RoutingProbabilitiesLayer


@keras.saving.register_keras_serializable()
class SingleWindowAttention(keras.layers.Layer):
    """
    Unified multi-head self-attention for a single window.

    This class merges multiple attention mechanisms into a single, configurable
    layer. It can perform standard linear attention, attention with a non-linear
    KAN-based Key projection, and use different normalization schemes like
    standard softmax, adaptive softmax, or hierarchical routing.

    This layer is designed for internal use by a parent windowing layer.
    """

    def __init__(
            self,
            dim: int,
            window_size: int,
            num_heads: int,
            attention_mode: str = "linear",
            normalization: str = "softmax",
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
    ) -> None:
        super().__init__(**kwargs)

        # --- Validate parameters ---
        valid_modes = {"linear", "kan_key"}
        if attention_mode not in valid_modes:
            raise ValueError(
                f"Invalid attention_mode. Expected one of {valid_modes}, "
                f"got '{attention_mode}'"
            )
        valid_norms = {"softmax", "adaptive_softmax", "hierarchical_routing"}
        if normalization not in valid_norms:
            raise ValueError(
                f"Invalid normalization. Expected one of {valid_norms}, "
                f"got '{normalization}'"
            )

        # --- Store configuration ---
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (
            qk_scale if qk_scale is not None else self.head_dim ** -0.5
        )
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
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # --- Create sub-layers based on configuration ---
        if self.attention_mode == "linear":
            self.qkv = keras.layers.Dense(
                self.dim * 3,
                use_bias=self.qkv_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="qkv",
            )
        elif self.attention_mode == "kan_key":
            self.query = keras.layers.Dense(
                self.dim, use_bias=False, name="query"
            )
            self.key = KANLinear(
                features=self.dim,
                grid_size=self.kan_grid_size,
                spline_order=self.kan_spline_order,
                activation=self.kan_activation,
                name="key_kan",
            )
            self.value = keras.layers.Dense(
                self.dim, use_bias=False, name="value"
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

        if self.normalization == "adaptive_softmax":
            self.adaptive_softmax = AdaptiveTemperatureSoftmax(
                name="adaptive_softmax", **(self.adaptive_softmax_config or {})
            )
        elif self.normalization == "hierarchical_routing":
            self.hierarchical_routing = RoutingProbabilitiesLayer(
                axis=-1, name="routing_probs"
            )

        if self.use_relative_position_bias:
            coords_h = keras.ops.arange(self.window_size, dtype="int32")
            coords_w = keras.ops.arange(self.window_size, dtype="int32")
            coords = keras.ops.stack(
                keras.ops.meshgrid(coords_h, coords_w, indexing="ij")
            )
            coords_flatten = keras.ops.reshape(coords, (2, -1))
            relative_coords = keras.ops.expand_dims(
                coords_flatten, 2
            ) - keras.ops.expand_dims(coords_flatten, 1)
            relative_coords = keras.ops.transpose(
                relative_coords, (1, 2, 0)
            )
            relative_coords_h = (
                    relative_coords[:, :, 0] + self.window_size - 1
            )
            relative_coords_w = (
                    relative_coords[:, :, 1] + self.window_size - 1
            )
            relative_coords_h *= 2 * self.window_size - 1
            self.relative_position_index = (
                    relative_coords_h + relative_coords_w
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer's weights."""
        if self.use_relative_position_bias:
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

        if self.attention_mode == "linear":
            self.qkv.build(padded_shape)
        else:
            self.query.build(padded_shape)
            self.key.build(padded_shape)
            self.value.build(padded_shape)
        self.proj.build(padded_shape)

        if self.attn_dropout is not None:
            self.attn_dropout.build(None)

        # FIX: The normalization layers act on the attention scores. We must
        # build them with the correct attention score shape, not None.
        num_tokens_in_window = self.window_size * self.window_size
        attention_scores_shape = (
            input_shape[0],  # batch_size (can be None)
            self.num_heads,
            num_tokens_in_window,
            num_tokens_in_window
        )

        if self.normalization == "adaptive_softmax":
            self.adaptive_softmax.build(attention_scores_shape)
        elif self.normalization == "hierarchical_routing":
            self.hierarchical_routing.build(attention_scores_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass for the unified single window attention."""
        input_shape = keras.ops.shape(inputs)
        B_actual, N_actual = input_shape[0], input_shape[1]
        N_target = self.window_size * self.window_size

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
            padded_user_mask = keras.ops.pad(
                attention_mask, [[0, 0], [0, padding_amount]]
            )
            final_attention_mask = (
                    keras.ops.cast(padded_user_mask, "int32")
                    * internal_padding_mask
            )

        B, N, C = keras.ops.shape(padded_inputs)
        if self.attention_mode == "linear":
            qkv = self.qkv(padded_inputs, training=training)
            qkv = keras.ops.reshape(
                qkv, (B, N, 3, self.num_heads, self.head_dim)
            )
            qkv = keras.ops.transpose(qkv, (2, 0, 3, 1, 4))
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            q_proj = self.query(padded_inputs, training=training)
            k_proj = self.key(padded_inputs, training=training)
            v_proj = self.value(padded_inputs, training=training)
            q = keras.ops.transpose(
                keras.ops.reshape(q_proj, (B, N, self.num_heads, self.head_dim)),
                (0, 2, 1, 3),
            )
            k = keras.ops.transpose(
                keras.ops.reshape(k_proj, (B, N, self.num_heads, self.head_dim)),
                (0, 2, 1, 3),
            )
            v = keras.ops.transpose(
                keras.ops.reshape(v_proj, (B, N, self.num_heads, self.head_dim)),
                (0, 2, 1, 3),
            )

        q = q * self.scale
        attn = keras.ops.matmul(q, keras.ops.transpose(k, (0, 1, 3, 2)))

        if self.use_relative_position_bias:
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

        attn = keras.ops.clip(attn, -30.0, 30.0)

        if self.normalization == "adaptive_softmax":
            attn = self.adaptive_softmax(attn, training=training)
        elif self.normalization == "hierarchical_routing":
            attn = self.hierarchical_routing(attn, training=training)
        else:
            attn = keras.ops.softmax(attn, axis=-1)

        if self.attn_dropout is not None:
            attn = self.attn_dropout(attn, training=training)
        x = keras.ops.matmul(attn, v)
        x = keras.ops.transpose(x, (0, 2, 1, 3))
        x = keras.ops.reshape(x, (B, N, C))
        x = self.proj(x, training=training)

        output = x[:, :N_actual, :]
        return output

    def get_config(self) -> Dict[str, Any]:
        """Serialize the layer's configuration."""
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
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