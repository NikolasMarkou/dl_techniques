"""
Unified single-window multi-head self-attention.

This layer implements multi-head self-attention restricted to a single square
window of side ``window_size`` (i.e. ``window_size ** 2`` tokens). It merges
several attention variants into one configurable layer: standard linear QKV
projection or a non-linear KAN-based Key projection, combined with selectable
score normalization (standard softmax, adaptive-temperature softmax, or
hierarchical routing probabilities). Internal padding ensures every window
reaches ``window_size ** 2`` tokens before attention is computed, and the
padded positions are stripped from the output.

Architecturally, the layer follows the scaled dot-product attention formula
``Attention(Q, K, V) = norm(Q K^T / sqrt(d_k) + bias) V``, where ``norm`` is
the configurable normalization function and ``bias`` is an optional learnable
relative position bias indexed by intra-window 2D coordinates. The relative
position bias table follows the Swin Transformer convention.

The layer supports two projection modes:
-   **linear**: a single fused dense layer produces ``Q``, ``K``, ``V``.
-   **kan_key**: separate dense layers produce ``Q`` and ``V``, while ``K``
    is produced by a KAN linear layer to inject a non-linear key projection.

And three normalization modes:
-   **softmax**: standard softmax over the last axis.
-   **adaptive_softmax**: temperature-adaptive softmax for sharper attention.
-   **hierarchical_routing**: routing probabilities for soft hierarchical mixing.

References:
    - Liu et al., 2021. Swin Transformer: Hierarchical Vision Transformer
      using Shifted Windows. (https://arxiv.org/abs/2103.14030)
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)

"""

import keras
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..ffn.kan_linear import KANLinear
from ..activations.adaptive_softmax import AdaptiveTemperatureSoftmax
from ..activations.routing_probabilities import RoutingProbabilitiesLayer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SingleWindowAttention(keras.layers.Layer):
    """
    Unified multi-head self-attention for a single window.

    Merges multiple attention mechanisms into a single configurable layer
    supporting standard linear QKV projection or non-linear KAN-based Key
    projection, with selectable normalization: standard softmax, adaptive
    temperature softmax, or hierarchical routing probabilities. Internal
    padding ensures every window reaches ``window_size ** 2`` tokens before
    attention is computed, then strips padding from the output.

    The scaled dot-product attention is computed as
    ``Attention(Q, K, V) = norm(Q K^T / sqrt(d_k) + bias) V``, where ``norm``
    is one of the configurable normalization functions and ``bias`` is an
    optional learnable relative position bias table indexed by intra-window
    2D coordinates (Swin convention).

    **Architecture Overview:**

    .. code-block:: text

        ┌───────────────────────────────────────────────────────┐
        │              SingleWindowAttention                    │
        │                                                       │
        │   Input [B, N_actual, dim]                            │
        │          │                                            │
        │          ▼                                            │
        │   ┌─────────────────────────────────────────────┐     │
        │   │  Pad to window_size^2 tokens                │     │
        │   │  + build internal padding mask              │     │
        │   └─────────────────────────────────────────────┘     │
        │          │                                            │
        │          ▼                                            │
        │   ┌─────────────────────────────────────────────┐     │
        │   │  QKV Projection                             │     │
        │   │    linear : fused Dense(3*dim)              │     │
        │   │    kan_key: Dense(Q) + KAN(K) + Dense(V)    │     │
        │   │                                             │     │
        │   │  Reshape ──► (B, heads, N, d_h) for Q,K,V   │     │
        │   │                  │                          │     │
        │   │                  ▼                          │     │
        │   │        scores = Q @ K^T / sqrt(d_k)         │     │
        │   │                  │                          │     │
        │   │                  ▼                          │     │
        │   │      [+ relative_position_bias]             │     │
        │   │      [+ additive padding/user mask]         │     │
        │   │      [clip(scores, -30, 30)]                │     │
        │   │                  │                          │     │
        │   │                  ▼                          │     │
        │   │     norm ∈ {softmax,                        │     │
        │   │             adaptive_softmax,               │     │
        │   │             hierarchical_routing}           │     │
        │   │                  │                          │     │
        │   │                  ▼                          │     │
        │   │        dropout ──► weights @ V              │     │
        │   │                  │                          │     │
        │   │                  ▼                          │     │
        │   │           Output Projection                 │     │
        │   └─────────────────────────────────────────────┘     │
        │          │                                            │
        │          ▼                                            │
        │   ┌─────────────────────────────────────────────┐     │
        │   │  Unpad ──► slice [:, :N_actual, :]          │     │
        │   └─────────────────────────────────────────────┘     │
        │          │                                            │
        │          ▼                                            │
        │   Output [B, N_actual, dim]                           │
        └───────────────────────────────────────────────────────┘

    :param dim: Total model dimension (split across heads). Must be positive
        and divisible by ``num_heads``.
    :type dim: int
    :param window_size: Height/width of the square attention window. The
        layer pads inputs up to ``window_size ** 2`` tokens.
    :type window_size: int
    :param num_heads: Number of attention heads. Must divide ``dim``.
    :type num_heads: int
    :param attention_mode: Projection mode. ``'linear'`` for standard
        dense QKV or ``'kan_key'`` for a KAN-based Key projection.
        Defaults to ``'linear'``.
    :type attention_mode: str
    :param normalization: Score normalization. One of ``'softmax'``,
        ``'adaptive_softmax'``, or ``'hierarchical_routing'``.
        Defaults to ``'softmax'``.
    :type normalization: str
    :param use_relative_position_bias: Whether to add a learnable relative
        position bias to attention scores. Defaults to ``True``.
    :type use_relative_position_bias: bool
    :param qkv_bias: Whether the fused QKV dense uses bias (linear mode).
        Defaults to ``True``.
    :type qkv_bias: bool
    :param qk_scale: Override for the QK scaling factor. If ``None``,
        defaults to ``head_dim ** -0.5``.
    :type qk_scale: Optional[float]
    :param dropout_rate: Dropout rate applied to attention weights. Must be
        between 0.0 and 1.0. Defaults to 0.0.
    :type dropout_rate: float
    :param proj_bias: Whether the output projection uses bias.
        Defaults to ``True``.
    :type proj_bias: bool
    :param kan_grid_size: Grid size for the KAN layer (``kan_key`` mode).
        Defaults to 5.
    :type kan_grid_size: int
    :param kan_spline_order: Spline order for the KAN layer.
        Defaults to 3.
    :type kan_spline_order: int
    :param kan_activation: Activation for the KAN layer.
        Defaults to ``'swish'``.
    :type kan_activation: str
    :param adaptive_softmax_config: Config dict forwarded to
        ``AdaptiveTemperatureSoftmax`` when that normalization is selected.
    :type adaptive_softmax_config: Optional[Dict[str, Any]]
    :param kernel_initializer: Initializer for kernel weights.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for bias weights.
        Defaults to ``'zeros'``.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param kwargs: Additional keyword arguments forwarded to the base Layer.

    :raises ValueError: If ``attention_mode`` is not one of
        ``{'linear', 'kan_key'}`` or ``normalization`` is not one of
        ``{'softmax', 'adaptive_softmax', 'hierarchical_routing'}``.
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

        # Validate inputs
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

        # Store ALL configuration parameters
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

        # CREATE sub-layers based on configuration
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

        # Precompute the static relative-position index table (Swin convention).
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
        """
        Build the layer weights and sub-layers.

        Allocates the learnable relative position bias table and explicitly
        builds the QKV / KAN / projection sub-layers against the *padded*
        per-window shape ``(B, window_size ** 2, dim)``. Normalization
        sub-layers are built with the full attention-score shape so they
        capture the correct last-axis dimensionality for serialization.

        :param input_shape: Shape tuple of the input tensor, expected as
            ``(batch_size, seq_len, dim)``. ``seq_len`` may be less than
            ``window_size ** 2`` since the layer pads internally.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.use_relative_position_bias:
            num_relative_positions = (2 * self.window_size - 1) ** 2
            self.relative_position_bias_table = self.add_weight(
                name="relative_position_bias_table",
                shape=(num_relative_positions, self.num_heads),
                initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                trainable=True,
                dtype=self.dtype,
            )

        # Sub-layers see the padded per-window shape.
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

        # Normalization layers act on attention scores: build with the
        # correct (B, heads, N, N) shape rather than None.
        num_tokens_in_window = self.window_size * self.window_size
        attention_scores_shape = (
            input_shape[0],
            self.num_heads,
            num_tokens_in_window,
            num_tokens_in_window,
        )

        if self.normalization == "adaptive_softmax":
            self.adaptive_softmax.build(attention_scores_shape)
        elif self.normalization == "hierarchical_routing":
            self.hierarchical_routing.build(attention_scores_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """
        Forward pass for the unified single-window attention.

        Pads the input up to ``window_size ** 2`` tokens, runs multi-head
        self-attention with optional relative position bias and configurable
        normalization, then slices the padding off the output. The internal
        padding mask is combined (multiplicatively) with any user-supplied
        ``attention_mask`` before being converted to an additive ``-1e9``
        bias on the attention scores.

        :param inputs: Token embeddings of shape ``(B, N_actual, dim)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional mask of shape ``(B, N_actual)`` with
            1 for valid tokens and 0 for padding. Combined with the internal
            padding mask.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Boolean indicating whether in training mode.
        :type training: Optional[bool]
        :return: Attended output of shape ``(B, N_actual, dim)``.
        :rtype: keras.KerasTensor
        """
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

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...],
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape, identical to the input shape.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple, identical to input shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization, includes all constructor parameters.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
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

# ---------------------------------------------------------------------
