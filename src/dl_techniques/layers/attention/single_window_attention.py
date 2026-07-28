"""
Unified single-window multi-head self-attention.

This layer implements multi-head self-attention restricted to a single square
window of side ``window_size`` (i.e. ``window_size ** 2`` tokens). It merges
several attention variants into one configurable layer: standard linear QKV
projection or a non-linear KAN-based Key projection, combined with a unified
probability output strategy applied to the attention scores. Internal padding
ensures every window reaches ``window_size ** 2`` tokens before attention is
computed, and the padded positions are stripped from the output.

Architecture:
    The layer is one Swin-style window's worth of attention, made configurable
    along two independent axes (how ``K`` is projected, and how scores become
    probabilities) instead of being forked into separate classes.

    1.  **Projection.** Two mutually exclusive modes:

        -   **linear**: a single fused dense layer produces ``Q``, ``K``, ``V``.
        -   **kan_key**: separate dense layers produce ``Q`` and ``V``, while ``K``
            is produced by a KAN linear layer to inject a non-linear key
            projection.

    2.  **Padding to a full window.** Internal padding ensures every window
        reaches ``window_size ** 2`` tokens before attention is computed, and the
        padded positions are stripped from the output. Callers therefore never
        have to pre-pad, and a partial trailing window is handled without a
        separate code path.

    3.  **Scoring.** Optional QK-normalization (``qk_norm_type``) applies a
        normalization layer to ``Q`` and ``K`` before the score matmul,
        stabilising attention logits. An optional learnable relative position
        bias, indexed by intra-window 2D coordinates and following the Swin
        Transformer convention, is added to the scores.

    4.  **Probability output.** Score-to-probability conversion is delegated to
        :class:`ProbabilityOutput` via ``probability_type`` /
        ``probability_config``. Score-level routing strategies (``routing``,
        ``deterministic_routing``, ``hierarchical``, ``hierarchical_routing``) are
        rejected at construction time, as they are not appropriate normalizations
        for raw attention scores in this layer.

Foundational Mathematics:
    The layer follows the scaled dot-product attention formula::

        Attention(Q, K, V) = prob( Q K^T / sqrt(d_k) + bias ) V

    where ``prob`` is the configurable :class:`ProbabilityOutput` strategy and
    ``bias`` is the optional relative position bias. Restricting the sum to one
    window of ``M = window_size ** 2`` tokens is what turns global attention's
    ``O(N^2)`` cost into ``O(N * M)`` for an image of ``N`` tokens: the bias table
    is indexed by relative coordinate, so it has ``(2M_s - 1)^2`` entries for
    window side ``M_s`` rather than one entry per absolute position pair.

References:
    - Liu et al., 2021. Swin Transformer: Hierarchical Vision Transformer
      using Shifted Windows. (https://arxiv.org/abs/2103.14030)
    - Vaswani et al., 2017. Attention Is All You Need.
      (https://arxiv.org/abs/1706.03762)

"""

# ---------------------------------------------------------------------

import keras
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .common import apply_attention_mask
from ..ffn.kan_linear import KANLinear
from ..activations import ProbabilityOutput
from ..norms.factory import create_normalization_layer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SingleWindowAttention(keras.layers.Layer):
    """
    Unified multi-head self-attention for a single window.

    Merges multiple attention mechanisms into a single configurable layer
    supporting standard linear QKV projection or non-linear KAN-based Key
    projection, with a unified probability output strategy applied to
    attention scores (via :class:`ProbabilityOutput`). Internal padding
    ensures every window reaches ``window_size ** 2`` tokens before
    attention is computed, then strips padding from the output.

    The scaled dot-product attention is computed as
    ``Attention(Q, K, V) = prob(Q K^T / sqrt(d_k) + bias) V``, where ``prob``
    is the configurable :class:`ProbabilityOutput` strategy and ``bias`` is
    an optional learnable relative position bias table indexed by intra-window
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
        │   │     prob = ProbabilityOutput(               │     │
        │   │              probability_type, config)      │     │
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
    :param probability_type: Probability strategy identifier forwarded to
        :class:`ProbabilityOutput` for converting attention scores into
        attention weights. Defaults to ``'softmax'``. Score-level routing
        strategies (``'routing'``, ``'deterministic_routing'``,
        ``'hierarchical'``, ``'hierarchical_routing'``) are not allowed.
    :type probability_type: str
    :param probability_config: Optional configuration dictionary forwarded
        to :class:`ProbabilityOutput` as its ``type_config`` argument.
    :type probability_config: Optional[Dict[str, Any]]
    :param qk_norm_type: Optional normalization type applied independently
        to ``Q`` and ``K`` before computing attention scores. When provided,
        normalization layers are constructed via
        :func:`create_normalization_layer`. Defaults to ``None`` (no QK-norm).
    :type qk_norm_type: Optional[str]
    :param qk_norm_kwargs: Optional keyword arguments forwarded to
        :func:`create_normalization_layer` when ``qk_norm_type`` is set.
    :type qk_norm_kwargs: Optional[Dict[str, Any]]
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
        ``{'linear', 'kan_key'}`` or if ``probability_type`` is a score-level
        routing strategy (``'routing'``, ``'deterministic_routing'``,
        ``'hierarchical'``, ``'hierarchical_routing'``).
    """

    def __init__(
            self,
            dim: int,
            window_size: int,
            num_heads: int,
            attention_mode: str = "linear",
            use_relative_position_bias: bool = True,
            qkv_bias: bool = True,
            qk_scale: Optional[float] = None,
            dropout_rate: float = 0.0,
            proj_bias: bool = True,
            kan_grid_size: int = 5,
            kan_spline_order: int = 3,
            kan_activation: str = "swish",
            probability_type: str = "softmax",
            probability_config: Optional[Dict[str, Any]] = None,
            qk_norm_type: Optional[str] = None,
            qk_norm_kwargs: Optional[Dict[str, Any]] = None,
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
        invalid_prob_types = {
            "routing",
            "deterministic_routing",
            "hierarchical",
            "hierarchical_routing",
        }
        if probability_type in invalid_prob_types:
            raise ValueError(
                f"Invalid probability_type '{probability_type}'. Score-level "
                f"routing strategies {invalid_prob_types} are not allowed for "
                f"SingleWindowAttention."
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
        self.use_relative_position_bias = use_relative_position_bias
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.dropout_rate = dropout_rate
        self.proj_bias = proj_bias
        self.kan_grid_size = kan_grid_size
        self.kan_spline_order = kan_spline_order
        self.kan_activation = kan_activation
        self.probability_type = probability_type
        self.probability_config = probability_config
        self.qk_norm_type = qk_norm_type
        self.qk_norm_kwargs = qk_norm_kwargs
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

        self.attn_prob = ProbabilityOutput(
            probability_type=self.probability_type,
            type_config=self.probability_config,
            name="attn_prob",
        )

        if self.qk_norm_type is not None:
            self.q_norm = create_normalization_layer(
                self.qk_norm_type,
                name="q_norm",
                **(self.qk_norm_kwargs or {}),
            )
            self.k_norm = create_normalization_layer(
                self.qk_norm_type,
                name="k_norm",
                **(self.qk_norm_kwargs or {}),
            )
        else:
            self.q_norm = None
            self.k_norm = None

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
        if self.built:
            return

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

        self.attn_prob.build(attention_scores_shape)

        if self.q_norm is not None:
            qk_shape = (
                input_shape[0],
                self.num_heads,
                num_tokens_in_window,
                self.head_dim,
            )
            self.q_norm.build(qk_shape)
            self.k_norm.build(qk_shape)

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
        # Shape: (B, N_actual, dim) -> (B, N_target, dim), N_target = window_size**2
        padded_inputs = keras.ops.pad(
            inputs, [[0, 0], [0, padding_amount], [0, 0]]
        )

        # Shape: (B, N_actual) + (B, padding_amount) -> (B, N_target)
        internal_padding_mask = keras.ops.concatenate(
            [
                keras.ops.ones((B_actual, N_actual), dtype="int32"),
                keras.ops.zeros((B_actual, padding_amount), dtype="int32"),
            ],
            axis=1,
        )

        final_attention_mask = internal_padding_mask
        if attention_mask is not None:
            # Shape: (B, N_actual) -> (B, N_target)
            padded_user_mask = keras.ops.pad(
                attention_mask, [[0, 0], [0, padding_amount]]
            )
            final_attention_mask = (
                    keras.ops.cast(padded_user_mask, "int32")
                    * internal_padding_mask
            )

        B, N, C = keras.ops.shape(padded_inputs)
        if self.attention_mode == "linear":
            # Shape: (B, N, dim) -> (B, N, 3*dim)
            qkv = self.qkv(padded_inputs, training=training)
            # Shape: (B, N, 3*dim) -> (B, N, 3, H, head_dim)
            qkv = keras.ops.reshape(
                qkv, (B, N, 3, self.num_heads, self.head_dim)
            )
            # Shape: (B, N, 3, H, head_dim) -> (3, B, H, N, head_dim)
            qkv = keras.ops.transpose(qkv, (2, 0, 3, 1, 4))
            # Shape: (3, B, H, N, head_dim) -> 3x (B, H, N, head_dim)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            q_proj = self.query(padded_inputs, training=training)
            k_proj = self.key(padded_inputs, training=training)
            v_proj = self.value(padded_inputs, training=training)
            # Shape: (B, N, dim) -> (B, N, H, head_dim) -> (B, H, N, head_dim), each
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

        if self.q_norm is not None:
            q = self.q_norm(q, training=training)
            k = self.k_norm(k, training=training)

        q = q * self.scale
        # Shape: (B, H, N, head_dim) @ (B, H, head_dim, N) -> (B, H, N, N)
        attn = keras.ops.matmul(q, keras.ops.transpose(k, (0, 1, 3, 2)))

        if self.use_relative_position_bias:
            # Shape: table (2*Ws-1)^2 x H, gathered by (N_target*N_target,) index
            #        -> (N_target*N_target, H)
            relative_position_bias = keras.ops.take(
                self.relative_position_bias_table,
                keras.ops.reshape(self.relative_position_index, (-1,)),
                axis=0,
            )
            # Shape: (N_target*N_target, H) -> (N_target, N_target, H)
            relative_position_bias = keras.ops.reshape(
                relative_position_bias, (N_target, N_target, -1)
            )
            # Shape: (N_target, N_target, H) -> (H, N_target, N_target)
            relative_position_bias = keras.ops.transpose(
                relative_position_bias, (2, 0, 1)
            )
            # Shape: (B, H, N, N) + (1, H, N, N) -> (B, H, N, N)
            attn = attn + keras.ops.expand_dims(relative_position_bias, 0)

        # DECISION plan-2026-07-27T183600-b4ef45f0/D-010
        # `clip(attn, -30, 30)` runs HERE, on the RAW scores, and must NOT be moved
        # back below the mask bias where it used to live.
        #
        # WHAT NOT TO DO, and why:
        #   * Do NOT re-order this after `apply_attention_mask`. Clipping the BIASED
        #     logits floors a masked position at `-30` instead of `MASK_BIAS_VALUE`,
        #     which turns a hard mask into a soft one: once every logit in a row
        #     already sits below the floor, the mask stops masking. MEASURED in
        #     float32 on unfixed HEAD, driving `relative_position_bias_table` to a
        #     uniform value and perturbing a MASKED token: leak 0.0 at bias 0.0,
        #     2.73e-04 at -20, and **0.439** at -50, against a kept-token signal of
        #     32.6. Pinned by
        #     `TestSingleWindowAttentionClipDoesNotFloorTheMask`.
        #   * Do NOT delete the clip instead. Its job — bounding the attention
        #     logits before the softmax — is independent of masking and is
        #     unaffected by this move; the raw scores are exactly what it was
        #     meant to bound.
        # ACCEPTED COST: for a mask that leaves every row's logits inside
        # [-30, 30] (the ordinary case) this changes nothing measurable, but a
        # masked position's softmax weight now goes to exactly 0 rather than
        # `exp(-30 - max)`. That is a deliberate numerics change in float32 too,
        # not only fp16.
        # See decisions.md D-010 (plan-2026-07-27T183600-b4ef45f0).
        attn = keras.ops.clip(attn, -30.0, 30.0)

        # THIS SITE'S MASK POLARITY, passed through verbatim: `broadcast_mask` is a
        # `1 = keep` predicate, so it IS the keep predicate `apply_attention_mask`
        # wants. Do NOT "normalize" it into a `> 0` comparison or invert it — the
        # helper performs no polarity inference by design, so an inversion here
        # raises nothing, changes no shape and stays finite; the layer would just
        # attend to the padding. `TestSingleWindowAttentionMaskPolarity` is the only
        # guard that can see it.
        #
        # DECISION plan-2026-07-27T183600-b4ef45f0/D-007
        # `out_dtype` is pinned to the SCORES' own dtype, so the biased scores stay
        # in the compute dtype (fp16 under `mixed_float16`), where
        # `MASK_BIAS_VALUE` is `-inf` again. That is deliberate and is NOT the bug
        # being fixed:
        #   * The bug was the ARITHMETIC form this replaces,
        #     `attn + (1.0 - keep) * -1e9`. In float16 `-1e9` is `-inf`
        #     (np.float16(-1e9) == -inf) and `(1.0 - keep) == 0` at every UNMASKED
        #     position, so the product is `0 * -inf = NaN`. THIS SITE HAD NO SAFE
        #     PATH AT ALL: the mask is unconditional here (the internal padding mask
        #     is always built and always applied), so `attention_mask=None` went
        #     down the same line. MEASURED at (B=2, N=64, dim=64, window_size=8,
        #     num_heads=4) under `mixed_float16`: 8192/8192 NaN with NO mask, with
        #     an all-ones mask, with right padding and with left padding; float32
        #     gave 0/8192 in every case.
        #   * Do NOT "improve" this to `out_dtype=None` (stay in float32) hoping to
        #     also rescue a fully-masked element. It cannot: the next consumer is
        #     `self.attn_prob`, a Keras layer with autocasting ON, MEASURED to see a
        #     float32 input inside its own `call()` as float16. Pinned by
        #     `TestSingleWindowAttentionMaskHazardIsReal::
        #     test_the_probability_sublayer_autocasts_a_float32_input`.
        # See decisions.md D-007 (plan-2026-07-27T183600-b4ef45f0).
        #
        # DECISION plan-2026-07-27T183600-b4ef45f0/D-009
        # The fully-masked-slice rescue arrives via `apply_attention_mask`'s DEFAULT
        # `rescue_axis=-1`: a slice of the mask that keeps NOTHING is treated as
        # keeping EVERYTHING, so an all-`-inf` row is never FORMED and no NaN
        # gradient is created either. Note the mask has no QUERY axis at this site,
        # so "keeps nothing" means "this batch element keeps no key at all".
        #
        # DECISION plan-2026-07-27T183600-b4ef45f0/D-017
        # The axis is DERIVED from this layer's own `probability_config` rather than
        # left to the helper's `-1` default: `ProbabilityOutput` reads its softmax
        # `axis` from `type_config` (`activations/probability_output.py:180`) and this
        # layer forwards `probability_config` VERBATIM, so a caller can move the
        # reduction axis and the pre-step-10 "checked, not assumed" claim held only for
        # the DEFAULT config. MEASURED at the sibling `gated_attention` under
        # `mixed_float16` with `probability_config={"axis": -2}` and a dead KEY COLUMN:
        # 8192/8192 non-finite. This site is also the one where the D-017 size-1
        # rejection is most visible: its mask is ALWAYS reshaped to `(B, 1, 1, N)`, so a
        # caller configuring `axis=-2` (a softmax over queries) now gets a named
        # `ValueError` rather than a mask that cannot mask. WHAT NOT TO DO: do NOT
        # restore a bare `-1`, and do NOT read this as the rank/shape INFERENCE the
        # D-009 anchor in `common.py` forbids — this reads the site's own declared
        # config. See decisions.md D-017 (plan-2026-07-27T183600-b4ef45f0).
        #
        # WHAT NOT TO DO: do NOT pass `rescue_axis=None` to "get the loud NaN back"
        # — the user ruled the finite-garbage semantics package-wide on 2026-07-28,
        # and opting out also restores the NaN GRADIENT. The full argument lives at
        # the D-009 / D-008 anchors in `common.py`.
        # See decisions.md D-009 and D-008 (plan-2026-07-27T183600-b4ef45f0).
        # Shape: (B, N) -> (B, 1, 1, N)  [broadcasts over heads and query axis]
        broadcast_mask = keras.ops.reshape(final_attention_mask, (B, 1, 1, N))
        attn = apply_attention_mask(
            attn,
            broadcast_mask,
            out_dtype=keras.backend.standardize_dtype(attn.dtype),
            rescue_axis=(self.probability_config or {}).get("axis", -1),
        )

        attn = self.attn_prob(attn, training=training)

        if self.attn_dropout is not None:
            attn = self.attn_dropout(attn, training=training)
        # Shape: (B, H, N, N) @ (B, H, N, head_dim) -> (B, H, N, head_dim)
        x = keras.ops.matmul(attn, v)
        # Shape: (B, H, N, head_dim) -> (B, N, H, head_dim)
        x = keras.ops.transpose(x, (0, 2, 1, 3))
        # Shape: (B, N, H, head_dim) -> (B, N, dim)
        x = keras.ops.reshape(x, (B, N, C))
        # Shape: (B, N, dim) -> (B, N, dim)
        x = self.proj(x, training=training)

        # Shape: (B, N_target, dim) -> (B, N_actual, dim)  [strip stage-1 padding]
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
                "use_relative_position_bias": self.use_relative_position_bias,
                "qkv_bias": self.qkv_bias,
                "qk_scale": self.qk_scale,
                "dropout_rate": self.dropout_rate,
                "proj_bias": self.proj_bias,
                "kan_grid_size": self.kan_grid_size,
                "kan_spline_order": self.kan_spline_order,
                "kan_activation": self.kan_activation,
                "probability_type": self.probability_type,
                "probability_config": self.probability_config,
                "qk_norm_type": self.qk_norm_type,
                "qk_norm_kwargs": self.qk_norm_kwargs,
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
