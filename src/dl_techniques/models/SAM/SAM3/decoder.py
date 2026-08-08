"""
SAM 3 DETR Decoder: three attention sub-blocks, boxRPB, and a presence token.
=============================================================================

:class:`Sam3DecoderLayer` is ONE layer of SAM 3's detection decoder;
:class:`Sam3TransformerDecoder` is the stack that repeats it, refines the
reference boxes and reads out the per-layer presence logits.

Based on:
---------
- Ravi, N. et al. (2025). "SAM 3: Segment Anything with Concepts."
- Carion, N. et al. (2020). DETR -- the decoder shape this one extends.
- Liu, S. et al. (2023). Grounding DINO -- the text cross-attention sub-block.

Key Features:
------------
- THREE attention sub-blocks per layer, not DETR's two.
- boxRPB: a log-compressed, box-conditioned relative position bias added to the
  RAW image-cross-attention scores, per head and per query, before the softmax.
- A presence token that rides the query sequence and is split back off.

Architecture Overview:
---------------------
1. **Self-attention**: ``q = k = tgt + query_pos``, ``v = tgt``; residual, norm2.
2. **Text cross-attention**: ``q = tgt + query_pos``, ``k = v = text memory``;
   residual, catext_norm.
3. **Image cross-attention**: ``q = tgt + query_pos``, ``k = image memory +
   memory_pos``, ``v = image memory``, ``scores += boxRPB``; residual, norm1.
4. **Feed-forward**: ``fc1 -> relu -> drop -> fc2 -> drop``; residual, norm3.
Settled configuration: ``d_model=256``, ``num_heads=8``,
``dim_feedforward=2048``, ``dropout_rate=0.1``, ``relu``, ``box_rpb="log"``.

Usage Examples:
--------------
```python
from dl_techniques.models.SAM.SAM3.decoder import Sam3TransformerDecoder
decoder = Sam3TransformerDecoder(d_model=256, num_heads=8, num_layers=6,
                                 num_queries=200, feat_size=(72, 72))
```

Measured caveats:
----------------
- No attention layer in this repository can carry the real-valued additive
  boxRPB bias into raw scores -- see the ``D-080`` anchor on
  ``_Sam3DecoderAttention`` and the measurement recorded there.
- Neither self- nor image cross-attention draws ``k`` and ``v`` from the same
  tensor: ``k`` carries a positional embedding ``v`` does not. That single
  asymmetry disqualifies the repo's cross-attention layer at those two sites and
  does NOT disqualify it at the text site, which uses it unmodified.
- The presence token has a ZEROED query position and an all-zero bias row, so it
  attends everywhere in image cross-attention whatever boxRPB says.
- The reference shares one pair of boxRPB embedding MLPs across every layer, so
  making them per-layer multiplies their parameters by the layer count with no
  shape symptom.
"""

import keras
import math
from keras import layers, ops
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.attention.factory import create_attention_layer
from dl_techniques.layers.ffn.factory import create_ffn_layer
from dl_techniques.layers.norms.factory import create_normalization_layer
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

# The log compression maps a delta of 8 normalized-image-units to +-1; the
# divisor is therefore log2(8) and NOT a free constant.
_BOX_RPB_SCALE = 8.0
_BOX_RPB_LOG_DIVISOR = math.log2(_BOX_RPB_SCALE)


def _box_rpb_log_compress(deltas: Any) -> Any:
    """Sign-preserving log compression of boxRPB's signed grid deltas.

    ``d' = d * 8``; ``d'' = sign(d') * log2(|d'| + 1) / log2(8)``.

    Three parts of that expression are each load-bearing and each has a probe
    in the guard set, because a candidate that drops one of them still returns
    a finite tensor of the right shape:

    - the ``sign`` factor -- dropping it makes the bias blind to which side of
      the box a grid position is on (it differs only at NEGATIVE deltas);
    - the ``+ 1`` -- dropping it makes ``d = 0`` evaluate ``log2(0) = -inf``;
    - the ``/ log2(8)`` -- dropping it triples the bias (it differs everywhere
      except at ``d = 0``).

    :param deltas: Signed deltas in normalized image units, any shape.
    :type deltas: Any
    :return: Compressed deltas, same shape and dtype.
    :rtype: Any
    """
    scaled = deltas * _BOX_RPB_SCALE
    compressed = ops.log2(ops.abs(scaled) + 1.0) / _BOX_RPB_LOG_DIVISOR
    return ops.sign(scaled) * compressed


def _box_rpb_bias(
        reference_boxes: Any,
        feat_size: Tuple[int, int],
        embed_x: keras.layers.Layer,
        embed_y: keras.layers.Layer,
        num_heads: int,
        mode: str = "log",
) -> Any:
    """Build boxRPB's per-head image cross-attention bias.

    For every query, the signed distance from each grid row to the box's top
    and bottom edges, and from each grid column to its left and right edges,
    is embedded per head by two independent MLPs and combined as an OUTER SUM.
    The result is the additive bias that the image cross-attention adds to its
    raw scores.

    :param reference_boxes: ``(batch, num_queries, 4)`` boxes in normalized
        ``cxcywh`` form -- the same parameterization the box-refinement chain
        carries.
    :type reference_boxes: Any
    :param feat_size: Image-memory grid ``(height, width)``. Its product must
        equal the image memory's key count.
    :type feat_size: Tuple[int, int]
    :param embed_x: Per-head embedding of the two column deltas; maps
        ``(..., 2) -> (..., num_heads)``.
    :type embed_x: keras.layers.Layer
    :param embed_y: The independent row-delta counterpart.
    :type embed_y: keras.layers.Layer
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param mode: ``"log"`` for the compressed form, ``"linear"`` for the raw
        deltas.
    :type mode: str
    :return: ``(batch, num_heads, num_queries, height * width)``.
    :rtype: Any
    :raises ValueError: If ``mode`` is not ``"log"`` or ``"linear"``.
    """
    if mode not in ("log", "linear"):
        raise ValueError(f"mode must be 'log' or 'linear', got {mode!r}")

    height, width = int(feat_size[0]), int(feat_size[1])
    dtype = reference_boxes.dtype

    centre_x, centre_y, box_w, box_h = ops.split(reference_boxes, 4, axis=-1)
    edges_x = ops.concatenate(
        [centre_x - 0.5 * box_w, centre_x + 0.5 * box_w], axis=-1)
    edges_y = ops.concatenate(
        [centre_y - 0.5 * box_h, centre_y + 0.5 * box_h], axis=-1)

    # Normalized grid coordinates: arange(N) / N, matching the reference.
    coords_y = ops.cast(ops.arange(height), dtype) / float(height)
    coords_x = ops.cast(ops.arange(width), dtype) / float(width)

    deltas_y = (ops.reshape(coords_y, (1, 1, height, 1))
                - ops.expand_dims(edges_y, axis=2))
    deltas_x = (ops.reshape(coords_x, (1, 1, width, 1))
                - ops.expand_dims(edges_x, axis=2))

    if mode == "log":
        deltas_y = _box_rpb_log_compress(deltas_y)
        deltas_x = _box_rpb_log_compress(deltas_x)

    bias_y = embed_y(deltas_y)  # (batch, num_queries, height, num_heads)
    bias_x = embed_x(deltas_x)  # (batch, num_queries, width,  num_heads)

    # Outer SUM over the two axes -- not a concatenation and not a product.
    bias = ops.expand_dims(bias_y, 3) + ops.expand_dims(bias_x, 2)
    bias = ops.transpose(bias, (0, 4, 1, 2, 3))
    return ops.reshape(
        bias, (-1, num_heads, ops.shape(bias)[2], height * width))


# DECISION plan-2026-08-04T044628-4c240b4c/D-080
# This attention class is the ONE module-private bias-injection helper this
# file is permitted, and it is deliberately UNREGISTERED (the `_SAM2RoPEAttention`
# / `_Sam3ViTDetAttention` precedent, D-008 / D-085). Do NOT replace it with
# `create_attention_layer('multi_head_cross', ...)` at either of its two call
# sites, and do NOT route the boxRPB bias through any `attention_mask=`
# parameter. Two independent contract failures, both MEASURED:
#   (a) `layers/attention/common.apply_attention_mask` treats `keep` as BINARY
#       (`> 0`). Handed boxRPB's real-valued bias it does NOT raise and does NOT
#       full-keep: it BINARIZES -- every positive entry gets no bias at all and
#       every non-positive entry gets the hard -1e9 mask. Measured max
#       per-position softmax deviation from the true additive bias: 0.366. A
#       silent value defect with no shape symptom.
#   (b) `MultiHeadCrossAttention` derives k AND v from ONE `kv_dense` on ONE
#       tensor. Both this layer's call sites need k from a POSITION-EMBEDDED
#       tensor and v from the un-embedded one, which that contract cannot
#       express at any configuration. (The text cross-attention site does NOT
#       have this asymmetry and therefore DOES use the stock layer.)
# See decisions.md D-080, D-107, D-109.
class _Sam3DecoderAttention(keras.layers.Layer):
    """Multi-head attention over three independent tensors, with a bias hook.

    Private implementation detail of :class:`Sam3DecoderLayer`. Unlike a stock
    cross-attention layer it projects ``query``, ``key`` and ``value``
    separately, so keys may carry a positional embedding that values do not,
    and it accepts a real-valued per-head additive bias that is added to the
    RAW scores before the softmax.

    :param d_model: Model width; also the output width.
    :type d_model: int
    :param num_heads: Number of heads. Must divide ``d_model``.
    :type num_heads: int
    :param dropout_rate: Dropout applied to the attention probabilities.
    :type dropout_rate: float
    :param use_bias: Whether the four projections carry biases.
    :type use_bias: bool
    :param kwargs: Additional ``Layer`` keyword arguments.
    :raises ValueError: If ``num_heads`` does not divide ``d_model``.
    """

    def __init__(
            self, d_model: int, num_heads: int, dropout_rate: float = 0.0,
            use_bias: bool = True, **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by "
                             f"num_heads ({num_heads})")

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout_rate)
        self.use_bias = bool(use_bias)
        self.head_dim = self.d_model // self.num_heads
        self._scale = 1.0 / math.sqrt(float(self.head_dim))

        self.q_proj = layers.Dense(self.d_model, use_bias=self.use_bias,
                                   name="q_proj")
        self.k_proj = layers.Dense(self.d_model, use_bias=self.use_bias,
                                   name="k_proj")
        self.v_proj = layers.Dense(self.d_model, use_bias=self.use_bias,
                                   name="v_proj")
        self.out_proj = layers.Dense(self.d_model, use_bias=self.use_bias,
                                     name="out_proj")
        self.attn_dropout = layers.Dropout(self.dropout_rate,
                                           name="attn_dropout")

    def build(
            self, query_shape: Tuple[Optional[int], ...],
            key_shape: Tuple[Optional[int], ...],
            value_shape: Optional[Tuple] = None,
            additive_bias_shape: Optional[Tuple] = None,
    ) -> None:
        """Build the four projections.

        :param query_shape: ``(batch, num_queries, d_model)``.
        :type query_shape: Tuple[Optional[int], ...]
        :param key_shape: ``(batch, num_keys, d_model)``.
        :type key_shape: Tuple[Optional[int], ...]
        :param value_shape: Value shape; defaults to ``key_shape``.
        :type value_shape: Optional[Tuple[Optional[int], ...]]
        :param additive_bias_shape: Unused; accepted so the layer builds from
            its full call signature.
        :type additive_bias_shape: Optional[Tuple[Optional[int], ...]]
        """
        if self.built:
            return
        value_shape = key_shape if value_shape is None else value_shape
        self.q_proj.build(tuple(query_shape))
        self.k_proj.build(tuple(key_shape))
        self.v_proj.build(tuple(value_shape))
        self.out_proj.build(
            (query_shape[0], query_shape[1], self.d_model))
        super().build(query_shape)

    def _split_heads(self, x: Any) -> Any:
        """Reshape ``(B, N, d_model)`` to ``(B, heads, N, head_dim)``."""
        shape = ops.shape(x)
        x = ops.reshape(x, (shape[0], shape[1], self.num_heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))

    def call(
            self, query: Any, key: Any, value: Any,
            additive_bias: Optional[Any] = None,
            training: Optional[bool] = None,
    ) -> Any:
        """Attend, optionally biasing the raw scores.

        :param query: ``(batch, num_queries, d_model)``.
        :type query: Any
        :param key: ``(batch, num_keys, d_model)``.
        :type key: Any
        :param value: ``(batch, num_keys, d_model)``.
        :type value: Any
        :param additive_bias: ``(batch, num_heads, num_queries, num_keys)``
            real-valued per-head bias, or ``None``.
        :type additive_bias: Optional[Any]
        :param training: Training-mode flag; affects attention dropout.
        :type training: Optional[bool]
        :return: ``(batch, num_queries, d_model)``.
        :rtype: Any
        """
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self._scale
        if additive_bias is not None:
            # The bias is REAL-VALUED and enters here, on the raw scores,
            # before any normalization. See this class's D-080 anchor.
            scores = scores + ops.cast(additive_bias, scores.dtype)

        attn = self.attn_dropout(ops.softmax(scores, axis=-1),
                                 training=training)
        out = ops.matmul(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        shape = ops.shape(out)
        out = ops.reshape(out, (shape[0], shape[1], self.d_model))
        return self.out_proj(out)

    def compute_output_shape(
            self, query_shape: Tuple[Optional[int], ...],
            key_shape: Optional[Tuple] = None,
            value_shape: Optional[Tuple] = None,
            additive_bias_shape: Optional[Tuple] = None,
    ) -> Tuple[Optional[int], ...]:
        """Return ``(batch, num_queries, d_model)``."""
        return (query_shape[0], query_shape[1], self.d_model)

    def get_config(self) -> Dict[str, Any]:
        """Return every ``__init__`` parameter."""
        config = super().get_config()
        config.update({
            "d_model": self.d_model, "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate, "use_bias": self.use_bias,
        })
        return config


@keras.saving.register_keras_serializable()
class Sam3DecoderLayer(keras.layers.Layer):
    """One SAM 3 detection-decoder layer: self, text-cross, image-cross, FFN.

    :param d_model: Model width. Default: ``256``.
    :type d_model: int
    :param num_heads: Heads in all three attention sub-blocks. Default: ``8``.
    :type num_heads: int
    :param dim_feedforward: Hidden width of the feed-forward block.
        Default: ``2048``.
    :type dim_feedforward: int
    :param dropout_rate: Dropout used by every sub-block. Default: ``0.1``.
    :type dropout_rate: float
    :param activation: Feed-forward activation. Default: ``"relu"``.
    :type activation: str
    :param use_text_cross_attention: Whether the text cross-attention sub-block
        exists. Default: ``True``.
    :type use_text_cross_attention: bool
    :param norm_epsilon: Epsilon of all four normalizations. Default: ``1e-5``
        -- the reference's value, NOT the Keras default of ``1e-3``.
    :type norm_epsilon: float
    :raises ValueError: On a non-positive width or an out-of-range dropout.

    Example:
        >>> import numpy as np
        >>> layer = Sam3DecoderLayer(d_model=8, num_heads=2,
        ...                          dim_feedforward=16, dropout_rate=0.0)
        >>> tgt = np.zeros((2, 5, 8), dtype="float32")
        >>> memory = np.zeros((2, 9, 8), dtype="float32")
        >>> text = np.zeros((2, 4, 8), dtype="float32")
        >>> out, presence = layer(tgt, memory, memory_text=text)
        >>> out.shape, presence
        ((2, 5, 8), None)
    """

    def __init__(
            self, d_model: int = 256, num_heads: int = 8,
            dim_feedforward: int = 2048, dropout_rate: float = 0.1,
            activation: str = "relu", use_text_cross_attention: bool = True,
            norm_epsilon: float = 1e-5, **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        for name, value in (("d_model", d_model), ("num_heads", num_heads),
                            ("dim_feedforward", dim_feedforward)):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by "
                             f"num_heads ({num_heads})")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got "
                             f"{dropout_rate}")

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout_rate = float(dropout_rate)
        self.activation = str(activation)
        self.use_text_cross_attention = bool(use_text_cross_attention)
        self.norm_epsilon = float(norm_epsilon)

        # Sub-layers, created unconditionally and stored FLAT (never in a
        # nested list -- a `List[List[Layer]]` store silently restores fresh
        # weights on a `.keras` round trip while every count and path matches,
        # measured in this package; see decisions.md D-098).
        self.self_attn = _Sam3DecoderAttention(
            self.d_model, self.num_heads, self.dropout_rate, name="self_attn")
        self.dropout2 = layers.Dropout(self.dropout_rate, name="dropout2")
        self.norm2 = create_normalization_layer(
            "layer_norm", epsilon=self.norm_epsilon, name="norm2")

        # The text sub-block is the ONE site whose keys and values come from
        # the same tensor, so the repository's cross-attention layer expresses
        # it exactly. Structural kwargs are set EXPLICITLY rather than inherited
        # from that layer's defaults (D-102).
        self.ca_text = create_attention_layer(
            "multi_head_cross", dim=self.d_model, num_heads=self.num_heads,
            dropout_rate=self.dropout_rate, use_bias=True,
            shared_qk_projections=False, probability_type="softmax",
            qk_norm_type=None, name="ca_text")
        self.catext_dropout = layers.Dropout(self.dropout_rate,
                                             name="catext_dropout")
        self.catext_norm = create_normalization_layer(
            "layer_norm", epsilon=self.norm_epsilon, name="catext_norm")

        self.cross_attn = _Sam3DecoderAttention(
            self.d_model, self.num_heads, self.dropout_rate, name="cross_attn")
        self.dropout1 = layers.Dropout(self.dropout_rate, name="dropout1")
        self.norm1 = create_normalization_layer(
            "layer_norm", epsilon=self.norm_epsilon, name="norm1")

        # `MLPBlock` is fc1 -> activation -> dropout -> fc2, which is exactly
        # the reference's `linear2(dropout3(relu(linear1(x))))`. Its OWN
        # docstring says the dropout is applied after both dense layers; the
        # code applies it once, after the activation. The code is what runs.
        self.ffn = create_ffn_layer(
            "mlp", hidden_dim=self.dim_feedforward, output_dim=self.d_model,
            activation=self.activation, dropout_rate=self.dropout_rate,
            use_bias=True, name="ffn")
        self.dropout4 = layers.Dropout(self.dropout_rate, name="dropout4")
        self.norm3 = create_normalization_layer(
            "layer_norm", epsilon=self.norm_epsilon, name="norm3")

        logger.info(
            f"Sam3DecoderLayer: d_model={self.d_model}, "
            f"heads={self.num_heads}, ffn={self.dim_feedforward}, "
            f"text_cross={self.use_text_cross_attention}"
        )

    def build(
            self, tgt_shape: Tuple[Optional[int], ...],
            memory_shape: Tuple[Optional[int], ...],
            memory_text_shape: Optional[Tuple] = None,
            **kwargs: Any,
    ) -> None:
        """Build every sub-block explicitly.

        :param tgt_shape: ``(batch, num_queries, d_model)``. When the presence
            token is used the query axis is one longer at call time; no
            sub-layer here depends on that extent.
        :type tgt_shape: Tuple[Optional[int], ...]
        :param memory_shape: ``(batch, num_keys, d_model)``.
        :type memory_shape: Tuple[Optional[int], ...]
        :param memory_text_shape: ``(batch, num_tokens, d_model)``; required
            when text cross-attention is enabled.
        :type memory_text_shape: Optional[Tuple[Optional[int], ...]]
        :param kwargs: Ignored; accepted so the layer builds from its full call
            signature.
        :raises ValueError: On a wrong rank or a width other than ``d_model``.
        """
        if self.built:
            return
        for name, shape in (("tgt", tgt_shape), ("memory", memory_shape)):
            if len(shape) != 3:
                raise ValueError(f"{name} must have shape (batch, seq, "
                                 f"d_model), got {shape}")
            if shape[-1] is not None and shape[-1] != self.d_model:
                raise ValueError(f"{name} width {shape[-1]} != d_model "
                                 f"{self.d_model}")
        tgt_shape, memory_shape = tuple(tgt_shape), tuple(memory_shape)

        self.self_attn.build(tgt_shape, tgt_shape, tgt_shape)
        self.dropout2.build(tgt_shape)
        self.norm2.build(tgt_shape)

        # The text sub-block's weights exist only when it is enabled: an
        # UNBUILT Keras layer holds no variables, so a disabled sub-block costs
        # exactly zero parameters while the attribute still always exists
        # (the D-012 precedent -- no conditional attribute, no `call()` branch
        # over structure).
        if self.use_text_cross_attention:
            if memory_text_shape is None:
                raise ValueError(
                    "memory_text_shape is required when "
                    "use_text_cross_attention=True")
            memory_text_shape = tuple(memory_text_shape)
            self.ca_text.build([tgt_shape, memory_text_shape])
            self.catext_dropout.build(tgt_shape)
            self.catext_norm.build(tgt_shape)

        self.cross_attn.build(tgt_shape, memory_shape, memory_shape)
        self.dropout1.build(tgt_shape)
        self.norm1.build(tgt_shape)

        self.ffn.build(tgt_shape)
        self.dropout4.build(tgt_shape)
        self.norm3.build(tgt_shape)
        super().build(tgt_shape)

    @staticmethod
    def _with_pos(tensor: Any, pos: Optional[Any]) -> Any:
        """Add a positional embedding when one is supplied."""
        return tensor if pos is None else tensor + pos

    def call(
            self, tgt: Any, memory: Any,
            query_pos: Optional[Any] = None,
            memory_pos: Optional[Any] = None,
            memory_text: Optional[Any] = None,
            text_padding_mask: Optional[Any] = None,
            image_cross_bias: Optional[Any] = None,
            memory_mask: Optional[Any] = None,
            presence_token: Optional[Any] = None,
            training: Optional[bool] = None,
    ) -> Tuple[Any, Optional[Any]]:
        """Run the four residual sub-blocks in order.

        :param tgt: Object queries ``(batch, num_queries, d_model)``.
        :type tgt: Any
        :param memory: Image memory ``(batch, height * width, d_model)``.
        :type memory: Any
        :param query_pos: Per-query positional embedding, same shape as ``tgt``.
        :type query_pos: Optional[Any]
        :param memory_pos: Per-key positional embedding, same shape as
            ``memory``. Added to the KEYS only, never to the values.
        :type memory_pos: Optional[Any]
        :param memory_text: Text memory ``(batch, num_tokens, d_model)``.
        :type memory_text: Optional[Any]
        :param text_padding_mask: ``(batch, num_tokens)``, ``True`` at PADDING
            positions -- the key-padding convention, the OPPOSITE polarity from
            the causal keep-mask the text tower builds.
        :type text_padding_mask: Optional[Any]
        :param image_cross_bias: boxRPB bias
            ``(batch, num_heads, num_queries, height * width)``.
        :type image_cross_bias: Optional[Any]
        :param memory_mask: Must be ``None``; see the raise below.
        :type memory_mask: Optional[Any]
        :param presence_token: ``(batch, 1, d_model)`` presence token, or
            ``None``.
        :type presence_token: Optional[Any]
        :param training: Training-mode flag.
        :type training: Optional[bool]
        :return: ``(tgt_out, presence_token_out)``; the second element is
            ``None`` when no presence token was supplied.
        :rtype: Tuple[Any, Optional[Any]]
        :raises ValueError: If ``memory_mask`` is supplied, or if text
            cross-attention is enabled and ``memory_text`` is missing.
        """
        # DECISION plan-2026-08-04T044628-4c240b4c/D-080
        # boxRPB REPLACES any other image cross-attention masking; the two are
        # mutually exclusive and the reference asserts the same thing. Do NOT
        # "helpfully" combine an external key mask with the bias here: a keep
        # mask goes through a binarizing helper that would silently discard the
        # bias's magnitude at every kept position (measured: max softmax
        # deviation 0.366). If phase 2 ever needs both, they must be summed as
        # ADDITIVE terms at this site, deliberately. See decisions.md D-080.
        if memory_mask is not None:
            raise ValueError(
                "Sam3DecoderLayer does not accept an external `memory_mask`: "
                "it is mutually exclusive with the boxRPB additive bias, which "
                "occupies the same slot in the image cross-attention. Pass the "
                "bias as `image_cross_bias` instead."
            )
        if self.use_text_cross_attention and memory_text is None:
            raise ValueError(
                "memory_text is required when use_text_cross_attention=True")

        if presence_token is not None:
            # The presence token joins the query sequence with a ZEROED
            # position: it is not a spatial query and must not be given one.
            tgt = ops.concatenate([presence_token, tgt], axis=1)
            if query_pos is not None:
                query_pos = ops.concatenate(
                    [ops.zeros_like(presence_token), query_pos], axis=1)

        # --- 1. self-attention ------------------------------------------
        q = self._with_pos(tgt, query_pos)
        attended = self.self_attn(q, q, tgt, training=training)
        tgt = self.norm2(tgt + self.dropout2(attended, training=training))

        # --- 2. text cross-attention ------------------------------------
        if self.use_text_cross_attention:
            keep = None
            if text_padding_mask is not None:
                keep = ops.logical_not(ops.cast(text_padding_mask, "bool"))
            attended = self.ca_text(
                self._with_pos(tgt, query_pos), memory_text,
                attention_mask=keep, training=training)
            tgt = self.catext_norm(
                tgt + self.catext_dropout(attended, training=training))

        # --- 3. image cross-attention, with the boxRPB bias -------------
        if presence_token is not None and image_cross_bias is not None:
            # A ZERO bias row for the presence token: it attends everywhere,
            # whatever per-query bias the real queries carry.
            zero_row = ops.zeros_like(image_cross_bias[:, :, :1, :])
            image_cross_bias = ops.concatenate(
                [zero_row, image_cross_bias], axis=2)
        attended = self.cross_attn(
            self._with_pos(tgt, query_pos),
            self._with_pos(memory, memory_pos), memory,
            additive_bias=image_cross_bias, training=training)
        tgt = self.norm1(tgt + self.dropout1(attended, training=training))

        # --- 4. feed-forward --------------------------------------------
        projected = self.ffn(tgt, training=training)
        tgt = self.norm3(tgt + self.dropout4(projected, training=training))

        if presence_token is None:
            return tgt, None
        return tgt[:, 1:], tgt[:, :1]

    def compute_output_shape(
            self, tgt_shape: Tuple[Optional[int], ...],
            memory_shape: Optional[Tuple] = None,
            memory_text_shape: Optional[Tuple] = None,
            **kwargs: Any,
    ) -> Tuple[Tuple, Tuple]:
        """Return the ``(tgt, presence_token)`` output shapes.

        :param tgt_shape: ``(batch, num_queries, d_model)``.
        :type tgt_shape: Tuple[Optional[int], ...]
        :param memory_shape: Unused.
        :type memory_shape: Optional[Tuple[Optional[int], ...]]
        :param memory_text_shape: Unused.
        :type memory_text_shape: Optional[Tuple[Optional[int], ...]]
        :param kwargs: Ignored.
        :return: ``((batch, num_queries, d_model), (batch, 1, d_model))``.
        :rtype: Tuple[Tuple, Tuple]
        """
        tgt_shape = tuple(tgt_shape)
        return (tgt_shape[0], tgt_shape[1], self.d_model), \
               (tgt_shape[0], 1, self.d_model)

    def get_config(self) -> Dict[str, Any]:
        """Return every ``__init__`` parameter.

        :return: Serializable configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "d_model": self.d_model, "num_heads": self.num_heads,
            "dim_feedforward": self.dim_feedforward,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "use_text_cross_attention": self.use_text_cross_attention,
            "norm_epsilon": self.norm_epsilon,
        })
        return config


@keras.saving.register_keras_serializable()
class Sam3TransformerDecoder(keras.layers.Layer):
    """SAM 3's detection-decoder stack: layers, box refinement, presence.

    Repeats :class:`Sam3DecoderLayer` ``num_layers`` times. Between layers it
    refines each query's reference box, rebuilds that query's conditional
    positional embedding and rebuilds boxRPB's additive bias from the refined
    box, and it reads a presence logit off the presence token after every
    layer. Every per-layer quantity is returned stacked on a LEADING
    ``num_layers`` axis, which is what an auxiliary-loss training phase needs.

    Four mechanisms carry this class's correctness, and each has its own guard:

    1. **The box delta is computed on the NORMED hidden state.** ``norm`` is
       applied first and the box head reads its output, not the raw one.
    2. **The refinement is additive in LOGIT space**:
       ``sigmoid(delta + inverse_sigmoid(reference))``, never
       ``reference + delta``.
    3. **The next layer's reference is DETACHED.** Gradients do not flow from a
       later layer back through the reference chain into an earlier layer's box
       head -- the reference is a per-layer anchor, not a differentiable path.
    4. **The box head's last projection is ZERO-initialized**, so at
       initialization layer 0 leaves its reference box exactly where it was.

    DAC query doubling is deliberately NOT implemented: the reference gates it
    on ``self.training`` at its only call site, so it is provably inert at
    inference. It is named in the package docstring rather than left absent.

    :param d_model: Model width. Default: ``256``.
    :type d_model: int
    :param num_heads: Attention heads per layer, and boxRPB's bias width.
        Default: ``8``.
    :type num_heads: int
    :param num_layers: Number of decoder layers. Default: ``6``.
    :type num_layers: int
    :param num_queries: Number of object queries. Default: ``200``.
    :type num_queries: int
    :param feat_size: Image-memory grid ``(height, width)``. Default:
        ``(72, 72)`` -- the settled ``resolution // stride = 1008 // 14``.
    :type feat_size: Tuple[int, int]
    :param dim_feedforward: Per-layer feed-forward width. Default: ``2048``.
    :type dim_feedforward: int
    :param dropout_rate: Per-layer dropout. Default: ``0.1``.
    :type dropout_rate: float
    :param activation: Per-layer feed-forward activation. Default: ``"relu"``.
    :type activation: str
    :param use_text_cross_attention: Whether each layer has the text
        cross-attention sub-block. Default: ``True``.
    :type use_text_cross_attention: bool
    :param box_rpb: ``"log"``, ``"linear"`` or ``"none"``. Default: ``"log"``.
    :type box_rpb: str
    :param use_presence_token: Whether the presence token and its readout head
        exist. Default: ``True``.
    :type use_presence_token: bool
    :param clamp_presence_logits: Whether presence logits are clamped.
        Default: ``True``.
    :type clamp_presence_logits: bool
    :param clamp_presence_logit_max_val: The symmetric presence clamp bound.
        Default: ``10.0`` -- deliberately NOT the scorer's ``12.0``.
    :type clamp_presence_logit_max_val: float
    :param use_normed_output_consistently: Whether the box delta reads the
        NORMED hidden state. Default: ``True``.
    :type use_normed_output_consistently: bool
    :param norm_epsilon: Epsilon of every normalization. Default: ``1e-5``.
    :type norm_epsilon: float
    :raises ValueError: On a non-positive size, an unknown ``box_rpb`` mode, or
        a ``feat_size`` that is not a pair of positive integers.

    Example:
        >>> import numpy as np
        >>> stack = Sam3TransformerDecoder(d_model=8, num_heads=2,
        ...                                num_layers=2, num_queries=5,
        ...                                feat_size=(3, 4),
        ...                                dim_feedforward=16,
        ...                                dropout_rate=0.0)
        >>> memory = np.zeros((2, 12, 8), dtype="float32")
        >>> text = np.zeros((2, 4, 8), dtype="float32")
        >>> hidden, boxes, presence, feats = stack(memory, memory_text=text)
        >>> hidden.shape, boxes.shape, presence.shape
        ((2, 2, 5, 8), (2, 2, 5, 4), (2, 2, 1))
    """

    def __init__(
            self, d_model: int = 256, num_heads: int = 8, num_layers: int = 6,
            num_queries: int = 200, feat_size: Tuple[int, int] = (72, 72),
            dim_feedforward: int = 2048, dropout_rate: float = 0.1,
            activation: str = "relu", use_text_cross_attention: bool = True,
            box_rpb: str = "log", use_presence_token: bool = True,
            clamp_presence_logits: bool = True,
            clamp_presence_logit_max_val: float = 10.0,
            use_normed_output_consistently: bool = True,
            norm_epsilon: float = 1e-5, **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        for name, value in (("d_model", d_model), ("num_heads", num_heads),
                            ("num_layers", num_layers),
                            ("num_queries", num_queries)):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if box_rpb not in ("none", "log", "linear"):
            raise ValueError(f"box_rpb must be 'none', 'log' or 'linear', got "
                             f"{box_rpb!r}")
        if len(feat_size) != 2 or min(feat_size) <= 0:
            raise ValueError(f"feat_size must be a pair of positive ints, got "
                             f"{feat_size}")
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even (the box sine embedding "
                             f"splits it in half), got {d_model}")

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.num_queries = int(num_queries)
        self.feat_size = (int(feat_size[0]), int(feat_size[1]))
        self.dim_feedforward = int(dim_feedforward)
        self.dropout_rate = float(dropout_rate)
        self.activation = str(activation)
        self.use_text_cross_attention = bool(use_text_cross_attention)
        self.box_rpb = str(box_rpb)
        self.use_presence_token = bool(use_presence_token)
        self.clamp_presence_logits = bool(clamp_presence_logits)
        self.clamp_presence_logit_max_val = float(clamp_presence_logit_max_val)
        self.use_normed_output_consistently = bool(
            use_normed_output_consistently)
        self.norm_epsilon = float(norm_epsilon)

        # Every sub-layer store here is FLAT. A `List[List[Layer]]` restores
        # freshly initialized kernels on a `.keras` round trip while the weight
        # count, every weight path and the parameter total all match -- measured
        # in this package, see decisions.md D-098.
        self.decoder_layers = [
            Sam3DecoderLayer(
                d_model=self.d_model, num_heads=self.num_heads,
                dim_feedforward=self.dim_feedforward,
                dropout_rate=self.dropout_rate, activation=self.activation,
                use_text_cross_attention=self.use_text_cross_attention,
                norm_epsilon=self.norm_epsilon, name=f"decoder_layer_{index}")
            for index in range(self.num_layers)
        ]
        self.norm = create_normalization_layer(
            "layer_norm", epsilon=self.norm_epsilon, name="norm")

        # DECISION plan-2026-08-04T044628-4c240b4c/D-112
        # The LAST projection of the box head is ZERO-initialized. Do NOT
        # "fix" this to a standard initializer: the whole refinement chain is
        # `sigmoid(delta + inverse_sigmoid(reference))`, so a zero delta is what
        # makes layer 0 an exact identity on its reference box at step 0. With
        # any non-zero init the first layer displaces every reference box before
        # a single gradient step, and boxRPB's bias -- which is built FROM the
        # reference box -- is displaced with it. There is no shape, dtype or
        # finiteness symptom. See decisions.md D-112.
        self.bbox_embed = self._make_mlp(3, self.d_model, 4, "bbox_embed",
                                         zero_init_last=True)
        self.ref_point_head = self._make_mlp(2, self.d_model, self.d_model,
                                             "ref_point_head")
        self.box_rpb_embed_x = self._make_mlp(2, self.d_model, self.num_heads,
                                              "box_rpb_embed_x")
        self.box_rpb_embed_y = self._make_mlp(2, self.d_model, self.num_heads,
                                              "box_rpb_embed_y")
        self.presence_token_out_norm = create_normalization_layer(
            "layer_norm", epsilon=self.norm_epsilon,
            name="presence_token_out_norm")
        self.presence_token_head = self._make_mlp(3, self.d_model, 1,
                                                  "presence_token_head")

        self.query_embed = None
        self.reference_points = None
        self.presence_token = None

        logger.info(
            f"Sam3TransformerDecoder: d_model={self.d_model}, "
            f"layers={self.num_layers}, queries={self.num_queries}, "
            f"feat_size={self.feat_size}, box_rpb={self.box_rpb}, "
            f"presence={self.use_presence_token}"
        )

    # -----------------------------------------------------------------
    # pure helpers
    #
    # These are METHODS, not module-level functions. `decoder.py` already ships
    # exactly two module-private functions (`_box_rpb_log_compress` and
    # `_box_rpb_bias`), and decisions.md D-109 records that a third is a
    # stop-and-surface trigger. Both helpers below have exactly ONE owner --
    # this class -- so they live on it, matching `Sam3DecoderLayer._with_pos`.
    # The boxRPB pair stayed at module level because they have TWO owners.
    # -----------------------------------------------------------------

    @staticmethod
    def _make_mlp(depth: int, hidden: int, out_dim: int, name: str,
                  zero_init_last: bool = False) -> list:
        """Build a flat ReLU MLP stack; only the last projection is linear."""
        stack = [layers.Dense(hidden, activation="relu", name=f"{name}_{index}")
                 for index in range(depth - 1)]
        extra = dict(kernel_initializer="zeros", bias_initializer="zeros") \
            if zero_init_last else {}
        stack.append(layers.Dense(out_dim, name=f"{name}_{depth - 1}", **extra))
        return stack

    @staticmethod
    def _build_mlp(stack: list, input_shape: Tuple) -> None:
        """Build a flat MLP stack, threading the running shape through it."""
        shape = tuple(input_shape)
        for dense in stack:
            dense.build(shape)
            shape = shape[:-1] + (dense.units,)

    @staticmethod
    def _run_mlp(stack: list, x: Any) -> Any:
        """Apply a flat MLP stack in order."""
        for dense in stack:
            x = dense(x)
        return x

    @staticmethod
    def _inverse_sigmoid(x: Any, eps: float = 1e-3) -> Any:
        """The reference's numerically guarded logit function."""
        x = ops.clip(x, 0.0, 1.0)
        return ops.log(ops.maximum(x, eps) / ops.maximum(1.0 - x, eps))

    @staticmethod
    def _sine_embed_for_boxes(boxes: Any, d_model: int) -> Any:
        """Sine-embed a ``cxcywh`` box into ``2 * d_model`` features.

        Each of the four box scalars is embedded into ``d_model // 2``
        features, interleaved ``sin`` at even channels and ``cos`` at odd, and
        the four are concatenated in the reference's order -- ``y, x, w, h``,
        NOT the ``cxcywh`` order the box itself is stored in.

        :param boxes: ``(batch, num_queries, 4)`` normalized ``cxcywh``.
        :type boxes: Any
        :param d_model: Model width; must be even.
        :type d_model: int
        :return: ``(batch, num_queries, 2 * d_model)``.
        :rtype: Any
        """
        num_feats = d_model // 2
        index = ops.cast(ops.arange(num_feats // 2), boxes.dtype)
        dim_t = ops.power(10000.0, (2.0 * index) / float(num_feats))
        parts = []
        for axis in (1, 0, 2, 3):
            scaled = boxes[..., axis:axis + 1] * (2.0 * math.pi) / dim_t
            pair = ops.stack([ops.sin(scaled), ops.cos(scaled)], axis=-1)
            shape = ops.shape(pair)
            parts.append(ops.reshape(pair, (shape[0], shape[1], num_feats)))
        return ops.concatenate(parts, axis=-1)

    def build(
            self, memory_shape: Tuple[Optional[int], ...],
            memory_text_shape: Optional[Tuple] = None,
            **kwargs: Any,
    ) -> None:
        """Create the stack's own weights and build every sub-layer.

        :param memory_shape: ``(batch, height * width, d_model)``.
        :type memory_shape: Tuple[Optional[int], ...]
        :param memory_text_shape: ``(batch, num_tokens, d_model)``; required
            when text cross-attention is enabled.
        :type memory_text_shape: Optional[Tuple[Optional[int], ...]]
        :param kwargs: Ignored; accepted so the layer builds from its full call
            signature.
        :raises ValueError: On a wrong rank, a width other than ``d_model``, or
            a key count that is not ``feat_size[0] * feat_size[1]``.
        """
        if self.built:
            return
        memory_shape = tuple(memory_shape)
        if len(memory_shape) != 3:
            raise ValueError(f"memory must have shape (batch, keys, d_model), "
                             f"got {memory_shape}")
        if memory_shape[-1] is not None and memory_shape[-1] != self.d_model:
            raise ValueError(f"memory width {memory_shape[-1]} != d_model "
                             f"{self.d_model}")
        keys = self.feat_size[0] * self.feat_size[1]
        if memory_shape[1] is not None and memory_shape[1] != keys:
            raise ValueError(
                f"memory has {memory_shape[1]} keys but feat_size "
                f"{self.feat_size} implies {keys}; boxRPB's bias is built on "
                f"that grid, so a mismatch is a silent wrong-geometry bias")

        normal = keras.initializers.RandomNormal(stddev=1.0)
        self.query_embed = self.add_weight(
            name="query_embed", shape=(self.num_queries, self.d_model),
            initializer=normal, trainable=True)
        self.reference_points = self.add_weight(
            name="reference_points", shape=(self.num_queries, 4),
            initializer=normal, trainable=True)
        if self.use_presence_token:
            # DECISION plan-2026-08-04T044628-4c240b4c/D-137
            # GLOROT, not the `normal` above, and the asymmetry is deliberate.
            # The reference builds all three of these as `nn.Embedding` (unit-
            # normal), and then `TransformerWrapper._reset_parameters` xavier-
            # uniform-initializes every `dim > 1` parameter EXCEPT names holding
            # `box_embed` / `query_embed` / `reference_points`. `query_embed`
            # and `reference_points` are on that exclusion list and keep N(0,1);
            # `presence_token` is NOT, so the reference ships it at xavier scale
            # over its `(1, d_model)` weight -- limit `sqrt(6/(d_model+1))`,
            # std 0.0882 at d_model=256 versus 1.0 here, an 11.3x divergence on
            # the model's ONLY presence signal. Keras `GlorotUniform` computes
            # the identical fans for this shape. Do NOT "tidy" this back to
            # `normal` for symmetry with its two neighbours.
            # See decisions.md D-137.
            self.presence_token = self.add_weight(
                name="presence_token", shape=(1, self.d_model),
                initializer=keras.initializers.GlorotUniform(), trainable=True)

        batch = memory_shape[0]
        tgt_shape = (batch, self.num_queries, self.d_model)
        for decoder_layer in self.decoder_layers:
            decoder_layer.build(tgt_shape, memory_shape, memory_text_shape)
        self.norm.build(tgt_shape)
        self._build_mlp(self.bbox_embed, tgt_shape)
        self._build_mlp(self.ref_point_head,
                        (batch, self.num_queries, 2 * self.d_model))
        for stack in (self.box_rpb_embed_x, self.box_rpb_embed_y):
            self._build_mlp(stack, (batch, self.num_queries, None, 2))
        if self.use_presence_token:
            presence_shape = (batch, 1, self.d_model)
            self.presence_token_out_norm.build(presence_shape)
            self._build_mlp(self.presence_token_head, presence_shape)
        super().build(memory_shape)

    def call(
            self, memory: Any,
            memory_text: Optional[Any] = None,
            text_padding_mask: Optional[Any] = None,
            memory_pos: Optional[Any] = None,
            tgt: Optional[Any] = None,
            reference_boxes: Optional[Any] = None,
            training: Optional[bool] = None,
    ) -> Tuple[Any, Any, Optional[Any], Optional[Any]]:
        """Run the layer stack with per-layer box refinement.

        :param memory: Image memory ``(batch, height * width, d_model)``.
        :type memory: Any
        :param memory_text: Text memory ``(batch, num_tokens, d_model)``.
        :type memory_text: Optional[Any]
        :param text_padding_mask: ``(batch, num_tokens)``, ``True`` at PADDING.
        :type text_padding_mask: Optional[Any]
        :param memory_pos: Per-key positional embedding, shaped like ``memory``.
        :type memory_pos: Optional[Any]
        :param tgt: Object queries ``(batch, num_queries, d_model)``. Defaults
            to this stack's own learned query embedding.
        :type tgt: Optional[Any]
        :param reference_boxes: Initial ``cxcywh`` boxes already in ``[0, 1]``,
            ``(batch, num_queries, 4)``. Defaults to
            ``sigmoid`` of this stack's learned reference points.
        :type reference_boxes: Optional[Any]
        :param training: Training-mode flag.
        :type training: Optional[bool]
        :return: ``(hidden_states, reference_boxes, presence_logits,
            presence_features)`` with shapes ``(num_layers, batch, num_queries,
            d_model)``, ``(num_layers, batch, num_queries, 4)``,
            ``(num_layers, batch, 1)`` and ``(batch, 1, d_model)``. The last two
            are ``None`` when the presence token is disabled.
        :rtype: Tuple[Any, Any, Optional[Any], Optional[Any]]
        """
        batch = ops.shape(memory)[0]
        if tgt is None:
            tgt = ops.broadcast_to(
                ops.expand_dims(ops.cast(self.query_embed, self.compute_dtype),
                                0),
                (batch, self.num_queries, self.d_model))
        if reference_boxes is None:
            reference_boxes = ops.sigmoid(ops.broadcast_to(
                ops.expand_dims(
                    ops.cast(self.reference_points, self.compute_dtype), 0),
                (batch, self.num_queries, 4)))
        else:
            reference_boxes = ops.cast(reference_boxes, self.compute_dtype)
        presence = None
        if self.use_presence_token:
            presence = ops.broadcast_to(
                ops.expand_dims(
                    ops.cast(self.presence_token, self.compute_dtype), 0),
                (batch, 1, self.d_model))

        hidden_states = []
        # The FIRST entry is the INITIAL reference, and the LAST layer's
        # refinement is deliberately NOT appended -- the stack therefore holds
        # exactly `num_layers` boxes, each one the reference the layer at that
        # index actually consumed. An off-by-one here misaligns every auxiliary
        # box loss with the layer that produced it, with no shape symptom.
        all_reference_boxes = [reference_boxes]
        presence_logits = []
        output = tgt

        for index, decoder_layer in enumerate(self.decoder_layers):
            # The conditional query position and boxRPB's bias are BOTH rebuilt
            # from the CURRENT reference box at every layer. Hoisting either one
            # out of the loop turns a conditional-DETR decoder into a static one.
            query_pos = self._run_mlp(
                self.ref_point_head,
                self._sine_embed_for_boxes(reference_boxes, self.d_model))
            image_cross_bias = None
            if self.box_rpb != "none":
                image_cross_bias = _box_rpb_bias(
                    reference_boxes, self.feat_size,
                    lambda t: self._run_mlp(self.box_rpb_embed_x, t),
                    lambda t: self._run_mlp(self.box_rpb_embed_y, t),
                    self.num_heads, self.box_rpb)

            output, presence = decoder_layer(
                output, memory, query_pos=query_pos, memory_pos=memory_pos,
                memory_text=memory_text, text_padding_mask=text_padding_mask,
                image_cross_bias=image_cross_bias, presence_token=presence,
                training=training)

            # DECISION plan-2026-08-04T044628-4c240b4c/D-113
            # The delta reads the NORMED hidden state and the next reference is
            # DETACHED. Do NOT "simplify" either half:
            #   * feeding the raw `output` to the box head is a silent value
            #     defect -- same shapes, same finiteness, different boxes;
            #   * removing `stop_gradient` re-opens a gradient path from every
            #     later layer back through the reference chain into every
            #     earlier layer's box head, which is the multi-layer credit
            #     assignment iterative box refinement exists to avoid.
            # See decisions.md D-113.
            normed = self.norm(output)
            delta = self._run_mlp(
                self.bbox_embed,
                normed if self.use_normed_output_consistently else output)
            refined = ops.sigmoid(
                delta + self._inverse_sigmoid(reference_boxes))
            reference_boxes = ops.stop_gradient(refined)
            if index != self.num_layers - 1:
                all_reference_boxes.append(refined)
            hidden_states.append(normed)

            if presence is not None:
                logits = ops.squeeze(
                    self._run_mlp(self.presence_token_head,
                                  self.presence_token_out_norm(presence)),
                    axis=-1)
                # DECISION plan-2026-08-04T044628-4c240b4c/D-111
                # This bound is 10.0 and the open-vocabulary scorer's is 12.0.
                # They are NOT the same quantity and must not be unified; only
                # a probe inside (10, 12] can tell them apart.
                # A DELIBERATE DIVERGENCE, recorded rather than hidden: at the
                # pinned reference SHA this clamp is a provable NO-OP -- it
                # calls the out-of-place `clamp` and discards the result, then
                # appends the UNCLAMPED tensor. Both constructor parameters,
                # the reference's own comment, and its scorer (which clamps
                # correctly) say the intent is a live clamp, so this port makes
                # it effective. Do NOT "restore parity" by deleting it: that
                # would also make `clamp_presence_logits=False` unreachable by
                # any test. See decisions.md D-111.
                if self.clamp_presence_logits:
                    logits = ops.clip(logits,
                                      -self.clamp_presence_logit_max_val,
                                      self.clamp_presence_logit_max_val)
                presence_logits.append(logits)

        return (ops.stack(hidden_states), ops.stack(all_reference_boxes),
                ops.stack(presence_logits) if presence_logits else None,
                presence)

    def compute_output_shape(
            self, memory_shape: Tuple[Optional[int], ...],
            memory_text_shape: Optional[Tuple] = None,
            **kwargs: Any,
    ) -> Tuple:
        """Return the four output shapes, derived from the stored config.

        :param memory_shape: ``(batch, keys, d_model)``.
        :type memory_shape: Tuple[Optional[int], ...]
        :param memory_text_shape: Unused.
        :type memory_text_shape: Optional[Tuple[Optional[int], ...]]
        :param kwargs: Ignored.
        :return: Shapes of ``(hidden_states, reference_boxes, presence_logits,
            presence_features)``.
        :rtype: Tuple
        """
        batch = tuple(memory_shape)[0]
        return (
            (self.num_layers, batch, self.num_queries, self.d_model),
            (self.num_layers, batch, self.num_queries, 4),
            (self.num_layers, batch, 1) if self.use_presence_token else None,
            (batch, 1, self.d_model) if self.use_presence_token else None,
        )

    def get_config(self) -> Dict[str, Any]:
        """Return every ``__init__`` parameter.

        :return: Serializable configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "d_model": self.d_model, "num_heads": self.num_heads,
            "num_layers": self.num_layers, "num_queries": self.num_queries,
            "feat_size": self.feat_size,
            "dim_feedforward": self.dim_feedforward,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "use_text_cross_attention": self.use_text_cross_attention,
            "box_rpb": self.box_rpb,
            "use_presence_token": self.use_presence_token,
            "clamp_presence_logits": self.clamp_presence_logits,
            "clamp_presence_logit_max_val": self.clamp_presence_logit_max_val,
            "use_normed_output_consistently":
                self.use_normed_output_consistently,
            "norm_epsilon": self.norm_epsilon,
        })
        return config
