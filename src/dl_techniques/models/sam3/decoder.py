"""
SAM 3's DETR decoder layer, with boxRPB and the presence token.

This module provides the single public class :class:`Sam3DecoderLayer` -- ONE
layer of SAM 3's detection decoder. The layer stack that repeats it, refines the
reference boxes and reads out the per-layer presence logits is a separate class
that lands in the next step of the same build.

**Why this is not a stock transformer decoder layer.** A DETR decoder layer has
two attention sub-blocks. This one has THREE, in this order:

.. code-block:: text

    tgt (batch, num_queries, d_model)
      |
      +-- 1. self-attention          q = k = tgt + query_pos, v = tgt
      |      residual, then norm2
      |
      +-- 2. text cross-attention    q = tgt + query_pos, k = v = text memory
      |      residual, then catext_norm
      |
      +-- 3. image cross-attention   q = tgt + query_pos,
      |                              k = image memory + memory_pos,
      |                              v = image memory,
      |                              scores += boxRPB per-head additive bias
      |      residual, then norm1
      |
      +-- 4. feed-forward            fc1 -> relu -> drop -> fc2 -> drop
             residual, then norm3

Three details carry the correctness of this layer:

1. **The image cross-attention takes a real-valued, per-head, per-query
   additive bias (boxRPB) into its RAW scores, before the softmax.** No existing
   attention layer in this repository can carry it -- see the ``D-080`` anchor
   on :class:`_Sam3DecoderAttention` and the measurement recorded there.
2. **Neither the self-attention nor the image cross-attention draws its keys and
   its values from the same tensor.** ``k`` carries a positional embedding that
   ``v`` does not. That single asymmetry is what disqualifies the repository's
   cross-attention layer at those two sites -- and does NOT disqualify it at the
   text site, which is why the text site uses it unmodified.
3. **The presence token rides along.** It is prepended to the query sequence
   with a ZEROED query position, given an all-zero bias row so that it always
   attends everywhere in image cross-attention regardless of any per-query
   boxRPB bias, and split back off after the feed-forward.

boxRPB itself -- the log-compressed, box-conditioned relative position bias --
is built by :func:`_box_rpb_bias` here and OWNED by the layer stack, not by this
layer: the reference shares one pair of embedding MLPs across every decoder
layer, so making them per-layer would multiply their parameters by the layer
count with no shape symptom.

At the settled SAM 3 configuration this layer is ``d_model=256``,
``num_heads=8``, ``dim_feedforward=2048``, ``dropout=0.1``, ``relu``, with text
cross-attention enabled and ``boxRPB="log"``.

References:
    - Ravi, N. et al. (2025). "SAM 3: Segment Anything with Concepts."
    - Carion, N. et al. (2020). "End-to-End Object Detection with Transformers"
      (DETR; the decoder shape this one extends).
    - Liu, S. et al. (2023). "Grounding DINO" (the text cross-attention
      sub-block inserted between self- and image cross-attention).
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
