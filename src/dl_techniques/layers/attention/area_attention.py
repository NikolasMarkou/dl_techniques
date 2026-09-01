"""
Area attention over 2D feature maps, as used by the YOLOv12 detector.

Standard self-attention over a ``(B, H, W, C)`` feature map is quadratic in the
number of pixels: flattening a ``H x W`` map gives ``H*W`` tokens and an
``(H*W, H*W)`` score matrix. Area attention keeps the same mathematics but
partitions the flattened token sequence into ``area`` contiguous groups and lets
each group attend only within itself, which reduces the score matrix to ``area``
independent ``(H*W/area, H*W/area)`` blocks. ``area=1`` recovers plain global
attention over the whole map, so one layer expresses both regimes.

This module is the relocated home of the ``AreaAttention`` that used to live in
``dl_techniques.layers.yolo12_blocks``. The relocation brought the layer up to
``layers/attention/GUIDE.md`` compliance; the additions are all off by default and
the layer is numerically identical to its pre-move self at default arguments.

Architecture:
    A self-contained attention block over a 4D ``(B, H, W, C)`` feature map. It is
    **not** a bare residual add-on: it carries its own positional-encoding branch
    and its own output projection.

    1.  **Two 1x1 projections.** ``qk_conv`` emits ``2 * dim`` channels (query and
        key concatenated, split on the feature axis afterwards) and ``v_conv``
        emits ``dim``. Both are :class:`~dl_techniques.layers.standard_blocks.ConvBlock`
        instances with ``activation_type='linear'``, i.e. Conv2D + normalization
        and an exact weightless identity in place of an activation.

    2.  **Depthwise positional encoding.** ``pe_conv`` is a ``5x5`` depthwise
        (``groups=dim``) ``ConvBlock`` applied to ``v``. Its output is added to the
        attention result rather than to the input, which is what makes the layer's
        positional signal value-conditioned.

    3.  **Grouping.** The projections are reshaped from ``(B, H, W, ·)`` to
        ``(B, H*W, ·)``. When ``area > 1`` **and** ``H*W`` is divisible by ``area``,
        the sequence is reshaped to ``(B, area, H*W/area, ·)`` and attention runs
        inside each group. Otherwise the layer silently falls back to global
        attention over the full sequence. That fallback is deliberate and
        load-bearing: yolo12 feeds it non-divisible spatial extents.

    4.  **Scoring.** Multi-head scaled dot product with the scale supplied by
        :func:`~dl_techniques.layers.attention.common.compute_attention_scale`
        (computed once in ``__init__``, never in ``call``). Optional QK
        normalization (``qk_norm_type``) is applied per head; an optional
        ``attention_mask`` is applied through
        :func:`~dl_techniques.layers.attention.common.apply_attention_mask`;
        :class:`~dl_techniques.layers.activations.ProbabilityOutput` converts scores
        to weights, followed by optional attention dropout.

    5.  **Aggregation and output projection.** The weights aggregate the values,
        the result is folded back to ``(B, H, W, dim)``, the positional encoding is
        added, and ``proj_conv`` (a 1x1 ``ConvBlock``) produces the output.

Relocation notes:
    * The normalization used by the four ``ConvBlock`` sub-layers is **not**
      hardcoded here. It arrives as data through ``normalization_kwargs``. yolo12's
      D-067 epsilon/momentum pair keeps exactly one home,
      ``dl_techniques.layers.yolo12_blocks.YOLO12_NORM_KWARGS``, and this package
      must not import it: ``layers/attention`` sits *below* ``layers/yolo12_blocks``
      in the dependency order. Passing ``normalization_kwargs=None`` therefore
      yields the normalization factory's own defaults, not yolo12's.
    * ``use_bias`` defaults to ``False``, matching the convolution convention of the
      pre-move yolo12 ``ConvBlock`` rather than
      :class:`~dl_techniques.layers.standard_blocks.ConvBlock`'s ``True``.
    * The layer keeps ``channels``-free naming (``dim``, ``num_heads``,
      ``dropout_rate``) per GUIDE.md section 2; it was already conformant.
"""

import keras
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.standard_blocks import ConvBlock
from dl_techniques.layers.activations import ProbabilityOutput
from dl_techniques.utils.keras_registration import register_dl_technique

from .common import (
    apply_attention_mask,
    compute_attention_scale,
    validate_head_divisibility,
)

# ---------------------------------------------------------------------


# DECISION plan-2026-09-01T055648-e6d380a5/D-002
# `legacy_alias=False` is REQUIRED, not stylistic. The bare legacy key
# `Custom>AreaAttention` is module-independent, and while the pre-move
# `yolo12_blocks.AreaAttention` still exists (it is deleted later in the same plan)
# both classes would claim it — `register_dl_technique` raises `AliasCollisionError`
# at import time, so the whole package fails to import. Do NOT "fix" that by dropping
# the flag once the old class is gone either: D-002 is a CLEAN BREAK on serialization
# keys — no `legacy_packages=`, no alias preservation, and no bare `Custom>` key for a
# class whose canonical key is its module path. See decisions.md D-002.
@register_dl_technique(
    "dl_techniques.layers.attention.area_attention", legacy_alias=False
)
class AreaAttention(keras.layers.Layer):
    """
    Area attention over a 4D ``(batch, height, width, channels)`` feature map.

    Multi-head self-attention that runs either globally (``area=1``) or within
    ``area`` contiguous groups of the flattened token sequence. Includes a
    depthwise positional-encoding branch and an output projection.

    **Architecture Overview:**

    .. code-block:: text

        ┌───────────────────────────────────┐
        │  Input [B, H, W, C]               │
        └──────────────┬────────────────────┘
                       ▼
        ┌───────────────────────────────────┐
        │  QK = ConvBlock 1x1 → [B,H,W,2D]  │
        │  V  = ConvBlock 1x1 → [B,H,W,D]   │
        │  PE = DW ConvBlock 5x5 on V       │
        └──────────────┬────────────────────┘
                       ▼
        ┌───────────────────────────────────┐
        │  Reshape to [B, areas, seq, D]    │
        │  optional QK-norm                 │
        │  scores = Q·Kᵀ * scale            │
        │  optional attention_mask          │
        │  ProbabilityOutput → dropout      │
        │  attention within each area       │
        └──────────────┬────────────────────┘
                       ▼
        ┌───────────────────────────────────┐
        │  Add PE + projection ConvBlock    │
        │  Output [B, H, W, dim]            │
        └───────────────────────────────────┘

    **Equivalence contract.** At default arguments — ``dropout_rate=0.0``,
    ``qk_norm_type=None``, ``probability_type='softmax'``, ``attention_mask=None`` —
    this layer reproduces the pre-move ``yolo12_blocks.AreaAttention`` bit-for-bit on
    identical weights, provided ``normalization_kwargs`` carries the same
    normalization configuration. That claim is a test, not a comment:
    ``tests/test_layers/test_the_yolo12_relocation_is_equivalent.py``.

    :param dim: Number of feature dimensions. Must be positive and divisible by
        ``num_heads``.
    :type dim: int
    :param num_heads: Number of attention heads. Defaults to 8.
    :type num_heads: int
    :param area: Number of attention groups; ``1`` means global attention. When
        ``area > 1`` but the flattened sequence length is not divisible by it, the
        layer falls back to global attention. Defaults to 1.
    :type area: int
    :param dropout_rate: Dropout applied to the attention weights. ``0.0``
        disables it. Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias: Whether the four ``ConvBlock`` convolutions carry a bias term.
        Defaults to ``False``.
    :type use_bias: bool
    :param normalization_type: Normalization type used by the four ``ConvBlock``
        sub-layers. Defaults to ``'batch_norm'``.
    :type normalization_type: str
    :param normalization_kwargs: Extra arguments forwarded to the normalization
        factory by every ``ConvBlock`` sub-layer. ``None`` means the factory's own
        defaults. Callers that need a specific epsilon/momentum pair supply it here.
    :type normalization_kwargs: Optional[Dict[str, Any]]
    :param probability_type: Score-normalization strategy handed to
        :class:`ProbabilityOutput`. Defaults to ``'softmax'``.
    :type probability_type: str
    :param probability_config: Configuration for the score-normalization strategy,
        e.g. ``{'axis': -1}``. ``None`` means the strategy's defaults.
    :type probability_config: Optional[Dict[str, Any]]
    :param qk_norm_type: Optional normalization applied per head to the query and
        key before scoring. ``None`` disables QK-norm and adds no weights.
    :type qk_norm_type: Optional[str]
    :param qk_norm_kwargs: Extra arguments for the QK normalization layers.
    :type qk_norm_kwargs: Optional[Dict[str, Any]]
    :param kernel_initializer: Weight initializer for the four convolutions.
        Defaults to ``'he_normal'``.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any

    :raises ValueError: If ``dim``, ``num_heads`` or ``area`` is not positive, if
        ``dim`` is not divisible by ``num_heads``, or if ``dropout_rate`` is outside
        ``[0, 1]``.

    Example:
        >>> import keras, numpy as np
        >>> layer = AreaAttention(dim=64, num_heads=8, area=4)
        >>> y = layer(np.zeros((2, 8, 8, 64), dtype="float32"))
        >>> y.shape
        (2, 8, 8, 64)
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            area: int = 1,
            dropout_rate: float = 0.0,
            use_bias: bool = False,
            normalization_type: str = "batch_norm",
            normalization_kwargs: Optional[Dict[str, Any]] = None,
            probability_type: str = "softmax",
            probability_config: Optional[Dict[str, Any]] = None,
            qk_norm_type: Optional[str] = None,
            qk_norm_kwargs: Optional[Dict[str, Any]] = None,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            **kwargs: Any
    ) -> None:
        """Validate the configuration and create every sub-layer.

        This layer owns no weights of its own; every weight belongs to a sub-layer.
        See the class docstring for the parameter reference.
        """
        super().__init__(**kwargs)

        # Validate inputs. The head-split check is delegated to the package's shared
        # validator; the message it emits already names `dim` and `num_heads`, which
        # are this layer's own argument spellings, so no `*_name` override is needed.
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        validate_head_divisibility(dim, num_heads)
        if area <= 0:
            raise ValueError(f"area must be positive, got {area}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")

        # Store ALL configuration parameters
        self.dim = dim
        self.num_heads = num_heads
        self.area = area
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.normalization_type = normalization_type
        self.normalization_kwargs = normalization_kwargs
        self.probability_type = probability_type
        self.probability_config = probability_config
        self.qk_norm_type = qk_norm_type
        self.qk_norm_kwargs = qk_norm_kwargs
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.head_dim = dim // num_heads

        # DECISION plan-2026-09-01T055648-e6d380a5/D-003
        # The scale is a plain Python float computed ONCE here. Do NOT move it back
        # into `call()` as `ops.cast(1.0 / ops.sqrt(ops.cast(head_dim, "float32")),
        # q.dtype)` — that was the pre-move spelling and it adds a live op to every
        # forward pass for a constant. MEASURED at the relocation (probed head_dim 2
        # and 16, this layer's own probe grid): the two spellings agree to the bit
        # once rounded to float32, so the swap is exactly behaviour-preserving.
        # See decisions.md D-003 and `common.compute_attention_scale`.
        self.scale = compute_attention_scale(self.head_dim)

        # CREATE all sub-layers in __init__ (they are unbuilt).
        #
        # The four convolutions are created FIRST and in this order — qk, v, pe,
        # proj. They are the only weight-bearing sub-layers at default arguments, and
        # the relocation's equivalence harness transfers weights by ordered
        # `set_weights`. Reordering them is a silent weight-permutation bug.
        _conv_kwargs = dict(
            activation_type="linear",
            normalization_type=self.normalization_type,
            normalization_kwargs=self.normalization_kwargs,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
        )

        # Query-Key projection
        self.qk_conv = ConvBlock(
            filters=self.dim * 2,
            kernel_size=1,
            name="qk",
            **_conv_kwargs
        )

        # Value projection
        self.v_conv = ConvBlock(
            filters=self.dim,
            kernel_size=1,
            name="v",
            **_conv_kwargs
        )

        # Position encoding (depthwise: one group per channel)
        self.pe_conv = ConvBlock(
            filters=self.dim,
            kernel_size=5,
            padding="same",
            groups=self.dim,
            name="pe",
            **_conv_kwargs
        )

        # Output projection
        self.proj_conv = ConvBlock(
            filters=self.dim,
            kernel_size=1,
            name="proj",
            **_conv_kwargs
        )

        # Score normalization. GUIDE.md section 3.1: never a hardcoded softmax.
        # At the default `probability_type='softmax'` this wraps
        # `keras.layers.Softmax(axis=-1)`, which is weightless and bit-identical to
        # the `keras.ops.nn.softmax(scores, axis=-1)` it replaced (measured).
        self.attn_prob = ProbabilityOutput(
            probability_type=self.probability_type,
            type_config=self.probability_config,
            name="attn_prob",
        )

        # Attention dropout. Weightless, and an exact no-op at rate 0.0.
        self.attn_dropout = keras.layers.Dropout(
            self.dropout_rate, name="attn_dropout"
        )

        # Optional per-head QK normalization. `None` creates nothing, so the default
        # configuration adds no weights and cannot perturb the weight ordering.
        if self.qk_norm_type is not None:
            self.q_norm = create_normalization_layer(
                self.qk_norm_type, name="q_norm", **(self.qk_norm_kwargs or {})
            )
            self.k_norm = create_normalization_layer(
                self.qk_norm_type, name="k_norm", **(self.qk_norm_kwargs or {})
            )
        else:
            self.q_norm = None
            self.k_norm = None

        logger.debug(
            f"AreaAttention initialized: dim={dim}, num_heads={num_heads}, "
            f"area={area}, head_dim={self.head_dim}"
        )

    def _score_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Best-effort static shape of the score tensor, for building sub-layers.

        Mirrors the branch taken by :meth:`call`: when the flattened sequence length
        is statically known and divisible by ``area``, the grouped branch is
        assumed; otherwise the global branch. When ``height`` or ``width`` is
        dynamic the sequence extents are reported as ``None``, which every
        weightless sub-layer built here accepts.

        :param input_shape: Shape tuple of the layer input, ``(B, H, W, C)``.
        :type input_shape: tuple
        :return: ``(batch, num_areas, num_heads, attended, attended)``.
        :rtype: tuple
        """
        height, width = input_shape[1], input_shape[2]
        if height is None or width is None:
            num_areas, attended = None, None
        else:
            seq_len = height * width
            if self.area > 1 and seq_len % self.area == 0:
                num_areas, attended = self.area, seq_len // self.area
            else:
                num_areas, attended = 1, seq_len
        return (input_shape[0], num_areas, self.num_heads, attended, attended)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build every sub-layer explicitly, in computational order.

        :param input_shape: Shape tuple of the input tensor, ``(B, H, W, C)``.
        :type input_shape: tuple
        """
        self.qk_conv.build(input_shape)
        self.v_conv.build(input_shape)

        # `pe_conv` and `proj_conv` operate on tensors with `dim` channels, not on
        # the raw input — build them with the correct intermediate shape.
        v_output_shape = self.v_conv.compute_output_shape(input_shape)
        self.pe_conv.build(v_output_shape)
        self.proj_conv.build(v_output_shape)

        score_shape = self._score_shape(input_shape)
        self.attn_prob.build(score_shape)
        self.attn_dropout.build(score_shape)

        if self.q_norm is not None:
            qk_shape = score_shape[:-1] + (self.head_dim,)
            self.q_norm.build(qk_shape)
            self.k_norm.build(qk_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through area attention.

        :param inputs: Input tensor of shape ``(batch, height, width, channels)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional **keep** mask over spatial positions, shaped
            ``(batch, height, width)`` or ``(batch, height * width)``. Nonzero means
            "attend here"; this layer's convention is ``1 = keep``, and the predicate
            is forwarded verbatim to
            :func:`~dl_techniques.layers.attention.common.apply_attention_mask`,
            which never infers polarity. ``None`` disables masking entirely — no mask
            op is added to the graph.
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Whether the layer runs in training mode.
        :type training: Optional[bool]

        :return: Output tensor of shape ``(batch, height, width, dim)``.
        :rtype: keras.KerasTensor
        """
        batch_size = keras.ops.shape(inputs)[0]
        height = keras.ops.shape(inputs)[1]
        width = keras.ops.shape(inputs)[2]

        # Generate query-key and value projections
        qk = self.qk_conv(inputs, training=training)
        v = self.v_conv(inputs, training=training)

        # Position encoding
        pe = self.pe_conv(v, training=training)

        # Reshape for attention computation
        seq_len = height * width
        qk = keras.ops.reshape(qk, (batch_size, seq_len, self.dim * 2))
        v = keras.ops.reshape(v, (batch_size, seq_len, self.dim))

        # Split query and key
        q, k = keras.ops.split(qk, 2, axis=-1)

        keep = None
        if attention_mask is not None:
            keep = keras.ops.reshape(attention_mask, (batch_size, seq_len))

        # The area branch is taken only when the sequence divides evenly. The
        # `else` fallback is a real, exercised branch of this layer, not an
        # oversight: yolo12 feeds spatial extents that do not divide by `area`.
        if self.area > 1 and seq_len % self.area == 0:
            area_size = seq_len // self.area
            q = keras.ops.reshape(q, (batch_size, self.area, area_size, self.dim))
            k = keras.ops.reshape(k, (batch_size, self.area, area_size, self.dim))
            v = keras.ops.reshape(v, (batch_size, self.area, area_size, self.dim))
            if keep is not None:
                # `(B, area, 1, 1, area_size)` — broadcast over the head and query
                # axes, exact along the key axis the softmax reduces over.
                keep = keras.ops.reshape(
                    keep, (batch_size, self.area, 1, 1, area_size)
                )

            attn_output = self._compute_attention(
                q, k, v, keep=keep, training=training
            )
            attn_output = keras.ops.reshape(
                attn_output, (batch_size, seq_len, self.dim)
            )
        else:
            if keep is not None:
                keep = keras.ops.reshape(keep, (batch_size, 1, 1, 1, seq_len))
            attn_output = self._compute_attention(
                keras.ops.expand_dims(q, 1),
                keras.ops.expand_dims(k, 1),
                keras.ops.expand_dims(v, 1),
                keep=keep,
                training=training,
            )
            attn_output = keras.ops.squeeze(attn_output, 1)

        # Reshape back to spatial dimensions
        attn_output = keras.ops.reshape(
            attn_output, (batch_size, height, width, self.dim)
        )

        # Add position encoding and apply final projection
        output = attn_output + pe
        return self.proj_conv(output, training=training)

    def _compute_attention(
            self,
            q: keras.KerasTensor,
            k: keras.KerasTensor,
            v: keras.KerasTensor,
            keep: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Compute multi-head scaled dot-product attention within each area.

        :param q: Query tensor of shape ``(batch, areas, seq, dim)``.
        :type q: keras.KerasTensor
        :param k: Key tensor of shape ``(batch, areas, seq, dim)``.
        :type k: keras.KerasTensor
        :param v: Value tensor of shape ``(batch, areas, seq, dim)``.
        :type v: keras.KerasTensor
        :param keep: Optional keep predicate already broadcast to the score rank,
            ``(batch, areas, 1, 1, seq)``. Nonzero means "attend here".
        :type keep: Optional[keras.KerasTensor]
        :param training: Whether the layer runs in training mode.
        :type training: Optional[bool]

        :return: Attention output of shape ``(batch, areas, seq, dim)``.
        :rtype: keras.KerasTensor
        """
        batch_size = keras.ops.shape(q)[0]
        num_areas = keras.ops.shape(q)[1]
        seq_len = keras.ops.shape(q)[-2]

        shape = (batch_size, num_areas, seq_len, self.num_heads, self.head_dim)
        q = keras.ops.reshape(q, shape)
        k = keras.ops.reshape(k, shape)
        v = keras.ops.reshape(v, shape)

        # Transpose for attention computation: [batch, areas, heads, seq, head_dim]
        q = keras.ops.transpose(q, (0, 1, 3, 2, 4))
        k = keras.ops.transpose(k, (0, 1, 3, 2, 4))
        v = keras.ops.transpose(v, (0, 1, 3, 2, 4))

        if self.q_norm is not None:
            q = self.q_norm(q, training=training)
            k = self.k_norm(k, training=training)

        # Scaled dot-product attention. `self.scale` is the constant computed in
        # `__init__`; it is a Python float and folds into the graph.
        scores = keras.ops.matmul(
            q, keras.ops.transpose(k, (0, 1, 2, 4, 3))
        ) * self.scale

        if keep is not None:
            # DECISION plan-2026-09-01T055648-e6d380a5/D-003
            # `rescue_axis` is DERIVED from this layer's own `probability_config`,
            # never hardcoded to the helper's `-1` default: a caller that moves the
            # softmax axis would otherwise get the degenerate-slice rescue applied
            # along an axis its softmax does not reduce over. Polarity is likewise
            # the caller's: `keep` is forwarded verbatim, because a uniform
            # `mask > 0` rewrite inverts masking at some sites with no exception and
            # no shape error. See GUIDE.md section 3.5 and decisions.md D-003.
            scores = apply_attention_mask(
                scores,
                keep,
                out_dtype=keras.backend.standardize_dtype(scores.dtype),
                rescue_axis=(self.probability_config or {}).get("axis", -1),
            )

        attn_weights = self.attn_prob(scores)
        attn_weights = self.attn_dropout(attn_weights, training=training)

        # Apply attention to values
        attn_output = keras.ops.matmul(attn_weights, v)

        # Reshape and transpose back
        attn_output = keras.ops.transpose(attn_output, (0, 1, 3, 2, 4))
        return keras.ops.reshape(
            attn_output, (batch_size, num_areas, seq_len, self.dim)
        )

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape.

        :param input_shape: Shape tuple of the input.
        :type input_shape: tuple

        :return: Output shape tuple, with the last axis replaced by ``dim``.
        :rtype: tuple
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: Dictionary containing every ``__init__`` argument.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "area": self.area,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "normalization_type": self.normalization_type,
            "normalization_kwargs": self.normalization_kwargs,
            "probability_type": self.probability_type,
            "probability_config": self.probability_config,
            "qk_norm_type": self.qk_norm_type,
            "qk_norm_kwargs": self.qk_norm_kwargs,
            "kernel_initializer": keras.initializers.serialize(
                self.kernel_initializer
            ),
        })
        return config

# ---------------------------------------------------------------------
