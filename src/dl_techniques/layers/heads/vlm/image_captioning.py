"""``ImageCaptioningHead``: a transformer decoder over text, cross-attending to vision."""

import keras
from typing import Dict, List, Optional, Union, Tuple, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ...attention.factory import create_attention_layer
from ...ffn.factory import (
    FFN_REGISTRY,
    assemble_ffn_config,
    create_ffn_from_config,
    FFNType,
)
from ...norms.factory import create_normalization_layer
from ....utils.masking import MaskFactory
from .task_types import VLMTaskConfig
from dl_techniques.utils.keras_registration import register_dl_technique
from .utils import _ffn_width_kwargs, _check_task_config_fields


@register_dl_technique("dl_techniques.layers.heads.vlm.image_captioning")
class ImageCaptioningHead(keras.layers.Layer):
    """
    A transformer decoder over text, cross-attending to vision features.

    This head generates a caption one token at a time. Each decoder layer runs
    three sub-blocks in order: causal self-attention over the text stream,
    cross-attention onto the vision features, then an FFN. Every sub-block adds
    its output back onto its input and normalizes afterwards, so this is a
    post-norm decoder with three residual adds per layer.

    It does NOT inherit from :class:`BaseVLMHead` and has no fusion layer. It
    derives from ``keras.layers.Layer`` directly.

    The residual adds carry no projection, so the text feature width must
    already equal ``hidden_dim``. The vision width is independent: the vision
    features only enter as the cross-attention key and value.

    **Architecture Overview:**

    .. code-block:: text

        vision_features            text_features
        (B, N_vis, D_vis)          (B, S, hidden_dim)
               │                          │
               │                          ▼ x
               │              ┌──────────────────────┐
               │              │ self_attention[i]    │
               │              │ causal keep mask     │
               │              └───────────┬──────────┘
               │                          ▼
               │              ┌──────────────────────┐
               │              │ norm[3i](x + attn)   │
               │              └───────────┬──────────┘
               │                          ▼
               │              ┌──────────────────────┐
               └──── kv ────► │ cross_attention[i]   │
                              └───────────┬──────────┘
                                          ▼
                              ┌──────────────────────┐
                              │ norm[3i+1](x + cross)│
                              └───────────┬──────────┘
                                          ▼
                              ┌──────────────────────┐
                              │ ffn_layers[i]        │
                              └───────────┬──────────┘
                                          ▼
                              ┌──────────────────────┐
                              │ norm[3i+2](x + ffn)  │
                              └───────────┬──────────┘
                                          ▼
                                repeat x num_layers
                                          ▼
                              ┌──────────────────────┐
                              │ output_proj Dense    │
                              └───────────┬──────────┘
                              ┌────────┴────────┐
                              ▼                 ▼
                          'logits'      'hidden_states'
                    (B, S, vocab_size)  (B, S, hidden_dim)

    The causal mask comes from ``MaskFactory.create_causal_mask``, not from
    ``keras.ops.tril``. See the note in ``call``.

    Input shape:
        ``{'vision_features': (batch, num_vision, vision_width),
        'text_features': (batch, seq_len, hidden_dim)}``, plus the optional
        keep masks ``'vision_mask': (batch, num_vision)`` and
        ``'text_mask': (batch, seq_len)``. The vision mask is the
        cross-attention's key mask. The text mask is combined with the causal
        mask in each self-attention, so a real token never attends to a
        padded one, which matters for left padding; the outputs at padded
        text positions are not meaningful and belong out of the loss.

    Output shape:
        ``{'logits': (batch, seq_len, vocab_size), 'hidden_states':
        (batch, seq_len, hidden_dim)}``.

    :param task_config: Configuration object for the task.
        ``vocab_size`` is required; ``num_classes`` must be unset.
    :type task_config: VLMTaskConfig
    :param num_layers: Number of decoder layers.
    :type num_layers: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param ffn_type: Type of feed-forward network in decoder blocks.

        **Not every registry type is usable, and this is the ONE place that says
        which.** Re-derive the list rather than trusting it. Run this::

            CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
              tests/test_layers/test_heads/test_vlm.py \
              -k "OutputWidthParamRouting" -v

        Both FFN construction sites supply ``hidden_dim`` conditionally from
        ``FFN_REGISTRY`` (D-008/D-020). They supply the OUTPUT width by the
        type's own parameter name, through ``_ffn_width_kwargs`` (D-014). The
        two sites are this head's per-layer loop and
        ``ImageTextMatchingHead._create_post_fusion_layers``. What that closed, and what remains
        closed and why:

        * ``kan`` and ``power_mlp`` -- newly usable on BOTH sites. Their width
          parameter was simply named something else (``features`` / ``units``)
          and was the only thing they were missing.
        * ``tversky`` -- still closed on BOTH sites, for two independent
          reasons. It also requires ``num_features``, a feature-bank size with
          no source in ``VLMTaskConfig``. And ``TverskyProjectionLayer.build()``
          hard-raises on anything but rank-2, which this head's rank-3 decoder
          stream can never satisfy.
        * ``gated_mlp`` -- a permanent capability gap. ``GatedMLP`` is 1x1-conv
          based and needs a rank-4 ``(B, H, W, C)`` input; this head passes
          rank-3 and ``BaseVLMHead`` rank-2. Its ``filters`` width parameter now
          reaches it, which is exactly why the failure moved from a missing
          parameter to a kernel-rank error.
        * ``counting``, ``logic``, ``mixer`` -- closed by a hyperparameter
          ``VLMTaskConfig`` does not carry and cannot derive: ``count_dim``,
          ``logic_dim``, and ``tokens_mlp_dim`` + ``channels_mlp_dim``
          respectively. Inventing defaults for those is a modelling decision,
          not a lookup, so it was not done. ``mixer`` also has no output-width
          concept at all. Its output shape is its input shape, so its
          ``output_dim_param`` is ``None``.

        The superseded claim, recorded so it is not re-derived: this docstring
        used to say the failures were caused by ``output_dim`` being hardcoded.
        Execution disproved that. ``output_dim`` was either accepted and
        present, or dropped by the factory's parameter filter, which raises as
        of D-023. Every failing type died earlier inside ``validate_ffn_config``
        on a DIFFERENT required key.

        Related, and also measured rather than assumed: several combinations
        once built and then failed inside ``call()``. Each is now refused in
        ``build()``, by name (D-011, D-056): ``VisualGroundingHead`` with
        ``fusion_strategy="attention_pooling"``, ``ImageCaptioningHead`` with a
        text width other than ``hidden_dim`` or a 2-D vision or text input,
        and ``VQAHead`` attention pooling with a width other than
        ``hidden_dim``. ``VQAHead`` ``"mean"``/``"max"`` on 2-D inputs, the
        last of them, now works: a 2-D input is taken as already pooled.
    :type ffn_type: FFNType
    :param ffn_expansion_factor: Width multiplier for the decoder FFN's hidden
        layer. Defaults to 4.

        Read by every FFN type that can use it, via one of two channels. Types
        that REQUIRE an explicit ``hidden_dim`` receive
        ``hidden_dim * ffn_expansion_factor``. Types that derive their own width
        receive the factor itself and apply their own rule to it. Of the 8 such
        registry types only ``swiglu`` does, and its rule is 2/3 of
        ``output_dim * factor``, rounded up to ``ffn_multiple_of`` (D-020).

        The remaining 7 optional-``hidden_dim`` types (``kan``, ``mixer``,
        ``tversky``, ...) have no expansion concept. They size themselves from
        unrelated parameters, so this value does not reach them.

    :type ffn_expansion_factor: int
    :param kwargs: Additional arguments for the base Layer.
    """

    def __init__(
        self,
        task_config: VLMTaskConfig,
        num_layers: int = 6,
        num_heads: int = 12,
        ffn_type: FFNType = "swiglu",
        ffn_expansion_factor: int = 4,
        **kwargs: Any,
    ) -> None:
        """
        Store the configuration and create every decoder sub-layer.

        The loop creates, per decoder layer, one self-attention layer, one
        cross-attention layer, one FFN and three normalization layers. The
        norms are kept in one flat list, so layer ``i`` owns indices ``3i``,
        ``3i + 1`` and ``3i + 2``. One final ``Dense`` projects to the
        vocabulary.

        The layer name is derived from ``task_config.name``, so a caller must
        not pass ``name``.

        :param task_config: Configuration object for the task.
        :type task_config: VLMTaskConfig
        :param num_layers: Number of decoder layers.
        :type num_layers: int
        :param num_heads: Number of attention heads. ``hidden_dim`` must divide
            by it.
        :type num_heads: int
        :param ffn_type: Which registered FFN type each decoder layer builds.
        :type ffn_type: FFNType
        :param ffn_expansion_factor: FFN width multiplier.
        :type ffn_expansion_factor: int
        :param kwargs: Additional arguments for the base Layer class.
        :raises ValueError: If ``hidden_dim`` is not divisible by
            ``num_heads``, ``task_config.vocab_size`` is unset, or
            ``task_config.num_classes`` is set.
        :return: None.
        :rtype: None
        """
        _check_task_config_fields(
            task_config, "ImageCaptioningHead", required=("vocab_size",))
        kwargs.setdefault("name", f"{task_config.name}_head")
        super().__init__(**kwargs)
        self.task_config = task_config
        # D-014: dead fallback removed (see BaseVLMHead.__init__).
        self.hidden_dim = task_config.hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_type = ffn_type
        self.ffn_expansion_factor = ffn_expansion_factor

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

        # CREATE all sub-layers in __init__
        self.cross_attention_layers = []
        self.self_attention_layers = []
        self.ffn_layers = []
        self.norm_layers = []

        for i in range(self.num_layers):
            cross_attn = create_attention_layer(
                "multi_head_cross",
                dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout_rate=self.task_config.dropout_rate,
                name=f"cross_attention_{i}",
            )
            self.cross_attention_layers.append(cross_attn)

            self_attn = create_attention_layer(
                "multi_head",
                dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout_rate=self.task_config.dropout_rate,
                name=f"self_attention_{i}",
            )
            self.self_attention_layers.append(self_attn)

            # Supply `hidden_dim` only to FFN types that REQUIRE it. 13 of the
            # registry's 21 types do (`mlp`, `geglu`, `glu`, `reglu`, `lowrank`,
            # ...), and omitting it raised
            # "Required parameters missing for mlp: ['hidden_dim']" -- so every
            # one of those `ffn_type` values was unusable on this head.
            #
            # It is passed CONDITIONALLY rather than always, because the default
            # `swiglu` lists `hidden_dim` as OPTIONAL and derives it internally
            # (2/3 rule from `ffn_expansion_factor`, rounded to
            # `ffn_multiple_of`). Passing it explicitly would override that
            # derivation and silently change the default path's widths.
            #
            # The OUTPUT width is routed by name, not hardcoded -- same helper,
            # same arguments as `ImageTextMatchingHead._create_post_fusion_layers` (I-4). See
            # `_ffn_width_kwargs` (D-014) for why the literal string
            # `"output_dim"` must not reappear here, and the `ffn_type` docstring
            # above for which types this does and does not make reachable.
            ffn_config = {
                "type": self.ffn_type,
                "name": f"ffn_{i}",
                **_ffn_width_kwargs(self.ffn_type, self.hidden_dim),
            }
            _entry = FFN_REGISTRY.get(self.ffn_type, {})
            if "hidden_dim" in _entry.get("required_params", ()):
                ffn_config["hidden_dim"] = self.hidden_dim * self.ffn_expansion_factor
            elif "ffn_expansion_factor" in set(
                _entry.get("required_params", ())
            ) | set(_entry.get("optional_params", {})):
                # D-020: forward the factor to types that derive their own width,
                # or this head's `ffn_expansion_factor` is inert for them. Same
                # rule as `ImageTextMatchingHead._create_post_fusion_layers` -- keeping the two
                # sites identical is the entire point of D-008.
                ffn_config["ffn_expansion_factor"] = self.ffn_expansion_factor

            # D-019: same pre-filter as `ImageTextMatchingHead._create_post_fusion_layers`, and
            # for the same reason (keeping the two sites identical is I-4).
            # MEASURED difference from site 1, recorded so nobody "fixes" a
            # symmetry that is not there: at HEAD this loop emitted ZERO dropped
            # keys over all 21 types, because -- unlike site 1 -- it never
            # injects `dropout_rate`, and every other key it sends is already
            # gated on the registry. The pre-filter is therefore a no-op here
            # TODAY; it is present so that adding one unconditional convenience
            # cannot silently re-arm the hazard.
            ffn_config = assemble_ffn_config(self.ffn_type, ffn_config)
            ffn = create_ffn_from_config(ffn_config)
            self.ffn_layers.append(ffn)

            norm1 = create_normalization_layer("rms_norm", name=f"norm1_{i}")
            norm2 = create_normalization_layer("rms_norm", name=f"norm2_{i}")
            norm3 = create_normalization_layer("rms_norm", name=f"norm3_{i}")
            self.norm_layers.extend([norm1, norm2, norm3])

        # Final projection to vocabulary
        self.output_proj = keras.layers.Dense(
            self.task_config.vocab_size, name=f"{self.name}_output_proj"
        )

    def build(self, input_shape: Union[Dict, Tuple, List]) -> None:
        """Explicitly build every sub-layer, in computational order.

        Without this, the sub-layers created in ``__init__`` stay unbuilt until
        Keras traces ``call()``. A ``.keras`` round-trip through a Functional
        model then fails to restore them. Measured as ``ValueError: A total of
        12 objects could not be loaded``, with ``<Dense name=kv, built=False>``
        as the example. This is the repo's documented lazy-sublayer
        serialization trap.

        Note the shape contract this head already relies on. ``call()`` adds
        each sub-block's output straight onto its input (``x + attn_output``),
        with no projection anywhere. So the text feature width must equal
        ``hidden_dim``, and the builds below use ``hidden_dim`` for the whole
        decoder stack. The vision width is independent. It only enters as the
        cross-attention's key/value width.

        :param input_shape: Shape dict with ``vision_features`` and
            ``text_features`` entries, as passed to ``call()``.
        :type input_shape: Union[Dict, Tuple, List]
        """
        if self.built:
            return

        text_shape = input_shape["text_features"]
        vision_shape = input_shape["vision_features"]
        for key, shape in (("text_features", text_shape),
                           ("vision_features", vision_shape)):
            if len(shape) != 3:
                raise ValueError(
                    f"ImageCaptioningHead: {key} must be rank 3, (batch, "
                    f"length, width), got shape {tuple(shape)}. The decoder "
                    f"attends over both sequences; a pooled input has none."
                )
        batch = text_shape[0]
        seq_len = text_shape[1]

        # DECISION plan-2026-10-14T062616-81938e81/D-056
        # Refuse a text width other than hidden_dim here, at build. Do NOT
        # rely on the residual add: before this check the head built and
        # then failed inside MultiHeadCrossAttention.call, naming neither the
        # head nor text_features. See decisions.md D-056.
        if text_shape[-1] is not None and text_shape[-1] != self.hidden_dim:
            raise ValueError(
                f"ImageCaptioningHead: text_features is {text_shape[-1]} wide, "
                f"but the decoder adds every sub-block back onto it without a "
                f"projection, so it must be task_config.hidden_size="
                f"{self.hidden_dim} wide. Project the text embeddings first; "
                f"vision_features may be any width."
            )

        # The decoder stream carries `hidden_dim` throughout (see the residual
        # note above).
        stream_shape = (batch, seq_len, self.hidden_dim)

        for i in range(self.num_layers):
            # Self-attention takes one shape; cross-attention takes [query, kv].
            self.self_attention_layers[i].build(stream_shape)
            self.norm_layers[i * 3].build(stream_shape)
            self.cross_attention_layers[i].build([stream_shape, tuple(vision_shape)])
            self.norm_layers[i * 3 + 1].build(stream_shape)
            self.ffn_layers[i].build(stream_shape)
            self.norm_layers[i * 3 + 2].build(stream_shape)

        self.output_proj.build(stream_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: Dict[str, keras.KerasTensor],
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """
        Run the decoder stack and project to the vocabulary.

        ``text_features`` must already be embedded. This head holds no
        embedding table, and the residual adds carry no projection, so the text
        width must equal ``hidden_dim``.

        :param inputs: Dict with ``vision_features`` and ``text_features``,
            and optionally ``vision_mask`` and ``text_mask``.
        :type inputs: Dict[str, keras.KerasTensor]
        :param training: Keras training flag, forwarded to every sub-layer.
        :type training: Optional[bool]
        :return: Dict with ``logits`` and ``hidden_states``.
        :rtype: Dict[str, keras.KerasTensor]
        """
        vision_features = inputs["vision_features"]
        # The text is assumed pre-embedded; this head holds no embedding table.
        text_features = inputs["text_features"]
        vision_mask = inputs.get("vision_mask")
        text_mask = inputs.get("text_mask")

        x = text_features
        seq_len = keras.ops.shape(x)[1]
        # Lower-triangular KEEP mask: 1 = attend to current and past, 0 = future.
        #
        # Build it from `MaskFactory.create_causal_mask`, an arange index
        # comparison, and not from `keras.ops.tril`. `keras.ops.tril` and `keras.ops.triu` share an
        # implementation that routes through a `tf.cond` rejecting a Python-bool
        # predicate once traced, raising `TypeError: pred must not be a Python
        # bool`. That works eagerly and fails on every graph path: `tf.function`,
        # `Model.predict`, `.keras` save and load, and `jit_compile=True`. It
        # fails for both static and symbolic sequence lengths. Keras downgrades
        # such a crash during build tracing to a UserWarning, so a green test
        # suite could not see it. The same trap is documented in the SD3 MMDiT
        # text encoders.
        #
        # MaskFactory returns the BLOCK polarity, True where a position must be
        # suppressed. This site needs the complementary KEEP mask as a float,
        # hence `logical_not` plus a cast.
        #
        # This block-to-keep adapter now exists at two sites: here and in the
        # MobileCLIP components module. Two is not enough to earn a shared
        # helper. A third consumer should promote it into a keep-polarity
        # variant in `utils/masking/factory.py` rather than copying it again.
        # Promoting it is deferred because it touches two shipped,
        # mixed-precision-sensitive call sites and must preserve the cast target
        # below. That target is `backend.floatx()` because it is what the
        # previous `keras.ops.ones(...)` defaulted to, which keeps the change
        # numerically inert.
        causal_mask = keras.ops.cast(
            keras.ops.logical_not(MaskFactory.create_causal_mask(seq_len, dtype="bool")),
            keras.config.floatx(),
        )
        # Expand to the (1, S, S) full-mask form the attention layers expect.
        causal_mask = keras.ops.expand_dims(causal_mask, 0)
        if text_mask is not None:
            # A padded token is no key for any query: (1, S, S) * (B, 1, S).
            # A query row left with no key (a left-padded position) is
            # rescued by the attention layer; its output is padding anyway.
            causal_mask = causal_mask * keras.ops.expand_dims(
                keras.ops.cast(text_mask, causal_mask.dtype), 1)

        for i in range(self.num_layers):
            # Self-attention with causal mask
            attn_output = self.self_attention_layers[i](
                x, attention_mask=causal_mask, training=training
            )
            x = self.norm_layers[i * 3](x + attn_output)

            # Cross-attention to vision features. `vision_mask` (B, N_vis) is
            # a key-padding mask, shared by every text query.
            cross_attn_output = self.cross_attention_layers[i](
                x, kv_input=vision_features, attention_mask=vision_mask,
                training=training
            )
            x = self.norm_layers[i * 3 + 1](x + cross_attn_output)

            # FFN
            ffn_output = self.ffn_layers[i](x, training=training)
            x = self.norm_layers[i * 3 + 2](x + ffn_output)

        logits = self.output_proj(x)
        return {"logits": logits, "hidden_states": x}

    def compute_output_shape(self, input_shape):
        """
        Return the output shapes, mirroring the ``call`` return value.

        :param input_shape: Shape dict; only ``text_features`` is read.
        :type input_shape: Dict
        :return: Shapes for ``logits`` and ``hidden_states``.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        text_shape = input_shape["text_features"]
        batch, seq_len = text_shape[0], text_shape[1]
        return {
            "logits": (batch, seq_len, self.task_config.vocab_size),
            "hidden_states": (batch, seq_len, self.hidden_dim),
        }

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments needed to rebuild this head.

        :return: Serializable layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "task_config": self.task_config.to_dict(),
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "ffn_type": self.ffn_type,
                "ffn_expansion_factor": self.ffn_expansion_factor,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImageCaptioningHead":
        """
        Reconstruct the head, deserializing the ``task_config`` dataclass.

        The stored ``name`` is passed on, so a custom name survives; the
        default is ``<task_config.name>_head``.

        :param config: Serialized layer configuration.
        :type config: Dict[str, Any]
        :return: A reconstructed head instance.
        :rtype: ImageCaptioningHead
        """
        config = dict(config)
        config["task_config"] = VLMTaskConfig.coerce(config["task_config"])
        return cls(**config)
