"""
SAM 3's MaskFormer segmentation head: prompt-conditioned pixels, one mask per query.

This module provides the single public class :class:`Sam3SegmentationHead`. It
turns the multi-scale pyramid produced by
:class:`~dl_techniques.models.SAM.SAM3.necks.Sam3DualViTDetNeck` plus the object
queries produced by
:class:`~dl_techniques.models.SAM.SAM3.decoder.Sam3TransformerDecoder` into one
binary mask logit map per query, in the textbook MaskFormer way: a pixel
embedding is decoded once for the whole image, every query is projected into
the same embedding space, and the mask is their dot product.

Architecture:
    Three stages, in this order::

        1. prompt cross-attend   (optional, pre-norm residual)
           tgt2 = cross_attend(LayerNorm(encoder_states), prompt, prompt)
           encoder_states = tgt2 + encoder_states          <- residual, NOT a
                                                              replacement

        2. pixel decoder         (top-down FPN merge, COARSEST first)
           prev = feats[-1]                                <- the coarsest map
           for curr in reversed(feats[:-1]):               <- coarse -> fine
               prev = curr + resize(prev, curr.shape, "nearest")
               prev = relu(GroupNorm(8)(Conv3x3(prev)))

        3. mask decode
           pixel_embed  = Conv1x1(prev)          -> d_model per pixel
           mask_embed   = 3-layer MLP(queries)   -> mask_dim per query
           pred_masks   = einsum("bqc,bhwc->bqhw", mask_embed, pixel_embed)
           semantic_seg = Conv1x1(prev)          -> 1 channel per pixel

    The cross-attend happens BEFORE pixel decoding, so the text prompt reaches
    the pixel features themselves and not only the queries. The result is then
    folded back into the coarsest pyramid level, which is the level the FPN
    merge starts from -- so prompt information propagates through every
    upsampling stage.

Implementation notes:
    - Feature maps are channels-LAST throughout, matching the trunk and the
      neck. The reference is channels-first; the merge, the group
      normalization and the mask einsum are all written for the channels-last
      layout directly rather than transposed into it.
    - The head has **no presence mechanism of any kind** -- not disabled, not
      built and left unused: absent. The shipped reference configuration
      constructs this head with its presence head switched off and drives the
      presence signal from the decoder's own presence token instead, so a
      presence branch here would be a second, dead signal.
    - The pixel decoder is part of this class rather than a class of its own.
      It has exactly one call site and no independent configuration surface.

References:
    - Cheng, B., Schwing, A., & Kirillov, A. (2021). "Per-Pixel Classification
      is Not All You Need for Semantic Segmentation" (MaskFormer; the
      per-query dot-product-with-pixel-embedding mask decode).
    - Lin, T.-Y. et al. (2017). "Feature Pyramid Networks for Object
      Detection" (the top-down merge with additive lateral connections).
"""

import keras
from keras import layers, ops
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.attention.factory import create_attention_layer
from dl_techniques.layers.norms.factory import create_normalization_layer
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class Sam3SegmentationHead(keras.layers.Layer):
    """SAM 3's MaskFormer head: FPN pixel decoder plus a per-query mask decode.

    **Architecture Overview:**

    .. code-block:: text

        prompt ──┐
                 ▼
        encoder_states ──▶ LN ──▶ cross-attend ──▶ (+) ──▶ reshape
                 │                                  ▲         │
                 └──────────────────────────────────┘         │
                                                              ▼
        feats[0] (finest) ... feats[-2]      feats[-1] (coarsest, REPLACED)
             │                    │                   │
             │                    │      ┌────────────┘
             │                    ▼      ▼
             │                  (+) ◀── resize nearest
             │                    │
             │                Conv3x3 ▶ GroupNorm(8) ▶ ReLU
             │                    │
             └───────▶ (+) ◀──────┘   ... one stage per remaining level
                        │
                    Conv3x3 ▶ GroupNorm(8) ▶ ReLU  ──▶ pixel features
                        │                                  │
              ┌─────────┴──────────┐                       │
              ▼                    ▼                       │
        Conv1x1 -> 1        Conv1x1 -> d_model             │
        semantic_seg          pixel_embed ──▶ einsum ◀── mask_embed(queries)
                                                 │
                                             pred_masks

    :param d_model: Working width. Every pyramid level arrives at this width
        and every stage keeps it.
    :type d_model: int
    :param upsampling_stages: Number of top-down merge stages. Must equal
        ``len(backbone_feats) - 1``.
    :type upsampling_stages: int
    :param mask_dim: Width of the mask embedding and therefore of the pixel
        embedding the einsum contracts against. ``None`` means ``d_model``.
    :type mask_dim: Optional[int]
    :param num_heads: Heads of the prompt cross-attention.
    :type num_heads: int
    :param num_groups: Groups of every stage's group normalization.
    :type num_groups: int
    :param interpolation_mode: Upsampling mode of the top-down merge, either
        ``"nearest"`` (the reference) or ``"bilinear"``.
    :type interpolation_mode: str
    :param use_cross_attend_prompt: Whether to build the prompt cross-attend.
    :type use_cross_attend_prompt: bool
    :param attention_dropout_rate: Dropout of the prompt cross-attention. The
        reference passes ``0``.
    :type attention_dropout_rate: float
    :param norm_epsilon: Epsilon of BOTH normalizations here -- the
        cross-attention's layer norm and every stage's group norm.
    :type norm_epsilon: float
    :param kwargs: Additional keyword arguments for the ``Layer`` base class.

    :raises ValueError: If any width or count is non-positive, if ``d_model``
        is not divisible by ``num_heads`` or by ``num_groups``, or if
        ``interpolation_mode`` is not supported.

    Example:
        >>> import numpy as np
        >>> head = Sam3SegmentationHead(d_model=8, upsampling_stages=1,
        ...                             num_heads=2, num_groups=2,
        ...                             use_cross_attend_prompt=False)
        >>> feats = [np.zeros((1, 4, 4, 8), "float32"),
        ...          np.zeros((1, 2, 2, 8), "float32")]
        >>> out = head(feats, np.zeros((1, 3, 8), "float32"),
        ...            np.zeros((1, 4, 8), "float32"))
        >>> out["pred_masks"].shape
        (1, 3, 4, 4)
    """

    SUPPORTED_INTERPOLATIONS: Tuple[str, ...] = ("nearest", "bilinear")

    def __init__(
            self,
            d_model: int = 256,
            upsampling_stages: int = 3,
            mask_dim: Optional[int] = None,
            num_heads: int = 8,
            num_groups: int = 8,
            interpolation_mode: str = "nearest",
            use_cross_attend_prompt: bool = True,
            attention_dropout_rate: float = 0.0,
            norm_epsilon: float = 1e-5,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        for name, value in (("d_model", d_model), ("num_heads", num_heads),
                            ("num_groups", num_groups),
                            ("upsampling_stages", upsampling_stages)):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by "
                             f"num_heads ({num_heads})")
        if d_model % num_groups != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by "
                             f"num_groups ({num_groups})")
        if interpolation_mode not in self.SUPPORTED_INTERPOLATIONS:
            raise ValueError(
                f"interpolation_mode={interpolation_mode!r} is not supported; "
                f"supported modes are {self.SUPPORTED_INTERPOLATIONS}")
        if not 0.0 <= attention_dropout_rate < 1.0:
            raise ValueError(f"attention_dropout_rate must be in [0, 1), got "
                             f"{attention_dropout_rate}")

        self.d_model = int(d_model)
        self.upsampling_stages = int(upsampling_stages)
        self.mask_dim = int(mask_dim) if mask_dim is not None else self.d_model
        self.num_heads = int(num_heads)
        self.num_groups = int(num_groups)
        self.interpolation_mode = str(interpolation_mode)
        self.use_cross_attend_prompt = bool(use_cross_attend_prompt)
        self.attention_dropout_rate = float(attention_dropout_rate)
        self.norm_epsilon = float(norm_epsilon)

        # DECISION plan-2026-08-04T044628-4c240b4c/D-119
        # There is NO presence mechanism on this class -- no constructor flag,
        # no sub-layer, no output key, no attribute. Do NOT add one "for
        # symmetry with the reference", and do NOT add a disabled one: the
        # shipped reference configuration constructs this head with
        # `presence_head=False` and drives presence from the DECODER's presence
        # token, which is the tensor the top-level model multiplies into the
        # class logits. A second presence signal here would be dead weight that
        # a future reader would wire up by mistake. See decisions.md D-119.
        self.cross_attn_norm = create_normalization_layer(
            "layer_norm", epsilon=self.norm_epsilon, name="cross_attn_norm")
        # Keys and values come from the SAME tensor here (the prompt), which is
        # the contract the repository's cross-attention layer expresses exactly.
        # Structural kwargs are set explicitly rather than inherited (D-102).
        self.cross_attend_prompt = create_attention_layer(
            "multi_head_cross", dim=self.d_model, num_heads=self.num_heads,
            dropout_rate=self.attention_dropout_rate, use_bias=True,
            shared_qk_projections=False, probability_type="softmax",
            qk_norm_type=None, name="cross_attend_prompt")

        # DECISION plan-2026-08-04T044628-4c240b4c/D-118
        # `epsilon` is passed EXPLICITLY to every group norm below. Keras
        # `GroupNormalization` defaults to `epsilon=1e-3`; the reference's
        # `nn.GroupNorm` defaults to `1e-5`. MEASURED here, and the same 100x
        # silent divergence this package already measured for
        # `LayerNormalization` at step 6. Do NOT drop the argument. See D-118.
        #
        # Both stacks are stored FLAT -- one list of `Conv2D`, one list of
        # `GroupNormalization`, never a list of per-stage pairs. A nested
        # `List[List[Layer]]` sub-layer store silently restores freshly
        # initialized kernels on a `.keras` round trip while the weight count,
        # the weight paths and the parameter total all match (D-098, measured
        # in this package on `necks.py`). Do NOT re-nest these for readability.
        self.pixel_convs: List[keras.layers.Layer] = [
            layers.Conv2D(self.d_model, kernel_size=3, padding="same",
                          use_bias=True, name=f"pixel_conv_{index}")
            for index in range(self.upsampling_stages)
        ]
        self.pixel_norms: List[keras.layers.Layer] = [
            layers.GroupNormalization(groups=self.num_groups,
                                      epsilon=self.norm_epsilon,
                                      name=f"pixel_norm_{index}")
            for index in range(self.upsampling_stages)
        ]

        self.semantic_seg_head = layers.Conv2D(
            1, kernel_size=1, use_bias=True, name="semantic_seg_head")
        self.instance_seg_head = layers.Conv2D(
            self.d_model, kernel_size=1, use_bias=True,
            name="instance_seg_head")

        # DECISION plan-2026-08-04T044628-4c240b4c/D-117
        # The mask-embedding MLP is composed HERE from three `Dense` layers
        # rather than reusing `layers/eomt_mask.py`'s `EomtMask`, whose mask
        # branch is otherwise parameter-EXACT for this site (197,376 params at
        # d_model=256, measured, matching the reference's 3-layer MLP to the
        # unit). The blocker is its class head: `EomtMask` builds a
        # `Dense(num_classes)` UNCONDITIONALLY, there is no flag that switches
        # it off, and `num_classes=0` raises -- so reusing it ships a fixed
        # class table (257 dead parameters at the settled width, evaluated on
        # every forward pass) into a head whose whole point is that its
        # vocabulary is open. Do NOT "simplify" this back to `EomtMask`: the
        # equivalence of the two mask branches is pinned by a test that
        # transplants weights between them, so the reuse remains checkable
        # without paying for the class head. See decisions.md D-117.
        self.mask_embed: List[keras.layers.Layer] = [
            layers.Dense(self.d_model, activation="relu", use_bias=True,
                         name="mask_embed_0"),
            layers.Dense(self.d_model, activation="relu", use_bias=True,
                         name="mask_embed_1"),
            layers.Dense(self.mask_dim, use_bias=True, name="mask_embed_2"),
        ]

        logger.info(
            f"Sam3SegmentationHead: d_model={self.d_model}, "
            f"stages={self.upsampling_stages}, mask_dim={self.mask_dim}, "
            f"upsample={self.interpolation_mode}, "
            f"cross_attend={self.use_cross_attend_prompt}"
        )

    # -----------------------------------------------------------------
    # shape arithmetic -- every helper is owned by this class alone, so each
    # is a `@staticmethod` rather than a module-level function (D-109/D-114:
    # module level is for helpers with more than one owner).
    # -----------------------------------------------------------------

    @staticmethod
    def _check_feature_shapes(
            shapes: Sequence[Tuple[Optional[int], ...]], d_model: int,
            stages: int,
    ) -> List[Tuple[Optional[int], ...]]:
        """Validate the pyramid's shapes and return them as tuples.

        :param shapes: One channels-last shape per pyramid level, finest first.
        :type shapes: Sequence[Tuple[Optional[int], ...]]
        :param d_model: The width every level must already have.
        :type d_model: int
        :param stages: The configured number of merge stages.
        :type stages: int
        :return: The same shapes, normalized to tuples.
        :rtype: List[Tuple[Optional[int], ...]]
        :raises ValueError: On a wrong level count, rank or width.
        """
        shapes = [tuple(shape) for shape in shapes]
        if len(shapes) != stages + 1:
            raise ValueError(
                f"the pyramid must hold upsampling_stages + 1 = {stages + 1} "
                f"levels, got {len(shapes)}")
        for index, shape in enumerate(shapes):
            if len(shape) != 4:
                raise ValueError(
                    f"pyramid level {index} must be a rank-4 channels-last "
                    f"map (batch, height, width, d_model), got {shape}")
            if shape[-1] is not None and shape[-1] != d_model:
                raise ValueError(
                    f"pyramid level {index} has width {shape[-1]}, which must "
                    f"already equal d_model ({d_model}); the neck projects "
                    f"every scale before this head sees it")
        return shapes

    def _merge(self, coarse: Any, fine: Any) -> Any:
        """Fuse one upsampled coarse level into the next finer level.

        :param coarse: The running top-down feature, channels-last.
        :type coarse: Any
        :param fine: This stage's lateral feature, channels-last.
        :type fine: Any
        :return: Their sum at ``fine``'s resolution.
        :rtype: Any
        :raises ValueError: If the fused width is not ``d_model``.
        """
        # DECISION plan-2026-08-04T044628-4c240b4c/D-120
        # The skip fusion is an ADDITION and the upsample is NEAREST-neighbour.
        # Neither is interchangeable with its obvious alternative:
        #   * a concatenating fusion is COHERENT -- it runs, it trains, and the
        #     following convolution simply infers a wider input. In this
        #     package's own iteration-1 precedent a coherent concat port left
        #     35 of 37 tests green and only the WIDTH assertions fired, which
        #     is why the width check below is explicit and lives on the fused
        #     tensor rather than being left to the convolution's build.
        #   * a bilinear upsample differs from nearest only at pixels strictly
        #     between two distinct coarse values; on a CONSTANT feature map the
        #     two coincide exactly, so a value oracle probed at a constant
        #     input cannot tell them apart.
        # See decisions.md D-120.
        size = (fine.shape[1], fine.shape[2])
        if size[0] is None or size[1] is None:
            raise ValueError(
                "the top-down merge resizes to a STATIC target grid; the "
                f"lateral feature's spatial shape is {size}")
        upsampled = ops.image.resize(
            coarse, size=size, interpolation=self.interpolation_mode)
        fused = fine + upsampled
        if fused.shape[-1] != self.d_model:
            raise ValueError(
                f"the top-down fusion produced width {fused.shape[-1]}, but "
                f"every stage must stay at d_model ({self.d_model}); a "
                f"concatenating fusion reaches this check")
        return fused

    def _decode_pixels(self, feats: List[Any], training: Optional[bool]) -> Any:
        """Run the top-down FPN merge over the whole pyramid.

        :param feats: Pyramid levels, FINEST first, all at ``d_model``.
        :type feats: List[Any]
        :param training: Keras training flag.
        :type training: Optional[bool]
        :return: The merged feature at the FINEST level's resolution.
        :rtype: Any
        """
        # DECISION plan-2026-08-04T044628-4c240b4c/D-121
        # The merge starts at the COARSEST level (`feats[-1]`) and walks the
        # remaining levels from coarse to fine (`feats[:-1]` REVERSED). Do NOT
        # start from the finest and walk outward: that also runs, also produces
        # a rank-4 map, and silently emits masks at the WRONG resolution --
        # which is why the guard is a resolution assertion. See D-121.
        running = feats[-1]
        for index, lateral in enumerate(reversed(feats[:-1])):
            running = self._merge(running, lateral)
            running = self.pixel_convs[index](running, training=training)
            running = ops.relu(
                self.pixel_norms[index](running, training=training))
        return running

    def _fold_encoder_states(
            self, encoder_hidden_states: Any, coarse_shape: Tuple,
    ) -> Any:
        """Fold the flat encoder sequence back into the coarsest feature map.

        The reference keeps its fused image tokens in a flat sequence that may
        carry non-spatial tokens after them, and takes the LEADING
        ``height * width`` of them, row-major.

        :param encoder_hidden_states: ``(batch, sequence, d_model)``.
        :type encoder_hidden_states: Any
        :param coarse_shape: The coarsest level's static shape.
        :type coarse_shape: Tuple
        :return: ``(batch, height, width, d_model)``.
        :rtype: Any
        """
        height, width = coarse_shape[1], coarse_shape[2]
        spatial = height * width
        flat = encoder_hidden_states[:, :spatial, :]
        return ops.reshape(
            flat, (ops.shape(encoder_hidden_states)[0], height, width,
                   self.d_model))

    # -----------------------------------------------------------------

    def build(
            self,
            backbone_feats_shape: Sequence[Tuple[Optional[int], ...]],
            obj_queries_shape: Tuple[Optional[int], ...],
            encoder_hidden_states_shape: Tuple[Optional[int], ...],
            prompt_shape: Optional[Tuple[Optional[int], ...]] = None,
            **kwargs: Any,
    ) -> None:
        """Build every sub-layer explicitly, from the call signature's shapes.

        :param backbone_feats_shape: One shape per pyramid level, FINEST first.
        :type backbone_feats_shape: Sequence[Tuple[Optional[int], ...]]
        :param obj_queries_shape: ``(batch, num_queries, d_model)`` or
            ``(num_layers, batch, num_queries, d_model)``.
        :type obj_queries_shape: Tuple[Optional[int], ...]
        :param encoder_hidden_states_shape: ``(batch, sequence, d_model)``.
        :type encoder_hidden_states_shape: Tuple[Optional[int], ...]
        :param prompt_shape: ``(batch, num_tokens, d_model)``; required when
            the prompt cross-attend is enabled.
        :type prompt_shape: Optional[Tuple[Optional[int], ...]]
        :param kwargs: Ignored; accepted so the layer builds from its full call
            signature. A boolean mask argument's shape arrives as ``None``.
        :raises ValueError: On a wrong pyramid, a wrong query rank or width, or
            a missing prompt shape.
        """
        if self.built:
            return
        shapes = self._check_feature_shapes(
            backbone_feats_shape, self.d_model, self.upsampling_stages)
        obj_queries_shape = tuple(obj_queries_shape)
        if len(obj_queries_shape) not in (3, 4):
            raise ValueError(
                f"obj_queries must be (batch, num_queries, d_model) or "
                f"(num_layers, batch, num_queries, d_model), got "
                f"{obj_queries_shape}")
        if obj_queries_shape[-1] is not None \
                and obj_queries_shape[-1] != self.d_model:
            raise ValueError(f"obj_queries width {obj_queries_shape[-1]} != "
                             f"d_model ({self.d_model})")

        encoder_hidden_states_shape = tuple(encoder_hidden_states_shape)
        if self.use_cross_attend_prompt:
            if prompt_shape is None:
                raise ValueError("prompt_shape is required when "
                                 "use_cross_attend_prompt=True")
            self.cross_attn_norm.build(encoder_hidden_states_shape)
            self.cross_attend_prompt.build(
                [encoder_hidden_states_shape, tuple(prompt_shape)])

        # Every stage consumes its LATERAL level's grid, walking coarse to fine.
        for index, lateral in enumerate(reversed(shapes[:-1])):
            stage_shape = (lateral[0], lateral[1], lateral[2], self.d_model)
            self.pixel_convs[index].build(stage_shape)
            self.pixel_norms[index].build(stage_shape)
        finest = (shapes[0][0], shapes[0][1], shapes[0][2], self.d_model)
        self.semantic_seg_head.build(finest)
        self.instance_seg_head.build(finest)

        query_shape = obj_queries_shape[1:] if len(obj_queries_shape) == 4 \
            else obj_queries_shape
        width = self.d_model
        for dense in self.mask_embed:
            dense.build(query_shape[:-1] + (width,))
            width = dense.units
        super().build(backbone_feats_shape)

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------

    def call(
            self,
            backbone_feats: Sequence[Any],
            obj_queries: Any,
            encoder_hidden_states: Any,
            prompt: Optional[Any] = None,
            prompt_padding_mask: Optional[Any] = None,
            training: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Decode one mask logit map per query, plus a semantic map.

        :param backbone_feats: Pyramid levels, FINEST first, every level
            already at ``d_model``.
        :type backbone_feats: Sequence[Any]
        :param obj_queries: ``(batch, num_queries, d_model)``, or the decoder's
            per-layer stack ``(num_layers, batch, num_queries, d_model)`` -- in
            which case only the LAST layer's queries are decoded, matching the
            shipped configuration's disabled auxiliary-mask path.
        :type obj_queries: Any
        :param encoder_hidden_states: ``(batch, sequence, d_model)``; its
            leading ``height * width`` tokens replace the coarsest pyramid
            level after the prompt cross-attend.
        :type encoder_hidden_states: Any
        :param prompt: ``(batch, num_tokens, d_model)``; required when the
            prompt cross-attend is enabled.
        :type prompt: Optional[Any]
        :param prompt_padding_mask: ``(batch, num_tokens)``, ``True`` at
            PADDING -- a key-padding mask, the opposite polarity to the causal
            KEEP mask the text tower builds. The polarity is in the name.
        :type prompt_padding_mask: Optional[Any]
        :param training: Keras training flag.
        :type training: Optional[bool]
        :return: ``pred_masks`` ``(batch, num_queries, height, width)`` and
            ``semantic_seg`` ``(batch, height, width, 1)``, both at the FINEST
            level's resolution. There is no presence key (D-119).
        :rtype: Dict[str, Any]
        :raises ValueError: If the cross-attend is enabled and no prompt is
            supplied.
        """
        feats = list(backbone_feats)

        if self.use_cross_attend_prompt:
            if prompt is None:
                raise ValueError(
                    "prompt is required when use_cross_attend_prompt=True")
            keep = None
            if prompt_padding_mask is not None:
                keep = ops.logical_not(ops.cast(prompt_padding_mask, "bool"))
            attended = self.cross_attend_prompt(
                self.cross_attn_norm(encoder_hidden_states, training=training),
                prompt, attention_mask=keep, training=training)
            # Pre-norm RESIDUAL: the normalized tensor feeds the attention, the
            # UN-normalized one carries the skip.
            encoder_hidden_states = attended + encoder_hidden_states

        feats[-1] = self._fold_encoder_states(
            encoder_hidden_states, tuple(feats[-1].shape))
        pixel_features = self._decode_pixels(feats, training)

        pixel_embed = self.instance_seg_head(pixel_features, training=training)
        queries = obj_queries[-1] if len(obj_queries.shape) == 4 \
            else obj_queries
        for dense in self.mask_embed:
            queries = dense(queries, training=training)
        return {
            "pred_masks": ops.einsum("bqc,bhwc->bqhw", queries, pixel_embed),
            "semantic_seg": self.semantic_seg_head(
                pixel_features, training=training),
        }

    def compute_output_shape(
            self,
            backbone_feats_shape: Sequence[Tuple[Optional[int], ...]],
            obj_queries_shape: Optional[Tuple] = None,
            encoder_hidden_states_shape: Optional[Tuple] = None,
            **kwargs: Any,
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """Return both output shapes, derived from stored config only.

        :param backbone_feats_shape: One shape per pyramid level, FINEST first.
        :type backbone_feats_shape: Sequence[Tuple[Optional[int], ...]]
        :param obj_queries_shape: ``(batch, num_queries, d_model)`` or the
            per-layer stack.
        :type obj_queries_shape: Optional[Tuple]
        :param encoder_hidden_states_shape: Unused; present for the build
            contract.
        :type encoder_hidden_states_shape: Optional[Tuple]
        :param kwargs: Ignored.
        :return: ``pred_masks`` and ``semantic_seg`` shapes.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        finest = tuple(list(backbone_feats_shape)[0])
        queries = None
        if obj_queries_shape is not None:
            obj_queries_shape = tuple(obj_queries_shape)
            queries = obj_queries_shape[-2]
        return {
            "pred_masks": (finest[0], queries, finest[1], finest[2]),
            "semantic_seg": (finest[0], finest[1], finest[2], 1),
        }

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all ``__init__`` parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "upsampling_stages": self.upsampling_stages,
            "mask_dim": self.mask_dim,
            "num_heads": self.num_heads,
            "num_groups": self.num_groups,
            "interpolation_mode": self.interpolation_mode,
            "use_cross_attend_prompt": self.use_cross_attend_prompt,
            "attention_dropout_rate": self.attention_dropout_rate,
            "norm_epsilon": self.norm_epsilon,
        })
        return config

# ---------------------------------------------------------------------
