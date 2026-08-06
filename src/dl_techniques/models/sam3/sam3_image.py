"""
SAM 3's top-level text-prompted image model: the six components wired together.

This module provides the single public class :class:`Sam3Image`, the ninth and
last class of the SAM 3 phase-1 package. It owns no learned weights of its own --
every parameter belongs to one of the six components it composes -- and its whole
substance is the DATA FLOW between them plus one arithmetic expression: the
presence x localization fusion.

Architecture:
    .. code-block:: text

        image ─▶ Sam3ViTDetBackbone ─▶ ONE trunk map
                                          │
                                 Sam3DualViTDetNeck
                                          │
                          [4x, 2x, 1x, 0.5x] + per-scale sine PE
                                          │   drop the coarsest `scalp` levels
                          [4x, 2x, 1x]  ───┴──▶ segmentation pyramid
                                 │
                            1x flattened ─▶ image memory (+ its PE)
                                 │
        token ids ─▶ Sam3TextEncoder ─▶ prompt ─┐
                                 │              │
                          Sam3TransformerDecoder┘
                                 │
             hidden states, per-layer anchors, presence logits
                    │              │                │
        Sam3DotProductScoring   box head        (sigmoid)
                    │              │                │
              class logits ◀── FUSION* ◀────────────┘
                    │              │
                    │         pred_boxes
                    │
             Sam3SegmentationHead ─▶ pred_masks, semantic_seg

``*`` **The FUSION step is OFF by default.** It runs only when
``supervise_joint_box_scores=True``; the constructor defaults it to ``False``
and ``from_variant`` never sets it, so on every default path ``pred_logits`` is
the scorer's output untouched and ``_fuse`` is not reached (D-124, traced at the
pinned reference SHA -- ``build_sam3_image_model`` never passes the key either).
Everything below about the fusion describes that opt-in expression.

Three mechanisms carry this class's correctness, and each has its own guard:

1. **The fused presence is the DECODER's**, never the segmentation head's. The
   segmentation head in this package has no presence mechanism of any kind, so
   there is exactly ONE presence signal in the model and no way to fuse the
   wrong one by accident.
2. **The fusion multiplies PROBABILITIES and then re-logits**, it does not
   multiply logits (when it runs at all -- see ``*`` above). The two candidates
   agree only along a thin curve in ``(class logit, presence logit)`` space;
   everywhere else they differ by O(1) nats.
3. **The final box is produced HERE, not by the decoder.** The decoder returns
   the anchor each layer CONSUMED; the last layer's refinement is applied at
   this level, which is why the box head is re-applied to the stacked hidden
   states rather than read out of the stack.

**Deliberately NOT built in phase 1**, and named here rather than left to be
rediscovered as a gap:

- the vision-language **early-fusion encoder** that upstream runs between the
  neck and the decoder. Phase 1 feeds the neck's image memory and the text
  tower's prompt straight into the decoder. That is a structural divergence,
  not an oversight, and it is the largest single one in this package;
- the exemplar / geometry prompt path (points and boxes), which needs bilinear
  ``grid_sample`` and ``roi_align`` primitives ``keras.ops`` does not have;
- DAC query doubling, provably inert at inference;
- the per-layer auxiliary output stacks an auxiliary-loss training phase would
  consume -- this class returns the LAST layer's quantities only. The stacks
  themselves are still produced and are reachable by calling the decoder
  directly;
- the ``cxcywh -> xyxy`` box conversion, which is a consumer-side utility.

References:
    - Meta AI (2025). "SAM 3: Segment Anything with Concepts."
    - Carion, N. et al. (2020). "End-to-End Object Detection with Transformers."
    - Liu, S. et al. (2022). "DAB-DETR: Dynamic Anchor Boxes are Better Queries
      for DETR" (the logit-space box refinement reused here).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import keras
from keras import ops

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.sam3.decoder import Sam3TransformerDecoder
from dl_techniques.models.sam3.maskformer_segmentation import Sam3SegmentationHead
from dl_techniques.models.sam3.model_misc import Sam3DotProductScoring
from dl_techniques.models.sam3.necks import Sam3DualViTDetNeck
from dl_techniques.models.sam3.text_encoder_ve import Sam3TextEncoder
from dl_techniques.models.sam3.vitdet import Sam3ViTDetBackbone

#: The six composed components, in the order they run. Serialization iterates
#: this tuple rather than repeating the names, so a component added to the
#: constructor without being added here fails the round trip loudly.
COMPONENT_KEYS: Tuple[str, ...] = (
    "backbone", "neck", "text_encoder", "transformer", "dot_prod_scoring",
    "segmentation_head",
)


# ---------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class Sam3Image(keras.Model):
    """SAM 3's text-prompted image model.

    :param backbone: The ViTDet trunk.
    :type backbone: Union[Sam3ViTDetBackbone, Dict[str, Any]]
    :param neck: The dual SimpleFPN neck.
    :type neck: Union[Sam3DualViTDetNeck, Dict[str, Any]]
    :param text_encoder: The CLIP text tower wrapper.
    :type text_encoder: Union[Sam3TextEncoder, Dict[str, Any]]
    :param transformer: The DETR decoder stack. Its presence token must be
        enabled -- see :attr:`Sam3Image.call`'s output contract.
    :type transformer: Union[Sam3TransformerDecoder, Dict[str, Any]]
    :param dot_prod_scoring: The open-vocabulary class-score head.
    :type dot_prod_scoring: Union[Sam3DotProductScoring, Dict[str, Any]]
    :param segmentation_head: The MaskFormer head. Required, not optional.
    :type segmentation_head: Union[Sam3SegmentationHead, Dict[str, Any]]
    :param num_feature_levels: How many of the neck's pyramid levels reach the
        decoder as image memory, counted from the coarsest KEPT level. Only
        ``1`` is supported; see the ``raises`` note.
    :type num_feature_levels: int
    :param scalp: How many of the neck's COARSEST levels are discarded before
        anything downstream sees them. ``1`` at the settled configuration, so
        the ``0.5x`` level is built by the neck and then dropped.
    :type scalp: int
    :param supervise_joint_box_scores: Whether the presence x localization
        fusion is applied to the class logits. Default ``False`` -- the
        reference's own default AND the value its image-model builder leaves in
        place.
    :type supervise_joint_box_scores: bool
    :param detach_presence_in_joint_score: Whether the presence probability is
        detached before it multiplies the class probability, so the fusion
        passes no gradient into the presence head.
    :type detach_presence_in_joint_score: bool
    :param joint_score_clamp: The symmetric bound applied after the fusion's
        re-logit. ``10.0`` at the settled configuration -- and MEASURED to be
        non-binding, see the anchor at the fusion.
    :type joint_score_clamp: float
    :param kwargs: Additional keyword arguments for the ``Model`` base class.

    :raises ValueError: If any component's width disagrees with another's, if
        the neck's pyramid cannot supply the requested levels, if the
        segmentation head's stage count does not match the kept pyramid, if the
        decoder has no presence token, or if ``num_feature_levels != 1``. The
        multi-level memory path needs the flattened-level bookkeeping
        (``level_start_index`` / ``spatial_shapes``) that only the deferred
        fusion encoder consumes, so building it here would be speculative
        surface with no reachable caller.

    Example:
        >>> import numpy as np
        >>> model = Sam3Image.from_variant("tiny")
        >>> out = model({
        ...     "image": np.zeros((1, 32, 32, 3), dtype="float32"),
        ...     "token_ids": np.zeros((1, 8), dtype="int32"),
        ...     "token_padding_mask": np.zeros((1, 8), dtype="bool"),
        ... })
        >>> sorted(out)
        ['pred_boxes', 'pred_logits', 'pred_masks', 'presence_logit', 'semantic_seg']
    """

    #: Every number here has exactly one home. Component geometry that a
    #: component's own constructor already defaults is NOT repeated.
    #:
    #: ``sam3`` is the settled, released configuration, read from the pinned
    #: reference's own builder. ``tiny`` is **NOT a published SAM 3 size**: it is
    #: a deliberately small development geometry that exists so this package has
    #: a runnable end-to-end gate, and it is named here rather than hidden in a
    #: test fixture so nobody mistakes it for a released checkpoint. ``small``
    #: is **also NOT a published SAM 3 size** -- it is a trainable-on-12-GB
    #: geometry designed for this package's first learnability run, every field
    #: of it derived from the released configuration's own RATIOS with each
    #: deviation signed and named below. No published SAM 3 size is invented:
    #: the reference ships exactly one, and it is ``sam3``.
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "sam3": {
            "img_size": 1008, "patch_size": 14, "embed_dim": 1024, "depth": 32,
            "num_heads": 16, "mlp_ratio": 4.625, "window_size": 24,
            "global_att_blocks": (7, 15, 23, 31), "pretrain_img_size": 336,
            "drop_path_rate": 0.1,
            "d_model": 256, "scale_factors": (4.0, 2.0, 1.0, 0.5),
            "add_sam2_neck": False, "scalp": 1,
            "text_width": 1024, "text_depth": 24, "text_heads": 16,
            "context_length": 32, "vocab_size": 49408,
            "num_queries": 200, "decoder_layers": 6, "decoder_heads": 8,
            "dim_feedforward": 2048, "dropout_rate": 0.1,
            "d_proj": 256, "prompt_mlp_hidden_dim": 2048,
            "prompt_mlp_dropout": 0.1, "seg_num_heads": 8, "seg_num_groups": 8,
        },
        # DECISION plan-2026-08-05T124709-6c4fac48/D-017
        # `small` is NOT a published SAM 3 size. It exists because the two
        # variants above are both unusable for a training run: `tiny`'s 8x8
        # trunk grid is degenerate (the decoder must invent 16x localization out
        # of 64 tokens -- the exact confounder the SAM 2 investigation never
        # controlled), and `sam3`'s 10,072.9 MiB FORWARD peak leaves no room for
        # AdamW's two moment buffers on a 12 GB card.
        #
        # Every field is DERIVED from the released configuration's own ratios,
        # read at the pinned reference (`sam3/model_builder.py`), and every
        # deviation is a SIGNED NAMED divergence. The reference publishes ONE
        # size, so the oracle is a set of ratios, not a smaller config:
        #   R1  trunk head width  embed_dim/num_heads = 64
        #   R3  img_size/pretrain_img_size = 3
        #   R4  mlp_ratio = 4.625
        #   R5  grid/window_size = 3
        #   R6  global_att_blocks: every 8th, and the LAST block is ALWAYS
        #       global (the trunk's single output map IS that block's output)
        #   R7  d_model = embed_dim/4
        #   R8  text_width = embed_dim, text head width 64, text_depth = 0.75*depth
        #   R9  decoder head width  d_model/decoder_heads = 32
        #   R10 dim_feedforward = 8*d_model      R11 d_proj = d_model
        #   R12 prompt_mlp_hidden_dim = 8*d_model     R13 seg_num_groups = 8
        #   R14 decoder_layers/depth = 0.1875    R15 context_length = 32
        # Do NOT "simplify" a field back to a round number: five of the values
        # here (mlp_ratio, text_width, text_heads, dim_feedforward,
        # context_length) are the reference's EXACT numbers and were proposed as
        # rounder ones during planning. See decisions.md D-017.
        "small": {
            # grid 16 = 224/14. Patch 14 is the reference's exact patch size;
            # 224 is the smallest side giving a power-of-two grid (so every neck
            # scale 4/2/1/0.5 lands on an integer) with >= 2 windows per side.
            "img_size": 224, "patch_size": 14,
            # 192/3 -> head width 64 = R1 EXACT. 192 is the smallest multiple of
            # 64 that keeps `dim//4 = 48` integral for the neck's 4x branch.
            "embed_dim": 192, "depth": 6, "num_heads": 3,
            # R4 EXACT -- and 192*4.625 = 888 exactly, so nothing rounds.
            "mlp_ratio": 4.625,
            # DIVERGENCE -1 on R5: 2 windows per side, not 3, because 16 is not
            # divisible by 3 and `vitdet.py` REFUSES a window that does not
            # divide the grid (this port has no zero-padding branch, D-087).
            "window_size": 8,
            # R6's invariant EXACT (block 5 is the last). DIVERGENCE +1 block:
            # the reference's 1/8 density gives 0.75 globals at depth 6, and
            # with 2x2 windows a single global block would be the ONLY place
            # information ever crosses a window boundary.
            "global_att_blocks": (2, 5),
            # DIVERGENCE -1 on R3: grid 8, i.e. ratio 2 not 3 (16/3 is not an
            # integer). `tiny` uses the same ratio 2.
            "pretrain_img_size": 112,
            # DECISION plan-2026-08-05T124709-6c4fac48/D-018
            # 0.0 for all three rates, a DIVERGENCE -0.1 from the reference on
            # each, taken deliberately and NOT because "a small model needs less
            # regularization". D-123 MEASURED that the shared `StochasticDepth`
            # short-circuits on `training is False` ONLY, so `training=None` --
            # what a plain `model(inputs)` passes down -- DROPS PATHS, and two
            # `.keras` round-trip outputs then differ by up to 2.22 with every
            # weight bit-identical. `small` is the variant that gets `fit()`,
            # round trips and a frozen-vs-joint A/B run on it, i.e. the three
            # places where silent stochasticity corrupts a COMPARISON rather
            # than merely adding noise. Regularization is one keyword away
            # (`from_variant("small", drop_path_rate=0.1)`) and a caller that
            # raises it must then pass `training=` explicitly everywhere.
            # Do NOT "restore" the shipped variant's 0.1 here.
            # See decisions.md D-018.
            "drop_path_rate": 0.0, "dropout_rate": 0.0,
            "prompt_mlp_dropout": 0.0,
            # DIVERGENCE +16 on R7 (embed_dim/4 = 48). 48 forces either a
            # 3-head decoder or a non-integral head width, and the segmentation
            # head requires `d_model % num_groups == 0` at num_groups=8.
            "d_model": 64,
            # Unchanged from BOTH existing variants: the finest kept level is
            # 16*4 = 64, so masks are 64x64 from a 16x16 trunk grid.
            "scale_factors": (4.0, 2.0, 1.0, 0.5),
            "add_sam2_neck": False, "scalp": 1,
            # R8 EXACT on width (= embed_dim) and on head width (192/3 = 64).
            # text_depth is a DIVERGENCE -0.5: 0.75*6 = 4.5, floored.
            "text_width": 192, "text_depth": 4, "text_heads": 3,
            # DECISION plan-2026-08-05T124709-6c4fac48/D-019
            # context_length 32 is R15 EXACT (the positional table costs 6,144
            # params, so shrinking it buys nothing and would cap step 5's
            # phrases). vocab_size 512 is chosen against a WORKLOAD, not against
            # the reference's 49,408 CLIP BPE table, which is meaningless for a
            # fixed category-name -> id map: 512 clears COCO's 80 categories by
            # 6.4x and leaves room for reserved ids, at 512*192 = 98,304 params
            # (1.7% of the variant). `tiny`'s 64 UNDER-FITS COCO's 80 -- that is
            # the mistake this number exists not to repeat. KNOWN CEILING: 512
            # does NOT cover LVIS's 1,203 categories.
            # See decisions.md D-019.
            "context_length": 32, "vocab_size": 512,
            # num_queries: DIVERGENCE -168. Q must exceed the max GT instances
            # per image with headroom; 200 is sized for LVIS-scale crowding.
            # D-005 measured the matcher FLAT in Q, so this is not a speed
            # choice and Q can be raised without a wall-clock penalty.
            # decoder_layers: DIVERGENCE +1.875 on R14 (0.1875*6 = 1.125) --
            # iterative box refinement with ONE layer degenerates, there is no
            # chain left for D-113's stop_gradient to break.
            # decoder_heads: DIVERGENCE -16 on R9 (head width 16, not 32); at
            # d_model 64, R9 exactly would mean 2 heads, which is `tiny`'s
            # degenerate count.
            "num_queries": 32, "decoder_layers": 3, "decoder_heads": 4,
            # R10 EXACT (8*64), R11 EXACT, R12 EXACT.
            "dim_feedforward": 512, "d_proj": 64,
            "prompt_mlp_hidden_dim": 512,
            # R13 EXACT on groups (8 channels per group here vs the reference's
            # 32 -- a consequence of d_model, not a choice). seg_num_heads is a
            # DIVERGENCE -4: at 8 heads the mask head's per-head width would be
            # 8; 4 makes it 16, equal to the decoder's.
            "seg_num_heads": 4, "seg_num_groups": 8,
        },
        "tiny": {
            "img_size": 32, "patch_size": 4, "embed_dim": 16, "depth": 2,
            "num_heads": 2, "mlp_ratio": 4.0, "window_size": 4,
            "global_att_blocks": (1,), "pretrain_img_size": 16,
            # DECISION plan-2026-08-04T044628-4c240b4c/D-123
            # 0.0 here and 0.1 in `sam3`, and this is NOT a "small model needs
            # less regularization" choice. MEASURED: the repository's shared
            # `StochasticDepth` short-circuits on `training is False` only --
            # `training=None`, which is what a plain `model(inputs)` passes
            # down, DROPS PATHS. So at any non-zero rate this model is
            # stochastic under its most natural invocation, and two `.keras`
            # round-trip outputs then differ by O(1) with every one of its 217
            # weights bit-identical (measured: 2.27 on `pred_masks`, and a
            # weight-by-weight diff finds nothing). The shipped variant keeps
            # the reference's 0.1 because that is the reference's number; the
            # development variant is 0.0 so this package's own gate is
            # deterministic without every test having to remember the flag. Do
            # NOT "restore" 0.1 here, and do NOT conclude from a green
            # round-trip at this variant that `training=None` is inference --
            # `test_model.py::TestTrainingFlagTrap` pins the trap itself.
            # See decisions.md D-123.
            "drop_path_rate": 0.0,
            "d_model": 8, "scale_factors": (4.0, 2.0, 1.0, 0.5),
            "add_sam2_neck": False, "scalp": 1,
            "text_width": 16, "text_depth": 2, "text_heads": 2,
            "context_length": 8, "vocab_size": 64,
            "num_queries": 5, "decoder_layers": 2, "decoder_heads": 2,
            "dim_feedforward": 16, "dropout_rate": 0.0,
            "d_proj": 8, "prompt_mlp_hidden_dim": 16,
            "prompt_mlp_dropout": 0.0, "seg_num_heads": 2, "seg_num_groups": 2,
        },
    }

    def __init__(
            self,
            backbone: Union[Sam3ViTDetBackbone, Dict[str, Any]],
            neck: Union[Sam3DualViTDetNeck, Dict[str, Any]],
            text_encoder: Union[Sam3TextEncoder, Dict[str, Any]],
            transformer: Union[Sam3TransformerDecoder, Dict[str, Any]],
            dot_prod_scoring: Union[Sam3DotProductScoring, Dict[str, Any]],
            segmentation_head: Union[Sam3SegmentationHead, Dict[str, Any]],
            num_feature_levels: int = 1,
            scalp: int = 1,
            supervise_joint_box_scores: bool = False,
            detach_presence_in_joint_score: bool = False,
            joint_score_clamp: float = 10.0,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Sub-layers -- created unconditionally, built explicitly in build().
        for key, value in zip(COMPONENT_KEYS, (
                backbone, neck, text_encoder, transformer, dot_prod_scoring,
                segmentation_head)):
            setattr(self, key, value if isinstance(value, keras.layers.Layer)
                    else keras.saving.deserialize_keras_object(value))

        if num_feature_levels != 1:
            raise ValueError(
                f"num_feature_levels must be 1 in phase 1, got "
                f"{num_feature_levels}: the multi-level image memory needs the "
                f"level_start_index / spatial_shapes bookkeeping that only the "
                f"deferred vision-language fusion encoder consumes")
        if scalp < 0:
            raise ValueError(f"scalp must be >= 0, got {scalp}")
        self.num_feature_levels = int(num_feature_levels)
        self.scalp = int(scalp)
        self.supervise_joint_box_scores = bool(supervise_joint_box_scores)
        self.detach_presence_in_joint_score = bool(
            detach_presence_in_joint_score)
        self.joint_score_clamp = float(joint_score_clamp)

        levels = len(self.neck.scale_factors) - self.scalp
        if levels < 1:
            raise ValueError(
                f"scalp ({self.scalp}) discards every one of the neck's "
                f"{len(self.neck.scale_factors)} pyramid levels")
        self.kept_levels = levels
        self.d_model = int(self.neck.d_model)

        # DECISION plan-2026-08-04T044628-4c240b4c/D-125
        # The segmentation head is REQUIRED and the decoder's presence token
        # MUST be enabled. The reference makes both optional (`segmentation_head
        # =None`, `presence_token=False`), and this port deliberately does not.
        # The reason is the OUTPUT CONTRACT: `call` returns a fixed five-key
        # dict, and a key set that varies with the configuration is the shape a
        # downstream consumer indexes blindly. Both are enabled in every shipped
        # image configuration, so nothing reachable is lost. Do NOT "restore
        # flexibility" by making either optional without first deciding what the
        # missing keys become -- `None` is not a value a Keras output structure
        # can carry. See decisions.md D-125.
        if not self.transformer.use_presence_token:
            raise ValueError(
                "Sam3Image requires a decoder with use_presence_token=True: "
                "the decoder's presence token is the model's ONLY presence "
                "signal (the segmentation head has no presence mechanism at "
                "all), and `presence_logit` is part of the fixed output "
                "contract")
        for name, width in (
                ("neck", self.d_model),
                ("text_encoder", self.text_encoder.d_model),
                ("transformer", self.transformer.d_model),
                ("dot_prod_scoring", self.dot_prod_scoring.d_model),
                ("segmentation_head", self.segmentation_head.d_model)):
            if width != self.d_model:
                raise ValueError(
                    f"{name} has d_model={width} but the neck emits "
                    f"{self.d_model}; every component downstream of the neck "
                    f"must share one width")
        if self.neck.dim != self.backbone.embed_dim:
            raise ValueError(
                f"neck.dim ({self.neck.dim}) must equal backbone.embed_dim "
                f"({self.backbone.embed_dim})")
        if self.segmentation_head.upsampling_stages != levels - 1:
            raise ValueError(
                f"segmentation_head.upsampling_stages "
                f"({self.segmentation_head.upsampling_stages}) must equal the "
                f"number of KEPT pyramid levels minus one ({levels - 1})")

        grid = self.backbone.grid_size
        scale = self.neck.scale_factors[levels - 1]
        self.memory_grid = (int(grid * scale), int(grid * scale))
        if tuple(self.transformer.feat_size) != self.memory_grid:
            raise ValueError(
                f"transformer.feat_size {tuple(self.transformer.feat_size)} "
                f"must equal the image-memory grid {self.memory_grid} implied "
                f"by a {grid}x{grid} trunk at scale {scale}; boxRPB's bias is "
                f"built on that grid, so a mismatch is a silent wrong-geometry "
                f"bias rather than a shape error")

        logger.info(
            f"Sam3Image: trunk {grid}x{grid}x{self.backbone.embed_dim}, "
            f"pyramid {levels} of {len(self.neck.scale_factors)} levels, "
            f"memory {self.memory_grid}, d_model={self.d_model}, "
            f"joint_scores={self.supervise_joint_box_scores}")

    # -----------------------------------------------------------------
    # variants
    # -----------------------------------------------------------------

    @classmethod
    def from_variant(cls, variant: str, **kwargs: Any) -> "Sam3Image":
        """Construct every component from one variant table entry.

        :param variant: ``'sam3'`` (the released configuration), ``'small'`` (a
            trainable-on-12-GB geometry, NOT a published size) or ``'tiny'``
            (a development geometry that is NOT a published size).
        :type variant: str
        :param kwargs: Explicit overrides. A table key is overridden in the
            table; anything else is forwarded to ``__init__``. Passing ``None``
            explicitly is the same as omitting the argument (S-3).
        :type kwargs: Any
        :return: The configured model.
        :rtype: Sam3Image
        :raises ValueError: If ``variant`` is unknown.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown SAM 3 variant '{variant}'. Available: "
                f"{sorted(cls.MODEL_VARIANTS)}. Only "
                f"'sam3' is a published size; 'small' and 'tiny' are this "
                f"package's own development geometries, and the other published "
                f"SAM 3 configurations were never read by this implementation "
                f"and are not invented here.")
        table = dict(cls.MODEL_VARIANTS[variant])
        overrides = {k: v for k, v in kwargs.items() if v is not None}
        table.update({k: v for k, v in overrides.items() if k in table})
        model_kwargs = {k: v for k, v in overrides.items() if k not in table}

        levels = len(table["scale_factors"]) - table["scalp"]
        grid = table["img_size"] // table["patch_size"]
        memory = int(grid * table["scale_factors"][levels - 1])
        backbone = Sam3ViTDetBackbone(
            img_size=table["img_size"], patch_size=table["patch_size"],
            embed_dim=table["embed_dim"], depth=table["depth"],
            num_heads=table["num_heads"], mlp_ratio=table["mlp_ratio"],
            window_size=table["window_size"],
            global_att_blocks=table["global_att_blocks"],
            drop_path_rate=table["drop_path_rate"],
            pretrain_img_size=table["pretrain_img_size"])
        neck = Sam3DualViTDetNeck(
            dim=table["embed_dim"], d_model=table["d_model"],
            scale_factors=table["scale_factors"],
            add_sam2_neck=table["add_sam2_neck"])
        text_encoder = Sam3TextEncoder(
            d_model=table["d_model"], width=table["text_width"],
            depth=table["text_depth"], num_heads=table["text_heads"],
            context_length=table["context_length"],
            vocab_size=table["vocab_size"])
        transformer = Sam3TransformerDecoder(
            d_model=table["d_model"], num_heads=table["decoder_heads"],
            num_layers=table["decoder_layers"],
            num_queries=table["num_queries"], feat_size=(memory, memory),
            dim_feedforward=table["dim_feedforward"],
            dropout_rate=table["dropout_rate"])
        dot_prod_scoring = Sam3DotProductScoring(
            d_model=table["d_model"], d_proj=table["d_proj"],
            prompt_mlp_hidden_dim=table["prompt_mlp_hidden_dim"],
            prompt_mlp_dropout=table["prompt_mlp_dropout"])
        segmentation_head = Sam3SegmentationHead(
            d_model=table["d_model"], upsampling_stages=levels - 1,
            num_heads=table["seg_num_heads"],
            num_groups=table["seg_num_groups"])

        logger.info("Creating Sam3Image variant '%s'", variant)
        return cls(
            backbone=backbone, neck=neck, text_encoder=text_encoder,
            transformer=transformer, dot_prod_scoring=dot_prod_scoring,
            segmentation_head=segmentation_head, scalp=table["scalp"],
            **model_kwargs)

    # -----------------------------------------------------------------
    # pyramid plumbing -- one owner (this class), so `@staticmethod`, not a
    # module-level function (D-109 / D-114).
    # -----------------------------------------------------------------

    def _scalped(self, features: Sequence[Any]) -> List[Any]:
        """Drop the ``scalp`` COARSEST levels of a finest-first pyramid."""
        features = list(features)
        return features[:len(features) - self.scalp] if self.scalp else features

    @staticmethod
    def _flatten(feature: Any) -> Any:
        """Fold a channels-last map's spatial axes into one token axis."""
        shape = ops.shape(feature)
        return ops.reshape(feature, (shape[0], shape[1] * shape[2], shape[3]))

    @staticmethod
    def _build_once(component: keras.layers.Layer, *shapes: Any) -> None:
        """Build a component unless it is already built.

        MEASURED necessity, not defensive style: on ``.keras`` load, Keras
        deserializes each component and rebuilds it from its own recorded build
        config BEFORE :meth:`build_from_config` runs. Calling ``build`` again on
        such a component raises ``ValueError: You cannot add new elements of
        state ... to a layer that is already built`` for any component whose
        ``build`` lacks its own re-entry guard. Do NOT remove this check.

        :param component: The sub-layer to build.
        :type component: keras.layers.Layer
        :param shapes: Positional shape arguments for that component's
            ``build``.
        :type shapes: Any
        """
        if not component.built:
            component.build(*shapes)

    # -----------------------------------------------------------------
    # build
    # -----------------------------------------------------------------

    def build(self, input_shape: Optional[Any] = None) -> None:
        """Build every component explicitly, from the stored configuration.

        Nothing here is left to lazy first-call construction: a subclassed
        model whose sub-layers build lazily restores an INCOMPLETE weight set
        from a ``.keras`` file, with no exception and no shape symptom.

        :param input_shape: Ignored; every shape is derived from the components'
            own configuration, which is what makes ``build(None)`` legal.
        :type input_shape: Optional[Any]
        """
        if self.built:
            return
        side = self.backbone.img_size
        image_shape = (None, side, side, self.backbone.in_channels)
        self._build_once(self.backbone, image_shape)
        trunk_shape = self.backbone.compute_output_shape(image_shape)
        self._build_once(self.neck, trunk_shape)
        pyramid = self._scalped(
            self.neck.compute_output_shape(trunk_shape)["sam3_features"])
        memory_shape = (None, pyramid[-1][1] * pyramid[-1][2], self.d_model)

        token_shape = (None, self.text_encoder.context_length)
        self._build_once(self.text_encoder, token_shape)
        prompt_shape = self.text_encoder.compute_output_shape(token_shape)
        self._build_once(self.transformer, memory_shape, prompt_shape)
        hidden_shape = (self.transformer.num_layers, None,
                        self.transformer.num_queries, self.d_model)
        self._build_once(self.dot_prod_scoring, hidden_shape, prompt_shape,
                         token_shape)
        self._build_once(self.segmentation_head, pyramid, hidden_shape,
                         memory_shape, prompt_shape)
        super().build(input_shape)

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build before Keras restores weights, so every variable exists.

        :param config: Ignored -- :meth:`build` derives everything from the
            components' own configuration.
        :type config: Dict[str, Any]
        """
        del config
        if not self.built:
            self.build(None)

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------

    def call(
            self, inputs: Dict[str, Any], training: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Detect and segment every instance a text prompt names.

        :param inputs: ``{'image': (B, img_size, img_size, 3), 'token_ids':
            (B, seq), 'token_padding_mask': (B, seq)}``. The padding mask is
            ``True`` AT PADDING -- a key-padding mask, the opposite polarity to
            the causal KEEP mask the text tower builds internally. It is
            optional; omitting it treats every token as valid.
        :type inputs: Dict[str, Any]
        :param training: Keras training flag, forwarded to every component.
            **Pass ``False`` explicitly for inference on the ``sam3`` variant.**
            That variant carries the reference's ``drop_path_rate=0.1`` and the
            shared ``StochasticDepth`` short-circuits on ``training is False``
            ONLY, so the ``training=None`` a plain ``model(inputs)`` passes down
            drops paths and the call is NOT deterministic (D-123; the trap is
            executed by ``test_model.py::TestTrainingFlagTrap``). ``tiny`` sets
            the rate to 0.0 and is unaffected.
        :type training: Optional[bool]
        :return: ``pred_logits`` ``(B, num_queries, 1)``, ``pred_boxes``
            ``(B, num_queries, 4)`` normalized ``cxcywh``, ``pred_masks``
            ``(B, num_queries, H, W)`` at the FINEST pyramid level,
            ``presence_logit`` ``(B, 1)``, and ``semantic_seg``
            ``(B, H, W, 1)``. Every key is always present.
        :rtype: Dict[str, Any]
        :raises ValueError: If ``'image'`` or ``'token_ids'`` is absent.
        """
        outputs_class, outputs_coord, presence_logits, seg = (
            self._forward_stacks(inputs, training=training))
        return {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "pred_masks": seg["pred_masks"],
            "semantic_seg": seg["semantic_seg"],
            "presence_logit": presence_logits[-1],
        }

    def _forward_stacks(
            self, inputs: Dict[str, Any], training: Optional[bool] = None,
    ) -> Tuple[Any, Any, Any, Dict[str, Any]]:
        """Run the whole forward pass and keep every layer's predictions.

        The entire body of :meth:`call` except its final ``[-1]`` slicing lives
        here, so the per-layer quantities an auxiliary-loss training phase needs
        are produced exactly once and the reported last-layer quantities are
        rows of the SAME tensors -- not a second, separately computed forward
        pass that could drift from it.

        :param inputs: As :meth:`call`.
        :type inputs: Dict[str, Any]
        :param training: As :meth:`call`.
        :type training: Optional[bool]
        :return: ``(outputs_class, outputs_coord, presence_logits, seg)``. The
            first three are stacked over decoder layers with the layer axis
            FIRST: ``(num_layers, B, num_queries, 1)``,
            ``(num_layers, B, num_queries, 4)`` and ``(num_layers, B, 1)``. The
            fourth is the segmentation head's own output dict, which has no
            layer axis -- that head consumes the whole hidden stack and emits
            one set of masks.
        :rtype: Tuple[Any, Any, Any, Dict[str, Any]]
        :raises ValueError: If ``'image'`` or ``'token_ids'`` is absent.
        """
        for key in ("image", "token_ids"):
            if key not in inputs:
                raise ValueError(f"Sam3Image.call requires inputs['{key}']")
        padding_mask = inputs.get("token_padding_mask")

        neck_out = self.neck(
            self.backbone(inputs["image"], training=training), training=training)
        pyramid = self._scalped(neck_out["sam3_features"])
        memory = self._flatten(pyramid[-1])
        memory_pos = self._flatten(self._scalped(neck_out["sam3_pos"])[-1])
        prompt = self.text_encoder(inputs["token_ids"], training=training)

        hidden, anchors, presence_logits, _ = self.transformer(
            memory, memory_text=prompt, text_padding_mask=padding_mask,
            memory_pos=memory_pos, training=training)

        outputs_class = self.dot_prod_scoring(
            hidden, prompt, padding_mask, training=training)
        # The decoder returns the anchor each layer CONSUMED, so the LAST
        # layer's refinement has not been applied yet. Re-applying the shared
        # box head to every layer's hidden state reproduces the decoder's own
        # per-layer refinement for layers 0..L-2 and produces layer L-1's, which
        # is the box this model reports.
        delta = Sam3TransformerDecoder._run_mlp(self.transformer.bbox_embed,
                                                hidden)
        outputs_coord = ops.sigmoid(
            delta + Sam3TransformerDecoder._inverse_sigmoid(anchors))

        if self.supervise_joint_box_scores:
            outputs_class = self._fuse(
                outputs_class, presence_logits, self.joint_score_clamp,
                self.detach_presence_in_joint_score)

        seg = self.segmentation_head(
            pyramid, hidden, memory, prompt=prompt,
            prompt_padding_mask=padding_mask, training=training)
        return outputs_class, outputs_coord, presence_logits, seg

    @staticmethod
    def _fuse(
            outputs_class: Any, presence_logits: Any, clamp: float,
            detach: bool = False,
    ) -> Any:
        """Fuse per-image presence into per-query class logits.

        :param outputs_class: ``(num_layers, batch, num_queries, 1)`` class
            logits.
        :type outputs_class: Any
        :param presence_logits: ``(num_layers, batch, 1)`` presence logits.
        :type presence_logits: Any
        :param clamp: Symmetric bound applied after the re-logit.
        :type clamp: float
        :param detach: Whether the presence probability is detached first.
        :type detach: bool
        :return: Fused class logits, shaped like ``outputs_class``.
        :rtype: Any
        """
        # DECISION plan-2026-08-04T044628-4c240b4c/D-122
        # The fusion multiplies PROBABILITIES and re-logits the product. Do
        # NOT "simplify" it to a multiply in logit space: the two agree only
        # along a thin curve (measured minimum separation 0.056 nats over a
        # 7x7 probe grid, versus 1.099 at the origin and 5.41 at
        # (class=-2, presence=-1)), so a wrong port is a silent value defect
        # with correct shapes everywhere.
        #
        # MEASURED, and it corrects two premises at once. `_inverse_sigmoid`
        # guards its argument with `eps=1e-3`, which bounds its OUTPUT to
        # +-log(1/eps - ...) = +-6.9078 in float64. The `clamp` below is
        # therefore a provable NO-OP at the reference's own eps: over a
        # [-40, 40]^2 grid of (class, presence) logits the clamped and
        # unclamped results differ by EXACTLY 0.0. It ships anyway, because
        # it is the reference's expression and because it becomes live the
        # moment eps shrinks (at eps=1e-7 the range is +-16.118 and the
        # clamp binds, moving the result by up to 3.09). So: do NOT delete
        # the clamp as dead code, and do NOT "fix" eps to make the clamp
        # matter -- eps is the literal that sets the saturation floor, and
        # a saturated presence drives the class logit to the EPS floor
        # (-6.9078), not to the clamp floor (-10.0). Both facts are pinned
        # by `test_model.py::TestFusionOracle`. See decisions.md D-122.
        #
        # The presence multiplied here is the DECODER's, the only presence
        # signal in this package; the segmentation head has none at all.
        presence = ops.sigmoid(presence_logits)
        if detach:
            presence = ops.stop_gradient(presence)
        return ops.clip(
            Sam3TransformerDecoder._inverse_sigmoid(
                ops.sigmoid(outputs_class) * ops.expand_dims(presence, 2)),
            -clamp, clamp)

    def compute_output_shape(
            self, input_shape: Optional[Any] = None
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """Return every output shape, derived from stored config only.

        :param input_shape: Ignored; the batch axis is reported as ``None``
            because every spatial extent is fixed by the components.
        :type input_shape: Optional[Any]
        :return: One shape per output key.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        queries = self.transformer.num_queries
        finest = int(self.backbone.grid_size * self.neck.scale_factors[0])
        return {
            "pred_logits": (None, queries, 1),
            "pred_boxes": (None, queries, 4),
            "pred_masks": (None, queries, finest, finest),
            "semantic_seg": (None, finest, finest, 1),
            "presence_logit": (None, 1),
        }

    def get_config(self) -> Dict[str, Any]:
        """Return every ``__init__`` parameter.

        :return: Serializable configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({key: keras.saving.serialize_keras_object(
            getattr(self, key)) for key in COMPONENT_KEYS})
        config.update({
            "num_feature_levels": self.num_feature_levels,
            "scalp": self.scalp,
            "supervise_joint_box_scores": self.supervise_joint_box_scores,
            "detach_presence_in_joint_score":
                self.detach_presence_in_joint_score,
            "joint_score_clamp": self.joint_score_clamp,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Sam3Image":
        """Reconstruct a model from :meth:`get_config`.

        :param config: The configuration dictionary.
        :type config: Dict[str, Any]
        :return: The reconstructed model.
        :rtype: Sam3Image
        """
        for key in COMPONENT_KEYS:
            config[key] = keras.saving.deserialize_keras_object(config[key])
        return cls(**config)

# ---------------------------------------------------------------------
