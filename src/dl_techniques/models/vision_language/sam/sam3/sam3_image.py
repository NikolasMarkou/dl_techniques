"""Sam3Image, concept-promptable segmentation: a text phrase in, every instance it names out.

SAM 1 and SAM 2 take geometric prompts naming one instance already located.
SAM 3 takes a noun phrase naming a concept, so the model must find every
instance of it or report none, which a per-prompt mask decoder cannot
express. It is a DETR-style set predictor instead: a fixed bank of object
queries is decoded in parallel, each emitting one box, one mask, and one
scalar logit, matched to the ground truth by a Hungarian assignment in the
training wrapper.

The vocabulary is open because that scalar logit is not a row of a class
table: :class:`~.model_misc.Sam3DotProductScoring` takes the scaled dot
product of the query and the pooled text prompt, so swapping the prompt
swaps the class with no retraining. A `presence` token rides the decoder's
query sequence, giving one image-level logit for whether the concept occurs
at all, independent of which query found it.

The image path is plain-ViT detection: one backbone feature map resampled
into a four-scale pyramid by a dual (detector + tracker), independently
weighted neck, feeding a MaskFormer segmentation head and a three-sub-block
decoder (query self-attention, text cross-attention, image cross-attention
with box-conditioned relative position bias). Boxes refine iteratively in
logit space as in DAB-DETR; the final box comes from re-applying the shared
box head after the decoder stack, not from the decoder's own last-layer
output, so :meth:`call_per_layer` is the supported route to per-layer boxes.

An opt-in `query_selection=True` adds a DINO-style mixed-selection head that
replaces the decoder's learned reference-point table with the top-scoring
image-memory positions; off by default, it adds no weight when disabled.

This is a phase-1 architecture: it omits the reference's vision-language
early-fusion encoder (image memory and prompt go straight to the decoder),
the exemplar/geometry prompt path, and DAC query doubling. As with SAM 2, no
pretrained weights ship or ever will (the SAM License is incompatible with
this repository's GPL-3.0); this is a reimplementation from the paper with
no accuracy claim. Only the `sam3` variant is a published size; `small` and
`tiny` are this package's own development geometries.

References:
    - Ravi et al., 2025. SAM 3: Segment Anything with Concepts.
    - Carion et al., 2020. End-to-End Object Detection with Transformers.
      (https://arxiv.org/abs/2005.12872)
    - Liu et al., 2022. DAB-DETR: Dynamic Anchor Boxes are Better Queries for
      DETR. (https://arxiv.org/abs/2201.12329)
    - Zhang et al., 2022. DINO: DETR with Improved DeNoising Anchor Boxes for
      End-to-End Object Detection. (https://arxiv.org/abs/2203.03605)
    - Liu et al., 2023. Grounding DINO: Marrying DINO with Grounded Pre-Training
      for Open-Set Object Detection. (https://arxiv.org/abs/2303.05499)
    - Li et al., 2022. Exploring Plain Vision Transformer Backbones for Object
      Detection. (https://arxiv.org/abs/2203.16527)
    - Cheng et al., 2021. Per-Pixel Classification is Not All You Need for
      Semantic Segmentation. (https://arxiv.org/abs/2107.06278)
    - Radford et al., 2021. Learning Transferable Visual Models From Natural
      Language Supervision. (https://arxiv.org/abs/2103.00020)
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import keras
from keras import ops

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.vision_language.sam.sam3.decoder import Sam3TransformerDecoder
from dl_techniques.models.vision_language.sam.sam3.maskformer_segmentation import Sam3SegmentationHead
from dl_techniques.models.vision_language.sam.sam3.model_misc import Sam3DotProductScoring
from dl_techniques.models.vision_language.sam.sam3.necks import Sam3DualViTDetNeck
from dl_techniques.models.vision_language.sam.sam3.query_selection import Sam3EncoderQuerySelection
from dl_techniques.models.vision_language.sam.sam3.text_encoder_ve import Sam3TextEncoder
from dl_techniques.models.vision_language.sam.sam3.vitdet import Sam3ViTDetBackbone
from dl_techniques.utils.keras_registration import register_dl_technique

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


@register_dl_technique("dl_techniques.models.sam3.sam3_image")
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
    :param query_selection: Whether DINO-style **mixed** encoder query selection
        is active. Default ``False``, at which value this model is behaviourally
        identical to the one that shipped before the flag existed: no head is
        created, no weight is added, and the decoder is called by the same
        expression with ``reference_boxes=None``. At ``True`` a
        :class:`~dl_techniques.models.vision_language.sam.sam3.query_selection.Sam3EncoderQuerySelection`
        head scores every image-memory position and its top ``num_queries``
        boxes become the decoder's INITIAL ``reference_boxes``, detached. Query
        CONTENT is untouched -- that is what makes it *mixed*.
    :type query_selection: bool
    :param prompt_conditioned_queries: Whether that proposal head READS the
        text prompt. Default ``False``, at which value no weight is added and
        the head is the prompt-blind one that shipped before. At ``True`` the
        pooled prompt FiLM-modulates the image memory before the head scores
        it, so the top-``num_queries`` SELECTION becomes prompt-dependent.
        Requires ``query_selection=True``; setting it alone raises rather than
        becoming a silent no-op.
    :type prompt_conditioned_queries: bool
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
            "prompt_mlp_dropout_rate": 0.1, "seg_num_heads": 8, "seg_num_groups": 8,
        },
        # DECISION plan-2026-08-05T124709-6c4fac48/D-017: `small` is not a
        # published SAM 3 size -- every field is derived field-by-field from
        # the released configuration's own ratios (R1-R15), a set of ratios
        # rather than one smaller config since the reference publishes only
        # one size. Do not round a field back to a nicer number; five values
        # here are the reference's exact numbers. See decisions.md.
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
            # DECISION plan-2026-08-05T124709-6c4fac48/D-018: 0.0 for all
            # three rates here, not the reference's 0.1.
            # StochasticDepth drops paths under training=None (D-123), and `small` is the variant that gets fit()/round-trip/A-B comparisons this would corrupt. See decisions.md.
            "drop_path_rate": 0.0, "dropout_rate": 0.0,
            "prompt_mlp_dropout_rate": 0.0,
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
            # DECISION plan-2026-08-05T124709-6c4fac48/D-019: vocab_size=512,
            # sized against a workload (COCO's 80 categories), not the
            # reference's 49,408-token CLIP BPE table, which is meaningless
            # for a fixed category-name-to-id map. Does not cover LVIS's
            # 1,203 categories. See decisions.md.
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
            # DECISION plan-2026-08-04T044628-4c240b4c/D-123: 0.0 here, 0.1 in
            # `sam3` -- StochasticDepth drops paths under training=None, so a
            # non-zero rate makes .keras round trips differ by O(1) with
            # every weight bit-identical. See decisions.md.
            "drop_path_rate": 0.0,
            "d_model": 8, "scale_factors": (4.0, 2.0, 1.0, 0.5),
            "add_sam2_neck": False, "scalp": 1,
            "text_width": 16, "text_depth": 2, "text_heads": 2,
            "context_length": 8, "vocab_size": 64,
            "num_queries": 5, "decoder_layers": 2, "decoder_heads": 2,
            "dim_feedforward": 16, "dropout_rate": 0.0,
            "d_proj": 8, "prompt_mlp_hidden_dim": 16,
            "prompt_mlp_dropout_rate": 0.0, "seg_num_heads": 2, "seg_num_groups": 2,
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
            query_selection: bool = False,
            prompt_conditioned_queries: bool = False,
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
        self.query_selection = bool(query_selection)
        self.prompt_conditioned_queries = bool(prompt_conditioned_queries)
        if self.prompt_conditioned_queries and not self.query_selection:
            raise ValueError(
                "prompt_conditioned_queries=True requires query_selection="
                "True: the flag conditions the ENCODER QUERY SELECTION head, "
                "and with query selection off no such head exists, so the "
                "flag would be a silent no-op -- a run that reports the arm's "
                "name while training the control")

        levels = len(self.neck.scale_factors) - self.scalp
        if levels < 1:
            raise ValueError(
                f"scalp ({self.scalp}) discards every one of the neck's "
                f"{len(self.neck.scale_factors)} pyramid levels")
        self.kept_levels = levels
        self.d_model = int(self.neck.d_model)

        # DECISION plan-2026-08-04T044628-4c240b4c/D-125: segmentation head is
        # required and the decoder's presence token must be enabled, unlike
        # the reference which makes both optional.
        # call() returns a fixed five-key dict; a key set that varies with config is the shape a downstream consumer indexes blindly. See decisions.md.
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

        # The head is created HERE, in `__init__`, and never on first call. A
        # subclassed `keras.Model` whose sub-layers materialize lazily restores
        # an INCOMPLETE weight set from a `.keras` file with no exception and no
        # shape symptom -- the same reason `build()` below builds every
        # component explicitly. At `query_selection=False` the attribute is
        # `None`, so the flag-off model owns exactly the weights it owned before
        # this flag existed.
        self.query_selection_head = None
        if self.query_selection:
            self.query_selection_head = Sam3EncoderQuerySelection(
                d_model=self.d_model,
                num_queries=self.transformer.num_queries,
                feat_size=self.memory_grid,
                prompt_conditioned=self.prompt_conditioned_queries,
                name="query_selection_head")

        logger.info(
            f"Sam3Image: trunk {grid}x{grid}x{self.backbone.embed_dim}, "
            f"pyramid {levels} of {len(self.neck.scale_factors)} levels, "
            f"memory {self.memory_grid}, d_model={self.d_model}, "
            f"joint_scores={self.supervise_joint_box_scores}, "
            f"query_selection={self.query_selection}, "
            f"prompt_conditioned_queries={self.prompt_conditioned_queries}")

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
            prompt_mlp_dropout_rate=table["prompt_mlp_dropout_rate"])
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
        if self.query_selection_head is not None:
            # ONE shape, at BOTH values of `prompt_conditioned_queries`: the
            # head's FiLM projection reads the POOLED prompt, whose width is
            # `d_model`, so it needs no prompt shape -- and Keras would refuse
            # a two-argument `build` here whose first argument is not named
            # after `call`'s first argument. See the anchor on that `build`.
            self._build_once(self.query_selection_head, memory_shape)
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

    def call_per_layer(
            self, inputs: Dict[str, Any], training: Optional[bool] = None,
            include_proposals: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return one prediction dict PER decoder layer, MAIN layer FIRST.

        Interface contract: the input is exactly :meth:`call`'s. The return is a
        list of ``num_layers`` dicts, each carrying the same five keys
        :meth:`call` returns. **Element 0 is the LAST decoder layer** -- the one
        :meth:`call` reports -- and is bit-equal to :meth:`call`'s output field
        by field; elements ``1..L-1`` are decoder layers ``0..L-2`` in order,
        which is the reference's ``aux_outputs`` order. The list is therefore in
        SUPERVISION order, not in depth order, so a consumer that packs it
        front-to-back gets the main block first by construction rather than by
        remembering to reverse. It raises exactly what :meth:`call` raises.

        ``pred_masks`` and ``semantic_seg`` are the SAME tensors in every
        element: the segmentation head consumes the whole hidden stack and emits
        one set of masks, so there is no per-layer mask to report. They are
        repeated rather than dropped so every element has one shape, and the
        consumer decides what to do with them (the training wrapper zero-fills
        the auxiliary blocks' mask channels -- see decisions.md D-005).

        Both production consumers run only at ``deep_supervision=True``:
        :meth:`~dl_techniques.models.vision_language.sam.sam3.training_model.Sam3TrainingModel.call`,
        and ``train.sam3.train_sam3.evaluate_sam3``, which packs the same blocks
        so the compiled loss's row stride agrees with the tensor it is handed.

        When ``include_proposals`` is ``True`` AND ``query_selection`` is on,
        ONE further element is appended **LAST**: the encoder query selection
        head's own block, in the same five-key shape, with ``pred_logits`` the
        selected objectness ``(B, Q, 1)``, ``pred_boxes`` the selected boxes
        ``(B, Q, 4)``, and ``presence_logit`` the ``max`` over the selected
        objectness ``(B, 1)``. Its ``pred_masks`` / ``semantic_seg`` are the
        same shared segmentation tensors every other element carries -- the
        packer zero-fills an auxiliary block's mask channels regardless (see
        ``pack_predictions``' D-005 anchor), so they are repeated for shape
        uniformity, not to be supervised. LAST is the position, not first: the
        decoder blocks keep the row offsets they have today, so a packed tensor
        built with the flag on is the flag-off tensor with one block appended.
        With either the argument or the flag off, the returned list is exactly
        what it has always been.

        :param inputs: As :meth:`call`.
        :type inputs: Dict[str, Any]
        :param training: As :meth:`call`.
        :type training: Optional[bool]
        :param include_proposals: Whether to append the encoder block. Ignored
            when ``query_selection`` is off -- there are no proposals to append.
        :type include_proposals: bool
        :return: ``num_layers`` output dicts, last decoder layer first, plus the
            encoder block when both the argument and the flag are on.
        :rtype: List[Dict[str, Any]]
        :raises ValueError: If ``'image'`` or ``'token_ids'`` is absent.
        """
        outputs_class, outputs_coord, presence_logits, seg, proposals = (
            self._forward_all(inputs, training=training))
        num_layers = int(outputs_class.shape[0])
        order = [num_layers - 1] + list(range(num_layers - 1))
        blocks = [{
            "pred_logits": outputs_class[index],
            "pred_boxes": outputs_coord[index],
            "pred_masks": seg["pred_masks"],
            "semantic_seg": seg["semantic_seg"],
            "presence_logit": presence_logits[index],
        } for index in order]
        if include_proposals and proposals is not None:
            blocks.append({
                "pred_logits": proposals["selected_objectness"],
                "pred_boxes": proposals["selected_boxes"],
                "pred_masks": seg["pred_masks"],
                "semantic_seg": seg["semantic_seg"],
                "presence_logit": ops.max(
                    proposals["selected_objectness"], axis=1),
            })
        return blocks

    def _forward_stacks(
            self, inputs: Dict[str, Any], training: Optional[bool] = None,
    ) -> Tuple[Any, Any, Any, Dict[str, Any]]:
        """Run the whole forward pass and keep every layer's predictions.

        The entire body of :meth:`call` except its final ``[-1]`` slicing lives
        in :meth:`_forward_all`, so the per-layer quantities an auxiliary-loss
        training phase needs are produced exactly once and the reported
        last-layer quantities are rows of the SAME tensors -- not a second,
        separately computed forward pass that could drift from it.

        This is the four-value view of :meth:`_forward_all`, kept as its own
        name because its four-tuple contract predates encoder query selection
        and every consumer of it is indifferent to the proposals. There is no
        second forward pass and no duplicated body: it is one delegation.

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
        return self._forward_all(inputs, training=training)[:4]

    def _forward_all(
            self, inputs: Dict[str, Any], training: Optional[bool] = None,
    ) -> Tuple[Any, Any, Any, Dict[str, Any], Optional[Dict[str, Any]]]:
        """Run the whole forward pass, proposals included.

        Interface contract: the input is exactly :meth:`call`'s. The return is
        :meth:`_forward_stacks`'s four values followed by the encoder query
        selection head's own output dict -- or ``None`` when
        ``query_selection`` is off, which is the value that also makes the
        decoder call below take its default (learned, image-independent)
        reference path. It raises exactly what :meth:`call` raises.

        :param inputs: As :meth:`call`.
        :type inputs: Dict[str, Any]
        :param training: As :meth:`call`.
        :type training: Optional[bool]
        :return: ``(outputs_class, outputs_coord, presence_logits, seg,
            proposals)``; see :meth:`_forward_stacks` for the first four and
            :meth:`~dl_techniques.models.vision_language.sam.sam3.query_selection.Sam3EncoderQuerySelection.call`
            for the fifth.
        :rtype: Tuple[Any, Any, Any, Dict[str, Any], Optional[Dict[str, Any]]]
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

        # ONE reference expression, computed once, so the flag-off path cannot
        # drift from the flag-on one: `None` is exactly what the decoder's own
        # `if reference_boxes is None` default branch consumes, so with the flag
        # off the call below is the call that shipped before this flag existed.
        proposals = None
        reference_boxes = None
        if self.query_selection_head is not None:
            # The prompt is passed at this ONE call site whatever
            # `prompt_conditioned_queries` says, and the HEAD's own flag is the
            # single gate on whether it is read. A second gate here would be a
            # second place for the two to disagree; with the head's flag off
            # these two arguments are inert and the flag-off path is the path
            # that shipped before this flag existed.
            proposals = self.query_selection_head(
                memory, prompt=prompt, prompt_padding_mask=padding_mask,
                training=training)

            # DECISION plan-2026-08-06T185813-fd80240f/D-006: proposals enter
            # the decoder detached; never remove this stop_gradient.
            # The head is supervised by its own packed block already; removing this reopens a credit-assignment path through the decoder with no shape/dtype symptom, only silent gradient changes. See decisions.md.
            reference_boxes = ops.stop_gradient(proposals["selected_boxes"])

        # `tgt` is deliberately NOT passed: query CONTENT stays the decoder's
        # learned `query_embed` table and only the POSITIONAL part comes from
        # the image. That is DINO's *mixed* query selection, and it is what
        # "query selection" means in this package (invariant I-2). Passing `tgt`
        # here would silently redefine the term.
        hidden, anchors, presence_logits, _ = self.transformer(
            memory, memory_text=prompt, text_padding_mask=padding_mask,
            memory_pos=memory_pos, reference_boxes=reference_boxes,
            training=training)

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
        return outputs_class, outputs_coord, presence_logits, seg, proposals

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
        # DECISION plan-2026-08-04T044628-4c240b4c/D-122: multiply
        # probabilities and re-logit the product; never multiply in logit
        # space, which agrees with this only along a thin curve. The clamp
        # below is a no-op at the reference's own eps=1e-3 but becomes live
        # if eps shrinks -- keep it. See decisions.md.
        #
        # The presence multiplied here is the decoder's, the only presence
        # signal in this package.
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
            "query_selection": self.query_selection,
            "prompt_conditioned_queries": self.prompt_conditioned_queries,
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
