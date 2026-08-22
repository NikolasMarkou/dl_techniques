"""
Streaming video segmentation: SAM's promptable decoder conditioned on a memory bank.

SAM 1 segments a frame from a prompt. Tracking that object through a video is a
different problem, because the thing that identifies the object on frame 200 is
not a click but frame 0 -- and everything in between. Running SAM 1 per frame
loses identity the moment the object is occluded, deforms, or leaves and
re-enters. SAM 2's answer is to keep the promptable decoder and insert a
learned recurrence in front of it: before the decoder sees a frame's pixel
features, those features cross-attend to a bounded memory of the recent past.
The prompt is then optional on every frame after the first, because the memory
carries what the prompt used to say.

That memory is deliberately narrow. A frame's prediction is compressed by the
memory encoder into a `mem_dim`-wide spatial map -- 8 channels at `tiny`, 64 at
`hiera_l`, against a `hidden_dim` of 256 for the pixel stream -- so storing
several frames costs a fraction of storing their features. Memory attention
therefore runs cross-attention with keys and values at `kv_in_dim = mem_dim`
while its queries stay at `d_model = hidden_dim`, an asymmetry the component
agreement check enforces at construction. Alongside the spatial memory the bank
also carries *object pointers*: single `hidden_dim`-wide vectors, one per
remembered frame, produced by a 3-layer MLP from whichever decoder output token
the model's own IoU head rated best. The spatial stream says where the object
was; the pointer stream says what it looked like.

The image side is a Hiera trunk -- hierarchical, four stages, window attention,
width and head count doubling while the grid halves via a max-pool on the
attention *queries* -- feeding an FPN neck that produces four `d_model` levels
plus a sine positional map each, of which the coarsest `scalp` levels are
dropped. The retained stride-16 level is the memory grid; the two finer levels
are handed to the mask decoder as high-resolution skips, which is why
`use_high_res_features=True` is required rather than optional. The decoder
itself is a sibling of SAM 1's, not a subclass: it prepends an object-score
token to the token block, returns four values instead of two, and can fall back
between its multimask and single-mask outputs on a stability score.

Temporal order is carried by exactly one mechanism, and this is the constraint
most likely to be broken by a well-meaning simplification. Memory attention's
rotary embedding is spatial-only and is broadcast *identically* across every
stacked memory frame, so it cannot distinguish frame `t-1` from frame `t-6`.
The distinction comes from `maskmem_tpos_enc`, a learned
`(num_maskmem, 1, 1, mem_dim)` table that lives on this class while the memory
bank returns only slot *indices* into it. The object-pointer tail of the memory
sequence is separate again: it gets a fixed sine encoding of how many frames
away each pointer is, projected down to `mem_dim` by `obj_ptr_tpos_proj` so it
cannot collide with the spatial positional encoding. Folding either into the
rotary embedding, or zeroing the pointer tail, yields a model that trains
happily and is temporally blind.

Occlusion is likewise marked twice, and both marks are load-bearing.
`no_obj_ptr` blends into the pointer stream by the object score; the *separate*
`no_obj_embed_spatial` is added into the encoded spatial memory in proportion to
`1 - is_appearing`. On top of those two marks, `_suppress_absent_object`
*erases* the mask itself, overwriting every logit with `NO_OBJ_SCORE = -1024.0`
on any row the score head calls empty. That value is transcribed, not chosen:
the memory encoder's `sigmoid(x) * 20 - 10` saturates it to exactly `-10`, and
`-1024` is representable in float16 so suppression survives `mixed_float16`.
The threshold is hard (`score > 0`) even when `soft_no_obj_ptr` is set, because
only the pointer blend may be soft. One consequence a loss must handle:
`ops.where` passes no gradient through the unselected branch, so a suppressed
row is gradient-free on the mask path and the score head needs its own explicit
loss.

The class has two entry points that differ in kind, not just in arguments.
`call` is the image path: encoder, prompt encoder, decoder, nothing else. It
never touches the memory bank or memory attention, and it is traceable, which
is what `fit()` needs. `stream_step` is the video path -- plain Python that
mutates the bank, branches on whether the bank is empty, and reads Python
integers out of its selection policy. It is deliberately never traced and never
routes through `self(...)`. Because `call` is memory-free, an image-only
inference is exactly SAM 1's shape at SAM 2's weights; `SAM2TrainingModel`
supplies the traceable multi-frame path by unrolling a static frame loop over
the submodules with a fresh, local memory bank.

Two defaults are the shipped *configuration* rather than the reference class
signature, because this port has no YAML layer to carry a config on top of a
constructor: `fixed_no_obj_ptr` defaults to `True` and `soft_no_obj_ptr` to
`False`, matching `sam2.1_hiera_l.yaml`. Taking the reference class defaults
would ship a model no released checkpoint was ever trained as. For the same
reason `obj_ptr_proj`, `obj_ptr_tpos_proj` and `no_obj_embed_spatial` have no
enabling flags at all -- every one of them is silent when absent. Relatedly,
`MODEL_VARIANTS` here stores only the numbers that live nowhere else; trunk and
neck geometry is read from `Hiera.MODEL_VARIANTS` and
`SAM2ImageEncoder.MODEL_VARIANTS`, and `from_variant` refuses an `image_size`
override for that reason. Only `tiny` and `hiera_l` are shipped: the other
published SAM 2 sizes' numbers were never read by this implementation and are
not invented.

No pretrained weights ship, and that is the deliberate position of the whole
`SAM/` package rather than an omission. SAM 2's released code is under the SAM
License, which is incompatible with this repository's GPL-3.0, so this is a
reimplementation from the paper and published configuration numbers -- no
upstream file copied, no upstream checkpoint loaded, and no accuracy or
tracking-quality claim made anywhere.

References:
    - Ravi et al., 2024. SAM 2: Segment Anything in Images and Videos.
      (https://arxiv.org/abs/2408.00714)
    - Kirillov et al., 2023. Segment Anything.
      (https://arxiv.org/abs/2304.02643)
    - Ryali et al., 2023. Hiera: A Hierarchical Vision Transformer without the
      Bells-and-Whistles. (https://arxiv.org/abs/2306.00989)
    - Lin et al., 2017. Feature Pyramid Networks for Object Detection.
      (https://arxiv.org/abs/1612.03144)
    - Su et al., 2021. RoFormer: Enhanced Transformer with Rotary Position
      Embedding. (https://arxiv.org/abs/2104.09864)
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import keras
from keras import ops

from dl_techniques.models.SAM.SAM1.mask_decoder import _build_mlp_head
from dl_techniques.models.SAM.SAM1.prompt_encoder import PromptEncoder
from dl_techniques.models.SAM.SAM1.transformer import TwoWayTransformer
from dl_techniques.models.SAM.SAM2.mask_decoder import SAM2MaskDecoder
from dl_techniques.models.SAM.SAM2.memory_attention import (
    DEFAULT_DROPOUT_RATE,
    SAM2MemoryAttention,
)
from dl_techniques.models.SAM.SAM2.memory_bank import SAM2MemoryBank
from dl_techniques.models.SAM.SAM2.memory_encoder import SAM2MemoryEncoder
from dl_techniques.models.SAM.SAM2.neck import SAM2ImageEncoder
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

#: Total stride from the input image to the memory grid. The memory encoder's
#: mask downsampler and the retained coarsest FPN level must agree on it, or the
#: encoded memory cannot be added to the pixel features.
MEMORY_STRIDE = 16

#: Depth of ``obj_ptr_proj``. Fixed by ``use_mlp_for_obj_ptr_proj: true``, which
#: selects ``MLP(hidden_dim, hidden_dim, hidden_dim, 3)`` over a single Linear.
_OBJ_PTR_PROJ_DEPTH = 3

#: Temperature of the object-pointer TEMPORAL sine encoding.
_OBJ_PTR_TPOS_TEMPERATURE = 10000.0

#: Placeholder mask logit for a frame the object-score head says holds no
#: object. Transcribed from the reference (``sam2_base.py:19``,
#: ``NO_OBJ_SCORE = -1024.0``) rather than chosen: it is not merely "a large
#: negative number", it is the specific value every downstream consumer of a
#: suppressed mask was trained against. In particular the memory encoder's
#: ``sigmoid(x) * 20 - 10`` saturates it to exactly ``-10`` (its lower limit),
#: and ``-1024`` is representable in float16, so the suppression survives
#: ``mixed_float16`` unchanged.
NO_OBJ_SCORE = -1024.0


def _sine_positional_encoding_1d(
        positions: Any,
        dim: int,
        temperature: float = _OBJ_PTR_TPOS_TEMPERATURE,
) -> Any:
    """Encode scalar positions as a 1D sine/cosine embedding.

    Interface contract (the object-pointer temporal encoding is its only
    caller today, but the shape rule is general):

    :param positions: ``(N,)`` scalar positions, already normalized by the
        caller. Anything ``ops.convert_to_tensor`` accepts.
    :type positions: Any
    :param dim: Output width. Must be positive and EVEN -- the two halves are
        the sines and the cosines of the same ``dim // 2`` frequencies.
    :type dim: int
    :param temperature: Geometric base of the frequency ladder.
    :type temperature: float
    :return: ``(N, dim)`` float32 encoding.
    :rtype: Any
    :raises ValueError: If ``dim`` is not a positive even number.

    The layout is ``concat([sin(a), cos(a)])`` -- the two halves are
    CONTIGUOUS, not interleaved. Interleaving is the more common convention and
    produces an encoding with the same shape, the same norm and the same
    pairwise distances under a fixed permutation, so it is invisible to every
    structural assertion and only shows up against a projection trained on the
    other layout.
    """
    if dim <= 0 or dim % 2 != 0:
        raise ValueError(
            f"dim must be a positive even number so the sine and cosine "
            f"halves are equal-width, got {dim}"
        )
    half = dim // 2
    frequency_index = ops.arange(half, dtype="float32")
    divisor = ops.power(
        ops.cast(temperature, "float32"),
        2.0 * ops.floor(frequency_index / 2.0) / float(half),
    )
    angles = ops.expand_dims(
        ops.cast(ops.convert_to_tensor(positions), "float32"), axis=-1
    ) / divisor
    return ops.concatenate([ops.sin(angles), ops.cos(angles)], axis=-1)


def _select_best_by_iou(tensor: Any, iou_predictions: Any) -> Any:
    """Gather each batch row's highest-IoU entry along the mask axis.

    Interface contract (both the pointer selection in :meth:`SAM2._decode` and
    the memory selection in :meth:`SAM2._store_memory` call it, so the two
    cannot drift apart):

    :param tensor: ``(B, M, ...)`` -- axis 1 is the mask/token axis.
    :type tensor: Any
    :param iou_predictions: ``(B, M)`` predicted IoU per entry.
    :type iou_predictions: Any
    :return: ``(B, 1, ...)`` -- axis 1 retained with length 1, so the result
        substitutes directly for a ``[:, 0:1]`` slice.
    :rtype: Any

    At ``M == 1`` this is exactly ``tensor[:, 0:1]``, so the single-mask path
    is unchanged; the two differ only under ``multimask_output=True``, where
    index 0 is multimask token 1 and is chosen by POSITION rather than by the
    model's own IoU estimate.

    INFERENCE-ONLY. Do NOT reuse this to pick which multimask slice a training
    loss supervises: upstream selects that slice by ``argmin(20 * loss_mask +
    loss_dice)`` against the ground truth (``training/loss_fns.py:219-297``),
    not by the predicted IoU, which is itself one of the things being trained.
    See ``progress.md`` "Iteration 2 -- carried constraints".
    """
    index = ops.argmax(iou_predictions, axis=-1)
    index = ops.cast(ops.reshape(index, (-1,) + (1,) * (len(tensor.shape) - 1)),
                     "int32")
    return ops.take_along_axis(tensor, index, axis=1)


@keras.saving.register_keras_serializable()
class SAM2(keras.Model):
    """Segment Anything 2 — promptable image and video segmentation.

    :param image_encoder: Hiera trunk + FPN neck + ``scalp`` drop.
    :type image_encoder: Union[SAM2ImageEncoder, Dict[str, Any]]
    :param prompt_encoder: SAM 1's prompt encoder, constructed as a SECOND
        instance (never shared with a SAM 1 model).
    :type prompt_encoder: Union[PromptEncoder, Dict[str, Any]]
    :param mask_decoder: The SAM 2 mask decoder.
    :type mask_decoder: Union[SAM2MaskDecoder, Dict[str, Any]]
    :param memory_attention: The memory-conditioning transformer stack.
    :type memory_attention: Union[SAM2MemoryAttention, Dict[str, Any]]
    :param memory_encoder: The mask-plus-features memory encoder.
    :type memory_encoder: Union[SAM2MemoryEncoder, Dict[str, Any]]
    :param num_maskmem: Spatial memory slots, including the conditioning
        bucket. Shipped value ``7``.
    :type num_maskmem: int
    :param image_size: Input resolution. ``None`` defers to the trunk's own
        ``image_size`` (S-3).
    :type image_size: Optional[int]
    :param multimask_output: Default mask-selection mode for :meth:`call` and
        :meth:`stream_step`. Either may override it per invocation.
    :type multimask_output: bool
    :param directly_add_no_mem_embed: On a frame with no memory, add
        ``no_mem_embed`` to the pixel features and SKIP memory attention
        entirely. Shipped value ``True``.
    :type directly_add_no_mem_embed: bool
    :param memory_temporal_stride_for_eval: Temporal subsampling stride of the
        non-conditioning selection. ``1`` during training.
    :type memory_temporal_stride_for_eval: int
    :param max_obj_ptrs_in_encoder: Cap on object pointers fed to memory
        attention.
    :type max_obj_ptrs_in_encoder: int
    :param soft_no_obj_ptr: Blend ``no_obj_ptr`` with the SIGMOID of the object
        score rather than its hard threshold. Defaults to ``False``, which is
        both the reference default and what the shipped config leaves unset.
    :type soft_no_obj_ptr: bool
    :param fixed_no_obj_ptr: Scale the predicted pointer by the
        object-appearing factor BEFORE adding the no-object term. Defaults to
        ``True``, which is what the shipped config sets; it is the reference
        CLASS default that is ``False``, and taking the class default here
        would ship an un-configured model.
    :type fixed_no_obj_ptr: bool
    :param kwargs: Additional keyword arguments for the ``Model`` base class.

    :raises ValueError: If the five components disagree on any shared width or
        grid — every such disagreement is silent at some call boundary, so all
        of them are refused at construction.

    Example:

    .. code-block:: python

        model = SAM2.from_variant("tiny")
        outputs = model({"image": images})          # image path, traceable

        model.stream_reset()
        for t, frame in enumerate(frames):          # video path, never traced
            out = model.stream_step(frame, frame_idx=t, is_conditioning=(t == 0))
    """

    #: Everything in this table is a number that lives NOWHERE else. The trunk
    #: geometry is read from ``Hiera.MODEL_VARIANTS`` and the neck/scalp
    #: geometry from ``SAM2ImageEncoder.MODEL_VARIANTS`` — see
    #: :meth:`from_variant`. A geometry restated in two homes is a latent
    #: defect, so this table deliberately does NOT repeat ``embed_dim``,
    #: ``stages``, ``window_spec``, ``image_size``, ``d_model`` or ``scalp``.
    #:
    #: Only ``tiny`` and ``hiera_l`` exist. The other published SAM 2 sizes'
    #: numbers were never read by this work; inventing them would be
    #: fabrication.
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "tiny": {
            "mem_dim": 8,
            "num_maskmem": 7,
            "memory_attention_layers": 2,
            "memory_attention_dim_feedforward": 64,
            "memory_attention_num_heads": 1,
            "memory_attention_downsample_rate": 1,
            "decoder_depth": 2,
            "decoder_num_heads": 2,
            "decoder_mlp_dim": 64,
            "prompt_mask_in_chans": 16,
            "dropout_rate": DEFAULT_DROPOUT_RATE,
        },
        "hiera_l": {
            "mem_dim": 64,
            "num_maskmem": 7,
            "memory_attention_layers": 4,
            "memory_attention_dim_feedforward": 2048,
            "memory_attention_num_heads": 1,
            "memory_attention_downsample_rate": 1,
            "decoder_depth": 2,
            "decoder_num_heads": 8,
            "decoder_mlp_dim": 2048,
            "prompt_mask_in_chans": 16,
            "dropout_rate": DEFAULT_DROPOUT_RATE,
        },
    }

    def __init__(
            self,
            image_encoder: Union[SAM2ImageEncoder, Dict[str, Any]],
            prompt_encoder: Union[PromptEncoder, Dict[str, Any]],
            mask_decoder: Union[SAM2MaskDecoder, Dict[str, Any]],
            memory_attention: Union[SAM2MemoryAttention, Dict[str, Any]],
            memory_encoder: Union[SAM2MemoryEncoder, Dict[str, Any]],
            num_maskmem: int = 7,
            image_size: Optional[int] = None,
            multimask_output: bool = False,
            directly_add_no_mem_embed: bool = True,
            memory_temporal_stride_for_eval: int = 1,
            max_obj_ptrs_in_encoder: int = 16,
            # DECISION plan-2026-08-04T044628-4c240b4c/D-037
            # These two defaults are the SHIPPED CONFIGURATION, not the
            # reference class signature: `sam2.1_hiera_l.yaml` sets
            # `fixed_no_obj_ptr: true` and leaves `soft_no_obj_ptr` unset (its
            # reference default is False). This port has no YAML layer, so the
            # constructor defaults ARE the shipped config and taking the class
            # signature's values would ship a model no released checkpoint was
            # trained as. Do NOT "restore" them to True/False. See D-037.
            soft_no_obj_ptr: bool = False,
            fixed_no_obj_ptr: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # Sub-layers -- created unconditionally, built explicitly in build().
        self.image_encoder = _as_layer(image_encoder)
        self.prompt_encoder = _as_layer(prompt_encoder)
        self.mask_decoder = _as_layer(mask_decoder)
        self.memory_attention = _as_layer(memory_attention)
        self.memory_encoder = _as_layer(memory_encoder)

        if num_maskmem < 1:
            raise ValueError(f"num_maskmem must be >= 1, got {num_maskmem}")

        self.num_maskmem = int(num_maskmem)
        # S-3: `None` defers to the trunk table; an explicit value always wins.
        self.image_size = int(
            self.image_encoder.trunk.image_size if image_size is None
            else image_size
        )
        self.multimask_output = bool(multimask_output)
        self.directly_add_no_mem_embed = bool(directly_add_no_mem_embed)
        self.memory_temporal_stride_for_eval = int(
            memory_temporal_stride_for_eval)
        self.max_obj_ptrs_in_encoder = int(max_obj_ptrs_in_encoder)
        self.soft_no_obj_ptr = bool(soft_no_obj_ptr)
        self.fixed_no_obj_ptr = bool(fixed_no_obj_ptr)

        # Derived, non-config: read from the components rather than restated.
        self.hidden_dim = int(self.image_encoder.neck.d_model)
        self.mem_dim = int(self.memory_encoder.out_dim)
        self._validate_component_agreement()

        # DECISION plan-2026-08-04T044628-4c240b4c/D-036
        # These two projections and `no_obj_embed_spatial` are NOT optional and
        # deliberately have no config flags. The shipped `sam2.1_hiera_l.yaml`
        # sets `use_mlp_for_obj_ptr_proj`, `proj_tpos_enc_in_obj_ptrs`,
        # `add_tpos_enc_to_obj_ptrs` and `no_obj_embed_spatial` all TRUE, and
        # this port has no YAML layer to carry them. Do NOT "restore
        # configurability" by adding flags defaulting to the reference class
        # signature (which turns all four off): every one of them is silent
        # when absent -- the model builds, trains and serializes without them,
        # and the only symptom is that every object pointer becomes temporally
        # indistinguishable. That is exactly how they went missing the first
        # time. See decisions.md D-036.
        self.obj_ptr_proj = _build_mlp_head(
            num_layers=_OBJ_PTR_PROJ_DEPTH,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            activation="relu",
            dense_name_template="obj_ptr_proj_dense{n}",
            name="obj_ptr_proj",
        )
        # Projects the pointer TEMPORAL encoding from `hidden_dim` (the width
        # the sine encoding is generated at, because `proj_tpos_enc_in_obj_ptrs`
        # is on) down to the memory width.
        self.obj_ptr_tpos_proj = keras.layers.Dense(
            self.mem_dim, name="obj_ptr_tpos_proj")

        # Plain-Python streaming state. Never a weight, never serialized: a
        # memory bank is a per-VIDEO object, not part of the architecture.
        self.memory_bank = SAM2MemoryBank(
            num_maskmem=self.num_maskmem,
            mem_dim=self.mem_dim,
            hidden_dim=self.hidden_dim,
            memory_temporal_stride_for_eval=self.memory_temporal_stride_for_eval,
            max_obj_ptrs_in_encoder=self.max_obj_ptrs_in_encoder,
        )
        self._stream_frame_counter = 0

        self.maskmem_tpos_enc = None
        self.no_mem_embed = None
        self.no_mem_pos_enc = None
        self.no_obj_ptr = None
        self.no_obj_embed_spatial = None

    # -----------------------------------------------------------------
    # construction-time agreement
    # -----------------------------------------------------------------

    @property
    def feature_grid(self) -> int:
        """Edge length of the memory / decoder feature grid.

        :return: ``image_size // MEMORY_STRIDE``.
        :rtype: int
        """
        return self.image_size // MEMORY_STRIDE

    # DECISION plan-2026-08-22T035419-a11304c8/D-090
    # DERIVED, deliberately: `dropout_rate` is NOT an `__init__` parameter and
    # NOT a `get_config()` key. Every live `Dropout` in a SAM 2 belongs to
    # `memory_attention`, which already stores and serializes the rate in its
    # own config; a second copy on the outer model would be a number with two
    # homes -- and one that can silently DISAGREE, because a caller may pass an
    # already-constructed `memory_attention` whose rate differs from whatever
    # the outer `__init__` was told. This property can never disagree, and it
    # round-trips for free through the nested config, so no pre-existing
    # `.keras` file gains a required key. Do NOT "complete" this by adding a
    # stored `self.dropout_rate` + config key. See decisions.md D-090.
    @property
    def dropout_rate(self) -> float:
        """Dropout rate actually in force on the memory-attention stack.

        :return: The rate carried by :attr:`memory_attention`.
        :rtype: float
        """
        return float(self.memory_attention.dropout_rate)

    def _validate_component_agreement(self) -> None:
        """Refuse every component mismatch that is silent downstream.

        :raises ValueError: On any width or grid disagreement.
        """
        checks = [
            ("mask decoder transformer_dim", self.mask_decoder.transformer_dim,
             self.hidden_dim, "the neck's d_model"),
            ("prompt encoder embed_dim", self.prompt_encoder.embed_dim,
             self.hidden_dim, "the neck's d_model"),
            ("memory attention d_model", self.memory_attention.d_model,
             self.hidden_dim, "the neck's d_model"),
            ("memory encoder in_dim", self.memory_encoder.in_dim,
             self.hidden_dim, "the neck's d_model"),
            ("memory attention kv_in_dim", self.memory_attention.kv_in_dim,
             self.mem_dim, "the memory encoder's out_dim"),
        ]
        for name, found, expected, source in checks:
            if int(found) != int(expected):
                raise ValueError(
                    f"{name} is {found} but must equal {expected} ({source}); "
                    f"a width mismatch here is a silent misassembly at some "
                    f"call boundary, not a shape error at this one"
                )
        if not self.mask_decoder.use_high_res_features:
            raise ValueError(
                "SAM2 always feeds the two high-resolution FPN skips to the "
                "mask decoder, so it must be built with "
                "use_high_res_features=True; without them the decoder still "
                "emits correctly shaped masks from a coarser stream"
            )

        grid = self.feature_grid
        if tuple(self.prompt_encoder.image_embedding_size) != (grid, grid):
            raise ValueError(
                f"the prompt encoder's image_embedding_size "
                f"{tuple(self.prompt_encoder.image_embedding_size)} must equal "
                f"the retained stride-{MEMORY_STRIDE} feature grid "
                f"({grid}, {grid}) at image_size={self.image_size}"
            )
        if tuple(self.memory_attention.feat_sizes) != (grid, grid):
            raise ValueError(
                f"the memory attention's feat_sizes "
                f"{tuple(self.memory_attention.feat_sizes)} must equal the "
                f"stride-{MEMORY_STRIDE} feature grid ({grid}, {grid}); the "
                f"rotary tables are built for that grid and a mismatch "
                f"silently rotates by the wrong positions"
            )
        if self.memory_encoder.mask_total_stride != MEMORY_STRIDE:
            raise ValueError(
                f"the memory encoder's mask_total_stride "
                f"{self.memory_encoder.mask_total_stride} must be "
                f"{MEMORY_STRIDE} so an image-resolution mask lands on the "
                f"feature grid"
            )

    # -----------------------------------------------------------------
    # variants
    # -----------------------------------------------------------------

    @classmethod
    def from_variant(cls, variant: str, **kwargs: Any) -> "SAM2":
        """Construct every component from the composed variant tables.

        The trunk/neck geometry is READ from ``Hiera.MODEL_VARIANTS`` and
        ``SAM2ImageEncoder.MODEL_VARIANTS`` rather than restated here, so there
        is exactly one home per number.

        :param variant: ``'tiny'`` or ``'hiera_l'``.
        :type variant: str
        :param kwargs: Explicit overrides. Any value given here wins over the
            variant table (S-3); passing ``None`` explicitly is the same as
            omitting the argument. ``dropout_rate`` is one of these: it is a
            table key, so ``from_variant('tiny', dropout_rate=0.0)`` reaches
            every ``Dropout`` in the memory-attention stack.
        :type kwargs: Any
        :return: The configured model.
        :rtype: SAM2
        :raises ValueError: If ``variant`` is unknown, if ``dropout_rate`` is
            outside ``[0.0, 1.0)``, or if ``image_size`` is overridden here —
            it belongs to ``Hiera.MODEL_VARIANTS``, which is its single home.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown SAM2 variant '{variant}'. Available: "
                f"{sorted(cls.MODEL_VARIANTS)}. Only these two are shipped: "
                f"the other published SAM 2 sizes' numbers were never read by "
                f"this implementation and are not invented here."
            )
        if kwargs.get("image_size") is not None:
            raise ValueError(
                "image_size cannot be overridden through SAM2.from_variant: it "
                "is trunk geometry and its single home is "
                "Hiera.MODEL_VARIANTS. Overriding it here would leave the "
                "trunk built for one resolution and everything downstream "
                "configured for another. Construct SAM2(...) directly with a "
                "matching image encoder instead."
            )

        table = dict(cls.MODEL_VARIANTS[variant])
        # S-3: an explicit `None` means "defer to the table", exactly as an
        # omitted argument does.
        overrides = {k: v for k, v in kwargs.items() if v is not None}
        table.update({k: v for k, v in overrides.items() if k in table})
        model_kwargs = {k: v for k, v in overrides.items() if k not in table}

        dropout_rate = float(table["dropout_rate"])
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(
                f"dropout_rate must be in [0.0, 1.0), got {dropout_rate}"
            )

        image_encoder = SAM2ImageEncoder.from_variant(variant)
        hidden_dim = int(image_encoder.neck.d_model)
        image_size = int(image_encoder.trunk.image_size)
        grid = image_size // MEMORY_STRIDE

        prompt_encoder = PromptEncoder(
            embed_dim=hidden_dim,
            image_embedding_size=(grid, grid),
            input_image_size=(image_size, image_size),
            mask_in_chans=table["prompt_mask_in_chans"],
        )
        mask_decoder = SAM2MaskDecoder(
            transformer_dim=hidden_dim,
            transformer=TwoWayTransformer(
                depth=table["decoder_depth"],
                embedding_dim=hidden_dim,
                num_heads=table["decoder_num_heads"],
                mlp_dim=table["decoder_mlp_dim"],
            ),
            use_high_res_features=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True,
            dynamic_multimask_via_stability=True,
        )
        memory_attention = SAM2MemoryAttention(
            d_model=hidden_dim,
            num_layers=table["memory_attention_layers"],
            dim_feedforward=table["memory_attention_dim_feedforward"],
            num_heads=table["memory_attention_num_heads"],
            downsample_rate=table["memory_attention_downsample_rate"],
            feat_sizes=(grid, grid),
            kv_in_dim=table["mem_dim"],
            # DECISION plan-2026-08-22T035419-a11304c8/D-090
            # The ONLY path by which a caller-chosen dropout rate reaches the
            # 12 (tiny) / 24 (hiera_l) live `Dropout` layers. Deleting this
            # keyword does not fail any shape, any count or any round trip --
            # the layer default silently takes over and the knob goes dead
            # exactly as it was before this step. The layer parameter is now
            # spelled `dropout_rate=` too (D-130 renamed it), so the model-level
            # knob and the layer knob finally share one name. See decisions.md
            # D-090 and D-130.
            dropout_rate=dropout_rate,
        )
        memory_encoder = SAM2MemoryEncoder(
            in_dim=hidden_dim,
            out_dim=table["mem_dim"],
            mask_total_stride=MEMORY_STRIDE,
        )

        logger.info("Creating SAM2 variant '%s'", variant)
        return cls(
            image_encoder=image_encoder,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder,
            memory_attention=memory_attention,
            memory_encoder=memory_encoder,
            num_maskmem=table["num_maskmem"],
            image_size=image_size,
            **model_kwargs,
        )

    # -----------------------------------------------------------------
    # build
    # -----------------------------------------------------------------

    def build(self, input_shape: Optional[Any] = None) -> None:
        """Build every component and create the four owned weights.

        Every sub-component is built EXPLICITLY rather than lazily on first
        call, so a ``.keras`` round-trip restores a complete weight set. (The
        two-way transformer inside the mask decoder still builds lazily — that
        is SAM 1's contract, unchanged here.)

        :param input_shape: Unused; the real input shape is fixed by
            ``image_size``.
        :type input_shape: Optional[Any]
        """
        if self.built:
            return

        size = self.image_size
        grid = self.feature_grid
        # Each sub-build is guarded on `.built` because deserialization builds
        # some components before this method runs, and NEITHER
        # `PromptEncoder.build` NOR `SAM2MaskDecoder.build` is idempotent: a
        # second call re-enters `add_weight` on an already-built layer and
        # raises "You cannot add new elements of state ... already built".
        if not self.image_encoder.built:
            self.image_encoder.build((None, size, size, 3))
        if not self.prompt_encoder.built:
            self.prompt_encoder.build(None)
        if not self.mask_decoder.built:
            self.mask_decoder.build(None)
        if not self.memory_attention.built:
            self.memory_attention.build(
                (None, grid * grid, self.hidden_dim),
                (None, None, self.mem_dim),
            )
        if not self.memory_encoder.built:
            self.memory_encoder.build([
                (None, grid, grid, self.hidden_dim),
                (None, size, size, self.memory_encoder.mask_in_chans),
            ])

        # DECISION plan-2026-08-04T044628-4c240b4c/D-026
        # `maskmem_tpos_enc` lives HERE and not in the memory bank, and the bank
        # returns SLOT INDICES rather than vectors. Do NOT "simplify" by moving
        # the table into the bank or by folding the temporal signal into the
        # rotary embedding: memory attention's RoPE table is spatial-only and is
        # broadcast IDENTICALLY across every memory frame (`repeat_k`), so the
        # temporal ordering of the memory is carried by this additive table
        # alone. Merging them yields a model that trains and cannot tell frame
        # t-1 from frame t-6. See decisions.md D-026 and H-13.
        self.maskmem_tpos_enc = self.add_weight(
            name="maskmem_tpos_enc",
            shape=(self.num_maskmem, 1, 1, self.mem_dim),
            initializer="zeros",
            trainable=True,
        )
        self.no_mem_embed = self.add_weight(
            name="no_mem_embed",
            shape=(1, 1, self.hidden_dim),
            initializer="zeros",
            trainable=True,
        )
        self.no_mem_pos_enc = self.add_weight(
            name="no_mem_pos_enc",
            shape=(1, 1, self.mem_dim),
            initializer="zeros",
            trainable=True,
        )
        self.no_obj_ptr = self.add_weight(
            name="no_obj_ptr",
            shape=(1, self.hidden_dim),
            initializer="zeros",
            trainable=True,
        )
        # The SPATIAL no-object embedding -- the second, independent
        # no-object mechanism (D-036). `no_obj_ptr` marks the pointer stream;
        # this marks the encoded spatial memory of an occluded frame.
        self.no_obj_embed_spatial = self.add_weight(
            name="no_obj_embed_spatial",
            shape=(1, self.mem_dim),
            initializer="zeros",
            trainable=True,
        )

        self.obj_ptr_proj.build((None, self.hidden_dim))
        self.obj_ptr_tpos_proj.build((None, self.hidden_dim))

        logger.debug(
            "SAM2 built: image_size=%d grid=%d hidden_dim=%d mem_dim=%d "
            "num_maskmem=%d",
            size, grid, self.hidden_dim, self.mem_dim, self.num_maskmem,
        )
        super().build(input_shape)

    def get_build_config(self) -> Optional[Dict[str, Any]]:
        """Force Keras to call :meth:`build_from_config` on load.

        A model saved UNBUILT has no weights in the archive, so forcing a build
        at load would raise. Returning ``None`` there preserves the stock
        unbuilt-save / unbuilt-load behaviour.

        :return: ``{'image_size': int}`` when built, else ``None``.
        :rtype: Optional[Dict[str, Any]]
        """
        if not self.built:
            return None
        return {"image_size": int(self.image_size)}

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Create this model's own weights before Keras restores them.

        # DECISION plan-2026-08-04T044628-4c240b4c/D-028
        # There is deliberately NO dummy forward pass here, and adding one
        # "for safety" would be a real cost: at ``hiera_l`` it is a full
        # 1024x1024 forward through 221M parameters on EVERY ``load_model``.
        # SAM 1's own ``build_from_config`` runs one because it measured 138 of
        # 202 weights restored without it. That does NOT transfer: measured at
        # this HEAD on Keras 3.8, a ``tiny`` model saved after a forward pass
        # reloads with **336 of 336** variables present BEFORE the first call —
        # Keras records a per-layer build config for lazily-built sub-layers
        # (including SAM 1's ``TwoWayTransformer``, 82 variables) and rebuilds
        # them itself. Do NOT re-add the forward without first re-running
        # ``test_weight_count_is_sampled_before_the_first_forward_call``, which
        # is the instrument that measures whether it is needed. See
        # decisions.md D-028.

        :param config: The dict returned by :meth:`get_build_config`.
        :type config: Dict[str, Any]
        """
        del config
        if not self.built:
            self.build(None)

    # -----------------------------------------------------------------
    # the image path -- traceable
    # -----------------------------------------------------------------

    def call(
            self,
            inputs: Dict[str, Any],
            training: Optional[bool] = None,
            multimask_output: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Segment one batch of images from prompts. No memory is involved.

        :param inputs: ``{'image': (B, image_size, image_size, 3)}`` plus the
            optional prompt keys ``'points'`` (a ``(coords, labels)`` pair),
            ``'boxes'`` and ``'masks'``. ``'masks'`` is a low-resolution mask
            PROMPT; see the re-feed note below before passing a previous
            call's ``low_res_logits`` back in through it.
        :type inputs: Dict[str, Any]
        :param training: Keras training flag.
        :type training: Optional[bool]
        :param multimask_output: Per-call override; ``None`` defers to the
            configured default (S-3).
        :type multimask_output: Optional[bool]
        :return: ``{'low_res_logits', 'iou_predictions', 'object_score_logits',
            'object_pointer'}``.
        :rtype: Dict[str, Any]
        :raises ValueError: If ``'image'`` is absent.

        ``low_res_logits`` is the training target, WITH ONE CAVEAT that a loss
        must account for: on any batch row whose ``object_score_logits`` is
        ``<= 0`` it is not a prediction at all but the uniform constant
        :data:`NO_OBJ_SCORE` (``-1024``), written by
        :meth:`_suppress_absent_object` (D-043). ``ops.where`` passes no
        gradient through the unselected branch, so on such a row the mask
        path is gradient-free and the decoder learns nothing from a mask loss.
        The score head itself is likewise unreachable from a mask loss --
        every consumer of ``object_score_logits`` in this package thresholds
        it hard at ``> 0`` -- so it needs an explicit loss on that output key.
        Upstream's recipe is a plain BCE at weight 1
        (``training/loss_fns.py:219-297``,
        ``sam2.1_hiera_b+_MOSE_finetune.yaml:288-293``), with the mask/dice/IoU
        losses gated by GROUND-TRUTH presence rather than by this predicted
        score. See ``progress.md`` "Iteration 2 -- carried constraints".

        DIVERGENCE FROM UPSTREAM ON THE SINGLE-IMAGE PATH. The suppression is
        applied by this port's shared :meth:`_decode`, so it fires for
        :meth:`call` as well as for :meth:`stream_step`. Upstream applies it
        only inside ``_forward_sam_heads`` (``sam2_base.py:358-368``), which is
        what the VIDEO predictor uses; its IMAGE predictor bypasses that method
        entirely, calling ``self.model.sam_mask_decoder(...)`` directly
        (``sam2_image_predictor.py:420``) and returning the real mask. So on a
        false-negative occlusion call -- a trained checkpoint whose score head
        emits a negative logit for an object that IS present -- upstream's
        ``SAM2ImagePredictor.predict`` hands back the mask and this method
        hands back ``-1024``. The behaviour is deliberate: this package exists
        to serve the memory-conditioned video path, where a mask that
        contradicts its own occlusion flag is a state the reference never
        writes into the memory bank. A caller that wants the image predictor's
        semantics reads ``object_score_logits`` and re-runs
        :meth:`SAM2MaskDecoder.call` itself.

        THE UPSTREAM ``+-32`` CLAMP IS DELIBERATELY NOT APPLIED HERE. Both
        upstream sites (``sam2_image_predictor.py:434``,
        ``sam2_video_predictor.py:262``) clamp at a mask-PROMPT boundary, not
        on a model output: the video predictor clamps the previous frame's
        ``pred_masks`` immediately before re-feeding them, and the image
        predictor clamps only the copy it returns for use as the next call's
        ``mask_input`` -- the upsampled masks it actually reports are computed
        from the UNCLAMPED logits at ``:425-427``. Applying it in
        :meth:`_decode` would therefore change no segmentation while destroying
        the ``-1024`` sentinel that D-043 exists to write. This port has no
        internal re-feed loop, so there is no site that needs it; a caller who
        does re-feed ``low_res_logits`` through the ``'masks'`` prompt key must
        clamp to ``[-32, 32]`` first, exactly as both upstream predictors do.
        """
        if "image" not in inputs:
            raise ValueError("SAM2 inputs must contain an 'image' key")

        encoded = self.image_encoder(inputs["image"], training=training)
        return self._decode(
            encoded=encoded,
            features=encoded["vision_features"],
            inputs=inputs,
            multimask_output=multimask_output,
            training=training,
        )

    def _decode(
            self,
            encoded: Dict[str, Any],
            features: Any,
            inputs: Dict[str, Any],
            multimask_output: Optional[bool],
            training: Optional[bool],
    ) -> Dict[str, Any]:
        """Run the prompt encoder and mask decoder over given image features.

        Shared by :meth:`call` (raw features) and :meth:`stream_step`
        (memory-conditioned features) so the two paths cannot drift apart.

        :param encoded: The image encoder's output dict.
        :type encoded: Dict[str, Any]
        :param features: ``(B, grid, grid, hidden_dim)`` features to decode.
        :type features: Any
        :param inputs: The prompt keys.
        :type inputs: Dict[str, Any]
        :param multimask_output: Per-call override, or ``None``.
        :type multimask_output: Optional[bool]
        :param training: Keras training flag.
        :type training: Optional[bool]
        :return: The four output tensors, keyed.
        :rtype: Dict[str, Any]
        """
        multimask = (
            self.multimask_output if multimask_output is None
            else bool(multimask_output)
        )
        sparse, dense = self.prompt_encoder(
            points=inputs.get("points"),
            boxes=inputs.get("boxes"),
            masks=inputs.get("masks"),
            training=training,
        )
        fpn = encoded["backbone_fpn"]
        low_res_logits, iou, object_score_logits, pointer_tokens = \
            self.mask_decoder(
                image_embeddings=features,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=multimask,
                high_res_features=[fpn[0], fpn[1]],
                training=training,
            )
        low_res_logits = self._suppress_absent_object(
            low_res_logits, object_score_logits)
        # DECISION plan-2026-08-04T044628-4c240b4c/D-047
        # Do NOT add upstream's `torch.clamp(low_res_masks, -32.0, 32.0)` here
        # "for parity". Both upstream sites apply it at a mask-PROMPT boundary
        # -- `sam2_video_predictor.py:262` on the previous frame's `pred_masks`
        # just before re-feeding them, `sam2_image_predictor.py:434` only on the
        # copy returned for use as the next call's `mask_input` (the masks it
        # reports come from `postprocess_masks` at `:425-427`, computed from the
        # UNCLAMPED logits). Here it would change no segmentation while
        # overwriting the `-1024` sentinel the line above just wrote, turning
        # every D-043 guard into an assertion about `-32`. The clamp belongs at
        # the caller's re-feed, which is where `SAM2.call`'s docstring puts it.
        # See decisions.md D-047.
        #
        # DECISION plan-2026-08-04T044628-4c240b4c/D-038
        # The pointer token is gathered by the model's OWN best-IoU estimate,
        # never by position. Do NOT "simplify" this back to
        # `pointer_tokens[:, 0, :]`: under `multimask_output=True` the decoder
        # has already sliced away the single-mask token, so index 0 is
        # *multimask token 1* -- an arbitrary one of three. The memory and the
        # object pointer would then be built from a mask the model did not
        # judge best, with no shape error and no measurable symptom at batch 1
        # on the single-mask path (where M == 1 and the two agree exactly).
        # See decisions.md D-038.
        #
        # DECISION plan-2026-08-04T044628-4c240b4c/D-044
        # The gather is CONDITIONAL on the pointer axis being longer than 1.
        # Do NOT drop this branch and gather unconditionally: the pointer axis
        # and the IoU axis are NOT the same length in general. The decoder
        # emits one pointer token per mask token only when
        # `use_multimask_token_for_obj_ptr` is set (its own default is False,
        # D-023), so at `multimask_output=True` with that default the pointer
        # axis is 1 while `iou` is 3 and an unconditional gather raises
        # `InvalidArgumentError: indices[0,0] = 1 is not in [0, 1)`. The
        # reference guards it the same way (`sam2_base.py:387`,
        # `if sam_output_tokens.size(1) > 1`). See decisions.md D-044.
        if int(pointer_tokens.shape[1]) > 1:
            selected_token = ops.squeeze(
                _select_best_by_iou(pointer_tokens, iou), axis=1)
        else:
            selected_token = pointer_tokens[:, 0, :]
        pointer = self.obj_ptr_proj(selected_token, training=training)
        return {
            "low_res_logits": low_res_logits,
            "iou_predictions": iou,
            "object_score_logits": object_score_logits,
            "object_pointer": self._blend_object_pointer(
                pointer, object_score_logits),
        }

    def _suppress_absent_object(self, logits: Any, score: Any) -> Any:
        """Replace every mask logit with :data:`NO_OBJ_SCORE` where absent.

        # DECISION plan-2026-08-04T044628-4c240b4c/D-043
        # This runs BEFORE the best-IoU gather and therefore before the high-
        # resolution resize, before the memory encoder and before the value is
        # returned to the caller. Do NOT move it later "so the raw mask is
        # still available", and do NOT delete it because the occlusion is
        # already flagged two other ways (`no_obj_ptr` on the pointer stream,
        # `no_obj_embed_spatial` on the spatial stream). Those two MARK an
        # occluded frame; this one ERASES its mask. Without it the memory bank
        # stores the object's real, unsuppressed mask together with an
        # occlusion flag -- a contradictory state the reference never writes,
        # and one with no shape, dtype or finiteness symptom. The threshold is
        # HARD (`score > 0`) even when `soft_no_obj_ptr` is set: the reference
        # comments that the spatial mask is "always a *hard* choice", and only
        # the POINTER blend may be soft (`sam2_base.py:358-368`).
        # See decisions.md D-043.

        :param logits: ``(B, M, h, w)`` mask logits, straight from the decoder.
        :type logits: Any
        :param score: ``(B, 1)`` object-score logits.
        :type score: Any
        :return: ``(B, M, h, w)``; rows whose score is ``<= 0`` are uniformly
            :data:`NO_OBJ_SCORE`, the rest are untouched.
        :rtype: Any
        """
        appearing = ops.reshape(score, (-1, 1, 1, 1)) > 0
        return ops.where(
            appearing, logits, ops.cast(NO_OBJ_SCORE, logits.dtype))

    def _blend_object_pointer(self, pointer: Any, score: Any) -> Any:
        """Interpolate the predicted pointer towards the learned ``no_obj_ptr``.

        # DECISION plan-2026-08-04T044628-4c240b4c/D-039
        # The `lambda * pointer` multiply happens ONLY under
        # `fixed_no_obj_ptr`; the `(1 - lambda) * no_obj_ptr` term is added
        # UNCONDITIONALLY. Do NOT "tidy" this into the symmetric-looking
        # `lambda * pointer + (1 - lambda) * no_obj`: the two expressions
        # coincide at lambda in {0, 1} and NOWHERE else, so a guard sited at
        # saturated object scores (+-30 logits) cannot tell them apart while
        # every real input lies strictly between. At `score = 0` the reference
        # returns `ptr + 0.5 * no_obj` and the symmetric form returns
        # `0.5 * ptr + 0.5 * no_obj`. See decisions.md D-039.

        :param pointer: ``(B, hidden_dim)`` predicted pointer.
        :type pointer: Any
        :param score: ``(B, 1)`` object-score logits.
        :type score: Any
        :return: ``(B, hidden_dim)`` blended pointer.
        :rtype: Any
        """
        appearing = (
            ops.sigmoid(score) if self.soft_no_obj_ptr
            else ops.cast(score > 0, pointer.dtype)
        )
        appearing = ops.cast(appearing, pointer.dtype)
        if self.fixed_no_obj_ptr:
            pointer = appearing * pointer
        no_obj = ops.cast(self.no_obj_ptr, pointer.dtype)
        return pointer + (1.0 - appearing) * no_obj

    # -----------------------------------------------------------------
    # the video path -- plain Python, NEVER traced
    # -----------------------------------------------------------------

    def stream_reset(self) -> None:
        """Clear the memory bank. Call once per video, before frame 0."""
        self.memory_bank.reset()
        self._stream_frame_counter = 0

    def stream_step(
            self,
            image: Any,
            frame_idx: Optional[int] = None,
            points: Optional[Tuple[Any, Any]] = None,
            boxes: Optional[Any] = None,
            masks: Optional[Any] = None,
            is_conditioning: bool = False,
            multimask_output: Optional[bool] = None,
            track_in_reverse: bool = False,
            run_memory_encoder: bool = True,
    ) -> Dict[str, Any]:
        """Track one frame, conditioned on the memory of the previous ones.

        This method is deliberately NOT traceable and deliberately does not go
        through ``self(...)``: it mutates Python state (the memory bank), takes
        a Python branch on whether the bank is empty, and reads Python integers
        out of the bank's selection policy.

        :param image: ``(B, image_size, image_size, 3)`` single frame.
        :type image: Any
        :param frame_idx: Index within the video; ``None`` continues the
            internal counter.
        :type frame_idx: Optional[int]
        :param points: Optional ``(coords, labels)`` prompt.
        :type points: Optional[Tuple[Any, Any]]
        :param boxes: Optional box prompt.
        :type boxes: Optional[Any]
        :param masks: Optional mask prompt.
        :type masks: Optional[Any]
        :param is_conditioning: Store this frame in the conditioning bucket
            (``t_pos = 0`` forever).
        :type is_conditioning: bool
        :param multimask_output: Per-call override; ``None`` defers to the
            configured default.
        :type multimask_output: Optional[bool]
        :param track_in_reverse: Track backwards in time.
        :type track_in_reverse: bool
        :param run_memory_encoder: Encode and store this frame's memory.
        :type run_memory_encoder: bool
        :return: :meth:`call`'s four keys plus ``'frame_idx'``,
            ``'num_memory_tokens'`` and ``'num_obj_ptr_tokens'``.
        :rtype: Dict[str, Any]
        """
        index = self._stream_frame_counter if frame_idx is None else int(frame_idx)

        encoded = self.image_encoder(image, training=False)
        features = encoded["vision_features"]
        conditioned, num_memory_tokens, num_ptr_tokens = self._condition_on_memory(
            features, encoded["vision_pos_enc"][-1], index, track_in_reverse)

        outputs = self._decode(
            encoded=encoded,
            features=conditioned,
            inputs={"points": points, "boxes": boxes, "masks": masks},
            multimask_output=multimask_output,
            training=False,
        )

        if run_memory_encoder:
            self._store_memory(
                index, features, outputs, is_conditioning=is_conditioning)

        self._stream_frame_counter = index + 1
        outputs = dict(outputs)
        outputs.update({
            "frame_idx": index,
            "num_memory_tokens": num_memory_tokens,
            "num_obj_ptr_tokens": num_ptr_tokens,
        })
        return outputs

    def _condition_on_memory(
            self,
            features: Any,
            features_pos: Any,
            frame_idx: int,
            track_in_reverse: bool,
    ) -> Tuple[Any, int, int]:
        """Run memory attention over the bank's readout for one frame.

        :param features: ``(B, grid, grid, hidden_dim)`` raw pixel features.
        :type features: Any
        :param features_pos: Positional encoding of ``features``, same shape.
        :type features_pos: Any
        :param frame_idx: The frame being tracked.
        :type frame_idx: int
        :param track_in_reverse: Track backwards in time.
        :type track_in_reverse: bool
        :return: ``(conditioned_features, num_memory_tokens,
            num_obj_ptr_tokens)``.
        :rtype: Tuple[Any, int, int]
        """
        shape = ops.shape(features)
        batch, grid_h, grid_w = shape[0], features.shape[1], features.shape[2]
        tokens = ops.reshape(features, (batch, grid_h * grid_w, self.hidden_dim))
        tokens_pos = ops.reshape(
            features_pos, (batch, grid_h * grid_w, self.hidden_dim))

        readout = self.memory_bank.read(
            frame_idx, track_in_reverse=track_in_reverse)

        if readout.memory is None:
            if self.directly_add_no_mem_embed:
                conditioned = tokens + ops.cast(self.no_mem_embed, tokens.dtype)
                return (
                    ops.reshape(
                        conditioned, (batch, grid_h, grid_w, self.hidden_dim)),
                    0, 0,
                )
            # DECISION plan-2026-08-04T044628-4c240b4c/D-027
            # A ONE-token dummy memory whose
            # content is zeros and whose position is the learned
            # `no_mem_pos_enc`. The reference expression
            # `no_mem_embed.expand(1, B, mem_dim)` cannot be transcribed: that
            # parameter is `hidden_dim`-wide and an expand cannot NARROW it to
            # the `mem_dim`-wide memory stream, so upstream's own fallback is
            # shape-impossible at the shipped widths (it is unreachable there
            # because `directly_add_no_mem_embed` is True). Do NOT "restore"
            # the reference expression. See decisions.md D-027.
            memory = ops.zeros((batch, 1, self.mem_dim), dtype=tokens.dtype)
            memory_pos = ops.broadcast_to(
                ops.cast(self.no_mem_pos_enc, tokens.dtype),
                (batch, 1, self.mem_dim),
            )
            num_ptr_tokens = 0
        else:
            # H-4: every fed-back boundary is detached. Without this, tracking N
            # frames builds one N-deep recurrent graph instead of N decodes.
            memory = ops.stop_gradient(readout.memory)
            memory_pos = ops.stop_gradient(readout.memory_pos)
            memory_pos = memory_pos + self._temporal_embedding(readout, memory)
            num_ptr_tokens = readout.num_obj_ptr_tokens

        conditioned = self.memory_attention(
            tokens,
            memory,
            features_pos=tokens_pos,
            memory_pos=memory_pos,
            num_obj_ptr_tokens=num_ptr_tokens,
            training=False,
        )
        return (
            ops.reshape(conditioned, (batch, grid_h, grid_w, self.hidden_dim)),
            int(memory.shape[1]),
            num_ptr_tokens,
        )

    def _temporal_embedding(self, readout: Any, memory: Any) -> Any:
        """Expand the bank's slot indices into an additive temporal encoding.

        The bank returns one slot index per selected FRAME plus that frame's
        token count; this expands them per TOKEN and gathers the learned rows.

        The two halves of the memory sequence carry temporal signal by
        DIFFERENT means, and neither is optional:

        * spatial frame tokens get a learned row of ``maskmem_tpos_enc``,
          selected by the bank's slot index;
        * object-pointer tokens get a FIXED sine encoding of the bank's
          ``obj_ptr_tpos`` (how many frames away that pointer is), projected to
          ``mem_dim`` by :attr:`obj_ptr_tpos_proj`.

        # DECISION plan-2026-08-04T044628-4c240b4c/D-040
        # The pointer tail is NOT zeros. Do NOT "simplify" it back to
        # `ops.zeros(...)` on the theory that "the temporal signal rides on the
        # pointer values themselves" -- it does not. The rotary embedding in
        # memory attention is spatial-only and is broadcast identically across
        # every memory token, and `maskmem_tpos_enc` is indexed by SPATIAL slot
        # and never reaches the pointer tail. With zeros here, a pointer from
        # frame t-1 and a pointer from frame t-15 are numerically
        # indistinguishable to memory attention, which is the entire mechanism
        # `add_tpos_enc_to_obj_ptrs: true` exists to provide. The bank was
        # already computing and returning `obj_ptr_tpos` and NOTHING consumed
        # it. See decisions.md D-040.

        :param readout: The bank's ``_MemoryReadout``.
        :type readout: Any
        :param memory: The assembled memory tensor, for dtype only.
        :type memory: Any
        :return: ``(1, num_memory_tokens, mem_dim)``.
        :rtype: Any
        """
        per_token: List[int] = []
        for slot, count in zip(readout.tpos_slots, readout.frame_token_counts):
            per_token.extend([int(slot)] * int(count))

        table = ops.reshape(
            self.maskmem_tpos_enc, (self.num_maskmem, self.mem_dim))
        encoding = ops.take(
            table, ops.convert_to_tensor(per_token, dtype="int32"), axis=0)
        encoding = ops.expand_dims(encoding, axis=0)
        if readout.num_obj_ptr_tokens:
            encoding = ops.concatenate(
                [encoding, self._object_pointer_temporal_encoding(readout)],
                axis=1)
        return ops.cast(encoding, memory.dtype)

    def _object_pointer_temporal_encoding(self, readout: Any) -> Any:
        """Sine-encode and project the pointer tail's temporal differences.

        The bank hands back ``obj_ptr_tpos`` already expanded PER TOKEN (each
        ``hidden_dim``-wide pointer splits into ``hidden_dim // mem_dim``
        tokens that share their pointer's temporal difference), so the sine
        encoding is taken per token here. That is value-identical to encoding
        per pointer and repeating afterwards -- the same function applied to
        already-duplicated inputs -- and avoids a second expansion rule that
        could drift from the bank's.

        :param readout: The bank's ``_MemoryReadout``.
        :type readout: Any
        :return: ``(1, num_obj_ptr_tokens, mem_dim)``.
        :rtype: Any
        """
        # Normalized by the largest temporal difference the pointer cap can
        # produce, so the encoding's frequency content does not depend on how
        # far into the video the tracker is. `max(..., 1)` only guards the
        # degenerate cap of 1.
        #
        # This is NOT "exactly as the reference does", and an earlier version
        # of this comment said it was. Two known divergences, both recorded in
        # progress.md under Deferred:
        #   (a) the reference clamps the cap to the video length first
        #       (`min(num_frames, max_obj_ptrs_in_encoder)`, `sam2_base.py:588`
        #       then `:628`), so on any video shorter than the cap its span is
        #       SMALLER than this constant 15;
        #   (b) this port's pointer loop is `range(1, max_obj_ptrs + 1)` where
        #       the reference's is `range(1, max_obj_ptrs)` (memory_bank.py,
        #       Warning 5(c)), so a saturated bank can reach `t_diff = 16` here
        #       and produce a normalized position of 16/15 = 1.067, i.e.
        #       OUTSIDE the [0, 1] range this span is defined over.
        # Both are latent at every configuration this suite exercises; neither
        # is invisible any more.
        span = float(max(self.max_obj_ptrs_in_encoder - 1, 1))
        positions = [float(diff) / span for diff in readout.obj_ptr_tpos]
        encoding = _sine_positional_encoding_1d(positions, self.hidden_dim)
        # `training=False` is hardcoded because this method is reachable ONLY
        # from `stream_step`, which is inference-only by construction. Inert
        # for a plain Dense; it stops being inert if this projection ever gains
        # dropout or normalization.
        encoding = self.obj_ptr_tpos_proj(encoding, training=False)
        return ops.expand_dims(encoding, axis=0)

    def _store_memory(
            self,
            frame_idx: int,
            features: Any,
            outputs: Dict[str, Any],
            is_conditioning: bool,
    ) -> None:
        """Encode this frame's prediction and push it into the bank.

        :param frame_idx: Index within the video.
        :type frame_idx: int
        :param features: ``(B, grid, grid, hidden_dim)`` RAW pixel features —
            not the memory-conditioned ones.
        :type features: Any
        :param outputs: The decoder outputs for this frame.
        :type outputs: Dict[str, Any]
        :param is_conditioning: Store in the conditioning bucket.
        :type is_conditioning: bool
        """
        # The SAME best-IoU selection the object pointer uses (D-038), through
        # the shared helper so the frame's memory and its pointer can never be
        # built from different masks.
        logits = _select_best_by_iou(
            outputs["low_res_logits"], outputs["iou_predictions"])
        logits = ops.transpose(logits, (0, 2, 3, 1))
        # A Python tuple of ints from config: `len()`-able, so this resize is
        # graph-legal as well as eager-legal.
        high_res = ops.image.resize(
            logits, (self.image_size, self.image_size),
            interpolation="bilinear")

        memory, memory_pos = self.memory_encoder([
            ops.stop_gradient(features), ops.stop_gradient(high_res)])
        memory = self._mark_occlusion(memory, outputs["object_score_logits"])
        self.memory_bank.add_frame(
            frame_idx,
            maskmem_features=memory,
            maskmem_pos_enc=memory_pos,
            obj_ptr=ops.stop_gradient(outputs["object_pointer"]),
            is_conditioning=is_conditioning,
        )

    def _mark_occlusion(self, memory: Any, score: Any) -> Any:
        """Add ``no_obj_embed_spatial`` in proportion to ``1 - is_appearing``.

        The threshold is HARD (``score > 0``) here, unlike the object-pointer
        blend, which follows :attr:`soft_no_obj_ptr`. That asymmetry is the
        reference's: the spatial mark is a binary "this frame is occluded" flag
        on the memory, while the pointer blend may be soft.

        :param memory: ``(B, h, w, mem_dim)`` encoded memory.
        :type memory: Any
        :param score: ``(B, 1)`` object-score logits.
        :type score: Any
        :return: ``(B, h, w, mem_dim)`` memory with the occlusion mark added.
        :rtype: Any
        """
        appearing = ops.cast(
            ops.reshape(score, (-1, 1, 1, 1)) > 0, memory.dtype)
        embedding = ops.cast(
            ops.reshape(self.no_obj_embed_spatial, (1, 1, 1, self.mem_dim)),
            memory.dtype)
        return memory + (1.0 - appearing) * embedding

    # -----------------------------------------------------------------
    # shapes / config
    # -----------------------------------------------------------------

    def compute_output_shape(
            self, input_shape: Dict[str, Tuple[Optional[int], ...]]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """Return the output shapes, derived from stored config.

        :param input_shape: ``{'image': (batch, H, W, 3)}``.
        :type input_shape: Dict[str, Tuple[Optional[int], ...]]
        :return: One shape per :meth:`call` output key.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        batch = tuple(input_shape.get("image", (None,)))[0]
        num_masks = (
            self.mask_decoder.num_multimask_outputs if self.multimask_output
            else 1
        )
        mask_edge = self.feature_grid * 4
        return {
            "low_res_logits": (batch, num_masks, mask_edge, mask_edge),
            "iou_predictions": (batch, num_masks),
            "object_score_logits": (batch, 1),
            "object_pointer": (batch, self.hidden_dim),
        }

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all ``__init__`` parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "image_encoder": keras.saving.serialize_keras_object(
                self.image_encoder),
            "prompt_encoder": keras.saving.serialize_keras_object(
                self.prompt_encoder),
            "mask_decoder": keras.saving.serialize_keras_object(
                self.mask_decoder),
            "memory_attention": keras.saving.serialize_keras_object(
                self.memory_attention),
            "memory_encoder": keras.saving.serialize_keras_object(
                self.memory_encoder),
            "num_maskmem": self.num_maskmem,
            "image_size": self.image_size,
            "multimask_output": self.multimask_output,
            "directly_add_no_mem_embed": self.directly_add_no_mem_embed,
            "memory_temporal_stride_for_eval":
                self.memory_temporal_stride_for_eval,
            "max_obj_ptrs_in_encoder": self.max_obj_ptrs_in_encoder,
            "soft_no_obj_ptr": self.soft_no_obj_ptr,
            "fixed_no_obj_ptr": self.fixed_no_obj_ptr,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SAM2":
        """Reconstruct a model from :meth:`get_config`.

        :param config: The configuration dictionary.
        :type config: Dict[str, Any]
        :return: The reconstructed model.
        :rtype: SAM2
        """
        for key in (
                "image_encoder", "prompt_encoder", "mask_decoder",
                "memory_attention", "memory_encoder",
        ):
            config[key] = keras.saving.deserialize_keras_object(config[key])
        return cls(**config)


# ---------------------------------------------------------------------


def _as_layer(component: Union[keras.layers.Layer, Dict[str, Any]]) -> Any:
    """Return a component, deserializing it if it arrived as a config dict.

    :param component: A layer instance or its serialized configuration.
    :type component: Union[keras.layers.Layer, Dict[str, Any]]
    :return: The layer instance.
    :rtype: Any
    """
    if isinstance(component, keras.layers.Layer):
        return component
    return keras.saving.deserialize_keras_object(component)


def create_sam2(
        variant: str = "hiera_l",
        num_maskmem: Optional[int] = None,
        multimask_output: Optional[bool] = None,
        directly_add_no_mem_embed: Optional[bool] = None,
        memory_temporal_stride_for_eval: Optional[int] = None,
        max_obj_ptrs_in_encoder: Optional[int] = None,
        mem_dim: Optional[int] = None,
        dropout_rate: Optional[float] = None,
        **kwargs: Any,
) -> SAM2:
    """Build a SAM 2 model from a variant name with optional overrides.

    Every keyword defaults to ``None``, which means "defer to the variant
    table" (S-3). An explicit value ALWAYS wins, including a falsy one such as
    ``multimask_output=False`` — which is why the sentinel is ``None`` and not a
    concrete default.

    :param variant: ``'tiny'`` or ``'hiera_l'``.
    :type variant: str
    :param num_maskmem: Override the spatial memory slot count.
    :type num_maskmem: Optional[int]
    :param multimask_output: Override the default mask-selection mode.
    :type multimask_output: Optional[bool]
    :param directly_add_no_mem_embed: Override the empty-memory policy.
    :type directly_add_no_mem_embed: Optional[bool]
    :param memory_temporal_stride_for_eval: Override the selection stride.
    :type memory_temporal_stride_for_eval: Optional[int]
    :param max_obj_ptrs_in_encoder: Override the object-pointer cap.
    :type max_obj_ptrs_in_encoder: Optional[int]
    :param mem_dim: Override the memory token width.
    :type mem_dim: Optional[int]
    :param dropout_rate: Override the memory-attention dropout rate. ``None``
        defers to the variant table, whose shipped value is
        ``DEFAULT_DROPOUT_RATE`` (0.1) for both variants.
    :type dropout_rate: Optional[float]
    :param kwargs: Further overrides forwarded to :meth:`SAM2.from_variant`.
    :type kwargs: Any
    :return: The configured model.
    :rtype: SAM2
    """
    return SAM2.from_variant(
        variant,
        num_maskmem=num_maskmem,
        multimask_output=multimask_output,
        directly_add_no_mem_embed=directly_add_no_mem_embed,
        memory_temporal_stride_for_eval=memory_temporal_stride_for_eval,
        max_obj_ptrs_in_encoder=max_obj_ptrs_in_encoder,
        mem_dim=mem_dim,
        dropout_rate=dropout_rate,
        **kwargs,
    )


# ---------------------------------------------------------------------
