"""H-Net: the assembled byte-level causal language model.

Defines :class:`HNet`, which maps byte ids to next-token logits through an
embedding, the recursive :class:`HNetStage` backbone, and a head that is either
tied to the embedding table or its own Dense. Initialization is not one value:
the embedding starts at stddev 1.0, every Linear at ``initializer_range``, and
the projections that write into the residual stream are scaled per stage by
``1 / sqrt(n_k)``, the outside-in cumulative residual count. The chunking ratio
loss reaches the optimizer through ``add_loss``, so stock ``compile`` and
``fit`` sum it in and scale it under ``mixed_float16``. Three things a caller
needs: chunk caps are fixed per level, so ``max_chunks`` is a constructor
argument with a derived default; the constructor flag is
``tie_word_embeddings`` while the architecture dataclass keeps the reference's
``tie_embeddings``, which the flag defaults to; and ``pretrained=True`` raises,
because no H-Net checkpoint ships here (a local ``.keras`` path restores the
backbone). With ``include_top=True`` and a ``task_config`` the model owns one
task head, named ``task_head``, in place of the logits.

References:
    - Hwang et al., 2025. Dynamic Chunking for End-to-End Hierarchical Sequence
      Modeling. (https://arxiv.org/abs/2507.07955)
    - Reference implementation: ``hnet/models/mixer_seq.py:22-62`` (the LM wrapper
      and its init asymmetry) and ``hnet/models/hnet.py:121-147`` (the per-stage
      depth scaling).
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import keras

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.models.language.hnet.config import (
    MODEL_VARIANTS,
    MIXER_ATTENTION,
    HNetArchConfig,
    get_variant_config,
    n_residuals_by_stage,
)
from dl_techniques.models.language.hnet.losses import (
    DEFAULT_TARGET_RATIO,
    total_ratio_loss,
)
from dl_techniques.models.language.hnet.stage import HNetStage
from dl_techniques.layers.heads.nlp import (
    NLPTaskConfig,
    build_model_task_head,
    finalize_head_outputs,
    head_attention_mask,
    task_config_to_dict,
    validate_task_head_knobs,
)
from dl_techniques.layers.heads.nlp.model_head import (
    TASK_HEAD_NAME,
    head_output_shape,
    ids_shape_of,
)
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.sequence_length_guard import (
    check_sequence_length,
    refuse_static_sequence_length,
)
from dl_techniques.utils.tied_embeddings import tied_embedding_logits
from dl_techniques.utils.weight_transfer import (
    load_backbone_from_checkpoint,
    pretrained_path_or_raise,
)

# ---------------------------------------------------------------------

__all__ = [
    "EMBEDDING_INIT_STDDEV",
    "INITIALIZER_RANGE",
    "MODEL_VARIANTS",
    "RATIO_LOSS_ALPHA",
    "RESIDUAL_WRITING_PROJECTIONS",
    "HNet",
    "create_hnet",
    "default_max_chunks",
]

# Re-exported, not redefined: this is `config.MODEL_VARIANTS`, the same object.

#: Standard deviation for the embedding matrix, from ``mixer_seq.py:60``. Embeddings
#: are initialised at 1.0 while Linears use :data:`INITIALIZER_RANGE`.
EMBEDDING_INIT_STDDEV: float = 1.0

#: Base standard deviation for every Linear, from ``mixer_seq.py:53``, before the
#: per-stage depth scaling is applied to the residual-writing ones.
INITIALIZER_RANGE: float = 0.02

#: Coefficient on the summed ratio loss; the per-level targets are the
#: ``target_ratios`` constructor argument. The reference training script that would set
#: this coefficient is not released, and grepping the whole reference repository for
#: ``0.03`` returns zero hits -- so, like
#: :data:`~dl_techniques.models.language.hnet.losses.DEFAULT_TARGET_RATIO`, this is a
#: documented default of this port and not a transcription from the reference code.
#: ``0.03`` is the load-balancing weight the paper reports.
RATIO_LOSS_ALPHA: float = 0.03

#: Dense names that write into the residual stream and so take the depth-scaled
#: initializer: the Mamba-2 output projection, the attention output projection and the
#: SwiGLU down projection. Every other Dense reads the stream and takes the base range.
RESIDUAL_WRITING_PROJECTIONS: Tuple[str, ...] = ("out_proj", "w_o", "down_proj")

# ---------------------------------------------------------------------

def default_max_chunks(num_stages: int, max_seq_len: int) -> Tuple[int, ...]:
    """Derive a chunk cap per chunking level by successive halving.

    The cap is ``max_seq_len // 2`` at the first chunking level and half again at
    each level below it, floored at 1. That allows 2x compression per level, three
    times looser than this port's ratio-loss target of
    :data:`~dl_techniques.models.language.hnet.losses.DEFAULT_TARGET_RATIO` = 6.0, so
    truncation stays uncommon while three quarters of the inner stage's compute is
    removed at every level. These are defaults: a real training run should measure the
    boundary counts it realises on its own corpus and pass ``max_chunks`` explicitly.

    :param num_stages: Total stages, innermost included. There are ``num_stages - 1``
        chunking levels.
    :type num_stages: int
    :param max_seq_len: The longest input sequence the model will see.
    :type max_seq_len: int
    :returns: One cap per chunking level, outermost first. Empty for a 1-stage layout.
    :rtype: Tuple[int, ...]
    """
    caps: List[int] = []
    length = int(max_seq_len)
    for _ in range(max(0, int(num_stages) - 1)):
        length = max(1, length // 2)
        caps.append(length)
    return tuple(caps)

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.hnet.model")
class HNet(keras.Model):
    """Byte-level causal language model with hierarchical dynamic chunking.

    An embedding feeds the recursive backbone, which returns the hidden states and
    one routing record per chunking level. The records become the ratio loss, which
    is contributed with ``add_loss`` rather than in a custom training step. The head
    is a Dense, or, when tied, a matmul against the embedding table. With
    ``include_top=True`` and a ``task_config`` the model owns ONE task head from
    ``layers/heads/nlp``, named ``task_head``, instead: it pools ``hidden`` under a
    padding mask and ``call`` returns its finalized output; the logits (and an untied
    ``lm_head``) do not exist then, while the ratio loss is contributed all the same.

    Architecture:

    .. code-block:: text

        input_ids [B, L] int32
                 │
                 ▼
        ┌─────────────────────┐
        │ embeddings          │  stddev 1.0
        └─────────────────────┘
                 │  [B, L, d_model[0]]
                 ▼
        ┌─────────────────────┐
        │ backbone  HNetStage │  recursive hierarchy
        └─────────────────────┘
                 │  hidden [B, L, d_model[0]]
                 ├──────────► routing records, one per level
                 │                        │
                 │                        ▼
                 │            ratio_loss * alpha ──► add_loss
                 ▼
           ┌─────┴─────┐
           ▼           ▼
          tied       untied
           │           │
           ▼           ▼
    hidden @ E^T     lm_head
           │           │
           └─────┬─────┘
                 ▼
           logits [B, L, vocab_size]

    The tied path owns no weights; the untied path is a Dense with no bias. With
    ``include_top=True`` the diagram's lower half (tied / untied) is replaced by
    ``task_head`` (``hidden`` and the padding mask in, the head's output out).

    Initializer scaling:

    .. code-block:: text

        Dense in an isotropic stack
                  │
          ┌───────┴───────┐
          ▼               ▼
       writes            reads
       residual          residual
       out_proj          in_proj, w_q, w_k, w_v
       w_o               gate_proj, up_proj
       down_proj
          │               │
          ▼               ▼
     range / sqrt(n_k)   range

    n_k is the outside-in cumulative residual count at stage k.

    :param arch_config: The parsed architecture -- layout, per-stage widths and per-stage
        attention/SSM knobs.
    :type arch_config: HNetArchConfig
    :param max_chunks: One fixed chunk cap per chunking level, outermost first, so
        ``len(max_chunks) == arch_config.num_stages - 1``. ``None`` derives them with
        :func:`default_max_chunks`.
    :type max_chunks: Optional[Sequence[int]]
    :param max_seq_len: Largest position the attention RoPE tables cover, and the length
        :func:`default_max_chunks` halves when ``max_chunks`` is ``None``.
    :type max_seq_len: int
    :param headdim: Mamba-2 SSM head width.
    :type headdim: int
    :param tie_word_embeddings: Share the embedding matrix with the LM head. ``None``
        takes the architecture's own ``tie_embeddings`` (the reference's spelling).
    :type tie_word_embeddings: Optional[bool]
    :param ratio_loss_alpha: Coefficient on the summed ratio loss.
    :type ratio_loss_alpha: float
    :param target_ratios: One target downsampling factor per chunking level. ``None``
        uses :data:`~dl_techniques.models.language.hnet.losses.DEFAULT_TARGET_RATIO`
        everywhere.
    :type target_ratios: Optional[Sequence[float]]
    :param initializer_range: Base standard deviation for every Linear in the stacks and
        for an untied LM head.
    :type initializer_range: float
    :param pad_token_id: Id treated as padding by the task head when ``call`` is given
        no mask. Read only with ``include_top=True``, and by the head alone (it never
        reaches the backbone). Defaults to None, which derives no padding (the head sees
        every position): id 0 is a real byte, so with right-padded batches pass the pad
        id or a mask.
    :type pad_token_id: Optional[int]
    :param include_top: Whether the model owns a task head. Defaults to False: no head
        layer and no head weight exist, and ``call`` returns the logits exactly as before.
    :type include_top: bool
    :param task_config: The head's task, as an ``NLPTaskConfig`` or a dict of its
        fields. Required when ``include_top`` is True, refused otherwise; stored as a
        plain dict. ``MULTIPLE_CHOICE`` is refused. A generation-type head's
        ``vocabulary_size`` is checked against the architecture's ``vocab_size``.
    :type task_config: Optional[Union[NLPTaskConfig, Dict[str, Any]]]
    :param head_config_overrides: Extra keyword arguments for the head constructor
        (e.g. ``{"pooling_type": "mean"}``, which replaces the ``'last'`` default). Only
        valid with ``include_top=True``.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :param **kwargs: Forwarded to :class:`keras.Model`.

    :raises TypeError: if ``arch_config`` is not an :class:`HNetArchConfig`.
    :raises ValueError: if ``max_chunks`` or ``target_ratios`` has the wrong length, if
        ``ratio_loss_alpha`` or ``initializer_range`` is negative, if a stack contains
        no residual-writing projection to depth-scale (which would mean a sub-layer was
        renamed out from under :data:`RESIDUAL_WRITING_PROJECTIONS`), if a Dense in a
        stack is already built when the depth-scaled init runs, if ``include_top=True``
        comes without a ``task_config``, or if a ``task_config`` or
        ``head_config_overrides`` comes with ``include_top=False``.

    Input:
        ``(batch, seq_len)`` integer byte ids, or a dict holding them under
        ``'input_ids'`` with an optional ``'attention_mask'`` (the same thing as the
        ``padding_mask`` argument of ``call``, which wins when both are given).

    Padding:
        A mask reaches the backbone (boundary selection, chunking, attention and the ratio
        loss ignore masked positions) and, with ``include_top=True``, the task head. It is
        supported for RIGHT padding only: the Mamba-2 mixer takes no mask, so a position
        after interior or left padding depends on the padded tokens' values. Measured, at
        real positions of a right-padded row the hidden states equal the same row run
        alone to about 1e-7, and the padded token VALUES change nothing.

    Example:
        >>> from dl_techniques.models.language.hnet.config import HNetArchConfig, AttnSpec
        >>> cfg = HNetArchConfig(
        ...     arch_layout=["m1", ["T1"], "m1"],
        ...     d_model=[16, 16],
        ...     d_intermediate=[0, 0],
        ...     attn_cfg=AttnSpec(num_heads=(2, 2), rotary_emb_dim=(4, 4),
        ...                       window_size=(-1, -1)),
        ... )
        >>> model = HNet(cfg, max_chunks=(8,), max_seq_len=32, headdim=8)
        >>> model(keras.ops.zeros((2, 16), dtype="int32")).shape
        (2, 16, 256)
    """

    #: The six shipped variants. The same object as
    #: :data:`~dl_techniques.models.language.hnet.config.MODEL_VARIANTS`, aliased onto
    #: the class so ``from_variant`` and the API tests reach the table through the class.
    MODEL_VARIANTS: Dict[str, HNetArchConfig] = MODEL_VARIANTS

    _SEQUENCE_LENGTH_MESSAGE = (
        "seq_len={seq_len} exceeds max_seq_len={limit}. HNet's stage-0 attention "
        "precomputes its RoPE tables for {limit} positions, so there is no "
        "rotation for position {limit}. Truncate the input to {limit} bytes, "
        "or build the model with max_seq_len>={seq_len}."
    )

    def __init__(
            self,
            arch_config: HNetArchConfig,
            max_chunks: Optional[Sequence[int]] = None,
            max_seq_len: int = 2048,
            headdim: int = 64,
            tie_word_embeddings: Optional[bool] = None,
            ratio_loss_alpha: float = RATIO_LOSS_ALPHA,
            target_ratios: Optional[Sequence[float]] = None,
            initializer_range: float = INITIALIZER_RANGE,
            pad_token_id: Optional[int] = None,
            include_top: bool = False,
            task_config: Optional[Union[NLPTaskConfig, Dict[str, Any]]] = None,
            head_config_overrides: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(arch_config, HNetArchConfig):
            raise TypeError(
                f"arch_config must be an HNetArchConfig, got "
                f"{type(arch_config).__name__}"
            )
        if ratio_loss_alpha < 0.0:
            raise ValueError(
                f"ratio_loss_alpha must be non-negative, got {ratio_loss_alpha}"
            )
        if initializer_range <= 0.0:
            raise ValueError(
                f"initializer_range must be positive, got {initializer_range}"
            )

        validate_task_head_knobs(include_top, task_config, head_config_overrides)

        n_levels = arch_config.num_stages - 1
        if max_chunks is None:
            max_chunks = default_max_chunks(arch_config.num_stages, max_seq_len)
        max_chunks = tuple(int(value) for value in max_chunks)
        if len(max_chunks) != n_levels:
            raise ValueError(
                f"max_chunks needs one entry per chunking level: expected {n_levels} "
                f"for a {arch_config.num_stages}-stage layout, got {len(max_chunks)} "
                f"({max_chunks})"
            )

        if target_ratios is None:
            target_ratios = (DEFAULT_TARGET_RATIO,) * n_levels
        target_ratios = tuple(float(value) for value in target_ratios)
        if len(target_ratios) != n_levels:
            raise ValueError(
                f"target_ratios needs one entry per chunking level: expected "
                f"{n_levels} for a {arch_config.num_stages}-stage layout, got "
                f"{len(target_ratios)} ({target_ratios})"
            )

        self.arch_config = arch_config
        self.max_chunks = max_chunks
        self.max_seq_len = max_seq_len
        self.headdim = headdim
        self.tie_word_embeddings = (
            bool(arch_config.tie_embeddings)
            if tie_word_embeddings is None
            else bool(tie_word_embeddings)
        )
        self.ratio_loss_alpha = float(ratio_loss_alpha)
        self.target_ratios = target_ratios
        self.initializer_range = float(initializer_range)

        self.vocab_size = arch_config.vocab_size
        self.d_embed = arch_config.d_model[0]

        self.pad_token_id = None if pad_token_id is None else int(pad_token_id)
        self.include_top = bool(include_top)
        self.task_config = (
            task_config_to_dict(task_config, vocab_size=self.vocab_size)
            if task_config is not None else None
        )
        self.head_config_overrides = (
            dict(head_config_overrides) if head_config_overrides is not None else None
        )

        # The backbone maps (B, L, D0) to (B, L, D0), so the embedding sits outside it.
        self.embeddings = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.d_embed,
            embeddings_initializer=keras.initializers.RandomNormal(
                mean=0.0, stddev=EMBEDDING_INIT_STDDEV
            ),
            name="embeddings",
        )

        self.backbone = HNetStage(
            arch_config=arch_config,
            stage_idx=0,
            max_chunks=max_chunks,
            max_seq_len=max_seq_len,
            headdim=headdim,
            name="backbone",
        )

        # A tied head owns no weights: `call` projects against the embedding table, so a
        # Dense here would be a trained-and-discarded parameter block. For the same reason
        # a model that owns a task head builds no `lm_head` either, tied or not.
        self.lm_head: Optional[keras.layers.Dense] = None
        if not self.tie_word_embeddings and not self.include_top:
            self.lm_head = keras.layers.Dense(
                self.vocab_size,
                use_bias=False,
                kernel_initializer=keras.initializers.RandomNormal(
                    mean=0.0, stddev=self.initializer_range
                ),
                name="lm_head",
            )

        self._apply_depth_scaled_init()

        # Created after the backbone and under its own fixed name, so the backbone's
        # layer names and weight paths are the same with or without a head.
        self.task_head = None
        if self.include_top:
            # DECISION plan-2026-10-11T170958-dc4d2202/D-005: headless stays the logits and the head
            # reads `hidden` (position-aligned with the mask, right padding only); do NOT feed it the
            # logits, build the untied `lm_head` beside it, or keep the head's 'cls' default --
            # position 0 of a causal model has seen only itself. See decisions.md.
            self.task_head = build_model_task_head(
                self.task_config,
                input_dim=self.d_embed,
                head_config_overrides=self.head_config_overrides,
                vocab_size=self.vocab_size,
                default_pooling="last",
            )

    # -----------------------------------------------------------------
    # initialisation
    # -----------------------------------------------------------------

    def _apply_depth_scaled_init(self) -> None:
        """Give every Dense in the isotropic stacks its initializer, scaled by depth.

        Walks the stage chain outside-in. At each stage the stacks are the encoder and
        the decoder, or the single main stack at the innermost stage, and every Dense in
        them is re-initialised: ``initializer_range / sqrt(n_k)`` for the residual-writing
        projections named in :data:`RESIDUAL_WRITING_PROJECTIONS`, plain
        ``initializer_range`` for every other one. Follows ``hnet/models/hnet.py:121-147``.

        :raises ValueError: if a stack contains no residual-writing projection, or if a
            Dense has already been built when this runs.
        """
        # DECISION plan-2026-09-09T042752-6d66ac56/D-021: the denominator is the per-stage
        # cumulative count; the hierarchy total makes stage 0 3.04x too small. See decisions.md.
        counts = n_residuals_by_stage(self.arch_config.stage_spec)

        stage = self.backbone
        for stage_idx, n_res in enumerate(counts):
            scaled = self.initializer_range / math.sqrt(float(n_res))
            if stage.is_innermost:
                stacks = [stage.main_network]
            else:
                stacks = [stage.encoder, stage.decoder]

            for stack in stacks:
                self._set_stack_initializers(stack, scaled, stage_idx)

            if stage.is_innermost:
                break
            stage = stage.main_network

    def _set_stack_initializers(
            self, stack: keras.layers.Layer, scaled_stddev: float, stage_idx: int
    ) -> None:
        """Re-initialise every :class:`keras.layers.Dense` inside one isotropic stack.

        :param stack: An
            :class:`~dl_techniques.models.language.hnet.components.HNetIsotropic`.
        :type stack: keras.layers.Layer
        :param scaled_stddev: The depth-scaled standard deviation for this stage.
        :type scaled_stddev: float
        :param stage_idx: Only used in error messages.
        :type stage_idx: int
        :raises ValueError: if the stack holds no residual-writing projection, or a Dense
            in it is already built.
        """
        n_scaled = 0
        for sub in stack._flatten_layers(include_self=True):
            if not isinstance(sub, keras.layers.Dense):
                continue
            if sub.built:
                raise ValueError(
                    f"Dense {sub.name!r} in stage {stage_idx}'s stack {stack.name!r} is "
                    f"already built, so replacing its kernel_initializer would be a "
                    f"silent no-op; the depth-scaled init must run before any variable "
                    f"is created"
                )
            writes_residual = sub.name in RESIDUAL_WRITING_PROJECTIONS
            stddev = scaled_stddev if writes_residual else self.initializer_range
            n_scaled += int(writes_residual)
            # DECISION plan-2026-09-09T042752-6d66ac56/D-021: set the initializer before the
            # variables exist; an assign inside a parent build() is discarded. See decisions.md.
            sub.kernel_initializer = keras.initializers.RandomNormal(
                mean=0.0, stddev=stddev
            )

        if n_scaled == 0:
            # A rename upstream would otherwise leave the whole stack unscaled in silence.
            raise ValueError(
                f"stage {stage_idx}'s stack {stack.name!r} contains no Dense named one "
                f"of {RESIDUAL_WRITING_PROJECTIONS}, so nothing would be depth-scaled. "
                f"A residual-writing projection has been renamed upstream; update "
                f"RESIDUAL_WRITING_PROJECTIONS in "
                f"dl_techniques.models.language.hnet.model"
            )

    # -----------------------------------------------------------------
    # build / call
    # -----------------------------------------------------------------

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the embedding, the backbone and the untied head or the task head.

        :param input_shape: ``(batch, seq_len)`` -- the shape of the int32 byte ids.
            Both entries may be ``None``. A dict holding it under ``'input_ids'`` is
            accepted too.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: if the input is not rank 2.
        """
        if self.built:
            return
        input_shape = ids_shape_of(input_shape)
        if len(input_shape) != 2:
            raise ValueError(
                f"HNet expects rank-2 (batch, seq_len) integer input, got shape "
                f"{input_shape}"
            )

        # First, so the cached step functions are dropped and no variable exists.
        if self._stage0_has_attention:
            refuse_static_sequence_length(
                self, input_shape[1], self.max_seq_len,
                self._SEQUENCE_LENGTH_MESSAGE.format(
                    seq_len=input_shape[1], limit=self.max_seq_len
                ),
            )
        self.embeddings.build(input_shape)
        hidden_shape = (input_shape[0], input_shape[1], self.d_embed)
        self.backbone.build(hidden_shape)
        if self.lm_head is not None:
            self.lm_head.build(hidden_shape)
        if self.task_head is not None:
            self.task_head.build({
                "hidden_states": hidden_shape,
                "attention_mask": tuple(input_shape),
            })
        super().build(input_shape)

    def call(
            self,
            inputs: Any,
            padding_mask: Optional[Any] = None,
            training: Optional[bool] = None,
    ) -> Any:
        """Map byte ids to next-token logits, or pool them through the task head,
        contributing the ratio loss on the way either way.

        :param inputs: ``(B, L)`` integer byte ids in ``[0, vocab_size)``, or a dict
            holding them under ``'input_ids'`` with an optional ``'attention_mask'``.
        :type inputs: Any
        :param padding_mask: Optional ``(B, L)`` validity mask, truthy at a real token
            (right padding only, see the class docstring). Excluded from boundary
            selection, chunking, attention and the ratio loss, and read by the task head
            as its mask. Overrides a dict input's ``'attention_mask'``.
        :type padding_mask: Optional[Any]
        :param training: Training-mode flag, forwarded to every sub-layer.
        :type training: Optional[bool]
        :returns: With ``include_top=False``, ``(B, L, vocab_size)`` logits. With
            ``include_top=True``, only the head's output reduced by
            ``finalize_head_outputs``: a bare tensor, or a dict of two or more tensors.
        :rtype: Any
        :raises ValueError: if a dict input has no ``'input_ids'`` key.
        """
        if isinstance(inputs, dict):
            if "input_ids" not in inputs:
                raise ValueError("Dictionary input must contain 'input_ids' key")
            if padding_mask is None:
                padding_mask = inputs.get("attention_mask")
            inputs = inputs["input_ids"]

        if self._stage0_has_attention:
            inputs = check_sequence_length(
                self, inputs, self.max_seq_len, "max_seq_len",
                self._SEQUENCE_LENGTH_MESSAGE,
            )
        hidden = self.embeddings(inputs)

        hidden, routing_records = self.backbone(
            hidden, padding_mask=padding_mask, training=training
        )

        if routing_records:
            # add_loss, not a custom train_step: stock fit() already sums model.losses and
            # applies scale_loss under mixed_float16.
            ratio = total_ratio_loss(routing_records, self.target_ratios)
            alpha = keras.ops.cast(self.ratio_loss_alpha, ratio.dtype)
            self.add_loss(keras.ops.cast(alpha * ratio, "float32"))

        if self.include_top:
            head_outputs = self.task_head(
                {
                    "hidden_states": hidden,
                    "attention_mask": head_attention_mask(
                        inputs, padding_mask, self.pad_token_id, hidden.dtype
                    ),
                },
                training=training,
            )
            return finalize_head_outputs(head_outputs)

        if self.tie_word_embeddings:
            embedding_weights = self.embeddings.embeddings
            return tied_embedding_logits(hidden, embedding_weights)
        return self.lm_head(hidden)

    @property
    def _stage0_has_attention(self) -> bool:
        """Whether the outermost stage has an attention block: the only RoPE limit."""
        top = self.arch_config.stage_spec
        stacks = (top.main,) if top.is_innermost else (top.encoder, top.decoder)
        return any(b.mixer == MIXER_ATTENTION for s in stacks for b in s.blocks)

    def compute_output_shape(self, input_shape: Any) -> Any:
        """Return the shapes of what :meth:`call` returns for a rank-2 input shape.

        :param input_shape: ``(batch, seq_len)``, or a dict holding it under
            ``'input_ids'``.
        :type input_shape: Any
        :returns: Headless, the logits shape ``(batch, seq_len, vocab_size)``. With a
            head, the head's output shape after the reduction ``call`` applies: one
            shape, or a mapping of two or more.
        :rtype: Any
        """
        ids_shape = ids_shape_of(input_shape)
        if self.include_top:
            return head_output_shape(self.task_head, ids_shape, self.d_embed)
        return (ids_shape[0], ids_shape[1], self.vocab_size)

    @property
    def hidden_size(self) -> int:
        """Alias for :attr:`d_embed`, for callers expecting the common name.

        Additive only: ``d_embed`` is not renamed or removed, and this is not
        a constructor argument, so ``get_config``/``from_config`` are
        unaffected. Mirrors ``Mamba2Model.hidden_size``'s alias of
        ``d_model``. Non-blocking for
        ``dl_techniques.models.common.masked_language_model.clm.CausalLanguageModel``,
        whose ``__init__`` only requires ``hidden_size`` when built with
        ``skip_head=False`` -- a headless HNet already bakes its own logits head, so
        it is wrapped with ``skip_head=True`` -- but adds attribute-surface
        parity with every other migrated backbone.

        :returns: :attr:`d_embed`.
        :rtype: int
        """
        return self.d_embed

    # -----------------------------------------------------------------
    # variants
    # -----------------------------------------------------------------

    @classmethod
    def from_variant(
            cls,
            variant: str,
            pretrained: Union[bool, str] = False,
            **kwargs: Any,
    ) -> "HNet":
        """Build one of the six shipped variants.

        :param variant: A key of
            :data:`~dl_techniques.models.language.hnet.config.MODEL_VARIANTS`.
        :type variant: str
        :param pretrained: If a string, the path of a local ``.keras`` checkpoint: the
            backbone (the embedding and the stage stack, by layer name) is restored from
            it, headed or headless; a ``task_head`` in the checkpoint is never copied and
            the model's own head keeps its fresh initialization, so a checkpoint of any
            HNet, headless or trained under any head, warm-starts a model with any other
            head or none. The backbone loads completely or the call raises: a checkpoint
            of another depth or width is refused, never truncated. An untied ``lm_head``
            is a top: it may be left out on either side only when one side carries a task
            head instead (or the vocabulary changed), so a tied headless checkpoint does
            not fill an untied model. To restore a whole model, head included, use
            ``keras.models.load_model``. If True, raises ``NotImplementedError``: no
            H-Net weights are distributed with this repository. Defaults to False.
        :type pretrained: Union[bool, str]
        :param **kwargs: Forwarded to :meth:`__init__` -- ``max_chunks``, ``max_seq_len``,
            ``target_ratios``, ``include_top``, ``task_config`` and so on.
        :returns: The model, randomly initialised, or with the checkpoint's backbone.
        :rtype: HNet
        :raises ValueError: if ``variant`` is not a known name (the message lists every
            available name), or a ``pretrained`` checkpoint does not fit the backbone.
        :raises NotImplementedError: if ``pretrained`` is True.
        :raises FileNotFoundError: if a ``pretrained`` path does not exist.
        """
        config = get_variant_config(variant)
        if pretrained and not isinstance(pretrained, str):
            raise NotImplementedError(
                f"no pretrained weights are distributed for H-Net variant {variant!r}: "
                f"this repository ships no H-Net checkpoint and the reference "
                f"checkpoints are not loadable here (the RoPE pairing and the chunk-cap "
                f"layout both diverge). Build with pretrained=False and train, or pass "
                f"pretrained=<local .keras path> to restore a backbone of your own."
            )
        weights_path = pretrained_path_or_raise(pretrained, "H-Net")
        model = cls(arch_config=config, **kwargs)
        if weights_path is not None:
            # A build, not a forward: no RNG is drawn and no loss is added.
            if not model.built:
                model.build((None, None))
            # DECISION plan-2026-10-11T170958-dc4d2202/D-005: by name and strict through the shared
            # rule; do NOT use model.load_weights (restores nothing on a mismatch and reports
            # success), drop TASK_HEAD_NAME (a head of equal shape would be copied) or drop the
            # `lm_head` top (a tied headless checkpoint would then fill an untied model with
            # nothing, and an untied one would be dropped into a tied model). See decisions.md.
            load_backbone_from_checkpoint(
                model,
                weights_path,
                head_prefixes=(TASK_HEAD_NAME,),
                vocab_layers=("embeddings",),
                tables=("embeddings/embeddings", "lm_head/kernel"),
                tops=("lm_head",),
            )
        return model

    # -----------------------------------------------------------------
    # serialization
    # -----------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return every constructor argument, JSON-safe.

        :returns: The configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "arch_config": self.arch_config.to_dict(),
            "max_chunks": list(self.max_chunks),
            "max_seq_len": self.max_seq_len,
            "headdim": self.headdim,
            "tie_word_embeddings": self.tie_word_embeddings,
            "ratio_loss_alpha": self.ratio_loss_alpha,
            "target_ratios": list(self.target_ratios),
            "initializer_range": self.initializer_range,
            "pad_token_id": self.pad_token_id,
            "include_top": self.include_top,
            "task_config": (
                dict(self.task_config) if self.task_config is not None else None
            ),
            "head_config_overrides": (
                dict(self.head_config_overrides)
                if self.head_config_overrides is not None else None
            ),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HNet":
        """Rebuild from :meth:`get_config`, reconstructing the architecture dataclass.

        :param config: The dictionary produced by :meth:`get_config`.
        :type config: Dict[str, Any]
        :returns: The rebuilt model.
        :rtype: HNet
        """
        config = dict(config)
        arch_config = config.pop("arch_config")
        if not isinstance(arch_config, HNetArchConfig):
            arch_config = HNetArchConfig.from_dict(arch_config)
        return cls(arch_config=arch_config, **config)


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


def create_hnet(
        variant: str = "hnet_1stage_L",
        pretrained: Union[bool, str] = False,
        **kwargs: Any,
) -> HNet:
    """Create an H-Net from a shipped variant name.

    Delegates to :meth:`HNet.from_variant` and holds no defaulting, validation or
    construction logic of its own.

    :param variant: A key of
        :data:`~dl_techniques.models.language.hnet.config.MODEL_VARIANTS`.
    :type variant: str
    :param pretrained: A local ``.keras`` path to restore the backbone from, or
        ``False``; ``True`` raises. See :meth:`HNet.from_variant`.
    :type pretrained: Union[bool, str]
    :param **kwargs: Forwarded to :meth:`HNet.from_variant`, including
        ``include_top=True`` with a ``task_config`` for a model with a task head.
    :returns: The model.
    :rtype: HNet
    :raises ValueError: if ``variant`` is not a known name.
    :raises NotImplementedError: if ``pretrained`` is True.

    Example:
        >>> model = create_hnet("hnet_1stage_L", max_seq_len=1024)
    """
    return HNet.from_variant(variant, pretrained=pretrained, **kwargs)

# ---------------------------------------------------------------------
