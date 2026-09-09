"""H-Net: the assembled byte-level causal language model.

This module is the top of the port. It is deliberately thin -- an embedding, the recursive
backbone built in ``stage.py``, and a language-modelling head -- because every piece of
architecture already lives one level down:

.. code-block:: text

    input_ids (B, L) int32
          |
    Embedding(256, d_model[0])          std = 1.0   <-- NOT 0.02
          |
    HNetStage(stage_idx=0)              the whole recursive hierarchy
          |                             -> hidden (B, L, d_model[0])
          |                             -> routing records, one per chunking level
          |                                    |
          |                                    +--> ratio_loss * alpha --> add_loss
          |
    lm_head                             Dense(256, no bias, std = 0.02)
          |                             or, tied, hidden @ embeddings.T
          v
    logits (B, L, 256)

Four things here are load-bearing and none of them has a shape symptom
---------------------------------------------------------------------

1. **The embedding is initialised at ``stddev = 1.0``, the LM head at ``0.02``.** This
   asymmetry is the reference's, stated in one place (``mixer_seq.py:55-62``): linears get
   ``initializer_range``, embeddings get 1.0. Giving the embedding 0.02 shrinks every
   input to the first stage by 50x and the model still trains -- badly, and with nothing
   failing.
2. **Residual-writing projections are depth-scaled PER STAGE**, ``0.02 / sqrt(n_k)`` where
   ``n_k`` is the OUTSIDE-IN CUMULATIVE residual count at stage ``k``. See the
   ``# DECISION`` anchor on :meth:`HNet._apply_depth_scaled_init`.
3. **The ratio loss reaches the optimizer through ``add_loss``**, never through a custom
   ``train_step``. Overriding ``train_step`` is a repository-wide invariant violation, and
   under ``mixed_float16`` it silently discards the framework's own ``scale_loss``, i.e.
   ``2**15`` of gradient magnitude.
4. **``pretrained=True`` raises.** No H-Net checkpoint is distributed with this repository
   and none is convertible from the reference (the RoPE pairing alone diverges, D-005), so
   the alternative -- warning and returning random weights -- would let a caller publish
   numbers from an untrained model.

The reference spells the weight-tying flag ``tie_embeddings``; this port spells the
constructor flag ``tie_word_embeddings``, the house spelling in three of the four models
here that tie (decision D-006). The architecture dataclass keeps the reference's spelling,
and the constructor flag defaults to it.

References:
    - Hwang et al., 2025. Dynamic Chunking for End-to-End Hierarchical Sequence
      Modeling. (https://arxiv.org/abs/2507.07955)
    - Reference implementation: ``hnet/models/mixer_seq.py:22-62`` (the LM wrapper and its
      init asymmetry) and ``hnet/models/hnet.py:121-147`` (the per-stage depth scaling).
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import keras

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.models.language.hnet.config import (
    MODEL_VARIANTS,
    HNetArchConfig,
    get_variant_config,
    n_residuals_by_stage,
)
from dl_techniques.models.language.hnet.losses import (
    DEFAULT_TARGET_RATIO,
    total_ratio_loss,
)
from dl_techniques.models.language.hnet.stage import HNetStage
from dl_techniques.utils.keras_registration import register_dl_technique

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

# `MODEL_VARIANTS` is re-exported here, not redefined: the house shape puts the variant
# table on the model module, and a second copy of a six-row cited table is a copy that
# drifts. It is `config.MODEL_VARIANTS`, the same object.

#: ``mixer_seq.py:60`` -- the embedding matrix is initialised at unit standard deviation,
#: NOT at ``initializer_range``. The comment upstream reads "embeddings are initialized
#: differently from linears" and this is the whole of it.
EMBEDDING_INIT_STDDEV: float = 1.0

#: ``mixer_seq.py:53`` -- the base standard deviation for every Linear, before the
#: per-stage depth scaling is applied to the residual-writing ones.
INITIALIZER_RANGE: float = 0.02

#: The ratio-loss coefficient. One scalar for the whole hierarchy; the per-level targets
#: are the ``target_ratios`` constructor argument.
RATIO_LOSS_ALPHA: float = 0.03

#: The layer names that write INTO the residual stream, and therefore take the
#: depth-scaled initializer. Upstream selects them by substring on the PyTorch module
#: path (``"out_proj" in name or "fc2" in name``, ``hnet.py:127-129``); this port's
#: equivalents are the Mamba-2 output projection (``out_proj``), the attention output
#: projection (``w_o``, upstream's fused ``out_proj``) and the SwiGLU down projection
#: (``down_proj``, upstream's ``fc2``). Every other Dense in a stack -- ``in_proj``,
#: ``w_q``/``w_k``/``w_v``, ``gate_proj``, ``up_proj`` -- READS the residual stream and
#: takes the unscaled :data:`INITIALIZER_RANGE`.
RESIDUAL_WRITING_PROJECTIONS: Tuple[str, ...] = ("out_proj", "w_o", "down_proj")


def default_max_chunks(num_stages: int, max_seq_len: int) -> Tuple[int, ...]:
    """Derive a chunk cap per chunking level by successive halving.

    D-007 replaced the reference's data-dependent ``max(boundary_mask.sum(-1))`` with a
    FIXED cap, which makes the cap a constructor argument that
    :meth:`HNet.from_variant` must supply something for. The default derived here is
    ``max_seq_len // 2`` at the first chunking level and half again at each level below
    it, floored at 1.

    The derivation, so the number is not a guess: the paper targets a compression of
    roughly ``6x`` per stage and this port's ratio loss defaults to
    :data:`~dl_techniques.models.language.hnet.losses.DEFAULT_TARGET_RATIO` = 6.0, so a
    cap of ``2x`` compression carries a 3x margin over the target -- generous enough that
    truncation is not the common case, while still removing three quarters of the inner
    stage's compute at every level. It is a DEFAULT, not a property of the architecture:
    a real training run should measure its realised boundary counts on its own corpus and
    pass ``max_chunks`` explicitly.

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


@register_dl_technique("dl_techniques.models.hnet.model")
class HNet(keras.Model):
    """Byte-level causal language model with hierarchical dynamic chunking.

    :param arch_config: The parsed architecture -- layout, per-stage widths and per-stage
        attention/SSM knobs.
    :type arch_config: HNetArchConfig
    :param max_chunks: One fixed chunk cap per chunking level, outermost first, so
        ``len(max_chunks) == arch_config.num_stages - 1``. ``None`` derives them with
        :func:`default_max_chunks`.
    :type max_chunks: Optional[Sequence[int]]
    :param max_seq_len: Largest position the attention RoPE tables cover.
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
    :param initializer_range: Base standard deviation for every Linear in the stacks.
    :type initializer_range: float
    :param kwargs: Forwarded to :class:`keras.Model`.

    :raises TypeError: if ``arch_config`` is not an :class:`HNetArchConfig`.
    :raises ValueError: if ``max_chunks`` or ``target_ratios`` has the wrong length, if
        ``ratio_loss_alpha`` or ``initializer_range`` is negative, or if a stack contains
        no residual-writing projection to depth-scale (which would mean a sub-layer was
        renamed out from under :data:`RESIDUAL_WRITING_PROJECTIONS`).

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

        # `mixer_seq.py:38` -- the HNet backbone is a map (B, L, D0) -> (B, L, D0), so the
        # embedding lives OUTSIDE it.
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

        # A tied head owns no weights at all: the projection is written inline in `call`
        # against the embedding table (D-006). Creating an unused Dense here would add a
        # real, trained-and-discarded parameter block that no shape test would notice.
        self.lm_head: Optional[keras.layers.Dense] = None
        if not self.tie_word_embeddings:
            self.lm_head = keras.layers.Dense(
                self.vocab_size,
                use_bias=False,
                kernel_initializer=keras.initializers.RandomNormal(
                    mean=0.0, stddev=self.initializer_range
                ),
                name="lm_head",
            )

        self._apply_depth_scaled_init()

    # -----------------------------------------------------------------
    # initialisation
    # -----------------------------------------------------------------

    def _apply_depth_scaled_init(self) -> None:
        """Give every Dense in the isotropic stacks the reference's initializer.

        Walks the stage chain outside-in. At stage ``k`` the stacks are the encoder and
        decoder (or, at the innermost stage, the single main stack), and every Dense in
        them is re-initialised: ``initializer_range / sqrt(n_k)`` for the residual-WRITING
        projections named in :data:`RESIDUAL_WRITING_PROJECTIONS`, plain
        ``initializer_range`` for every other one. Transcribed from
        ``hnet/models/hnet.py:121-147``.

        :raises ValueError: if a stack contains no residual-writing projection, or if a
            Dense has already been built when this runs.
        """
        # DECISION plan-2026-09-09T042752-6d66ac56/D-021: the denominator is
        # `n_residuals_by_stage(...)` -- the OUTSIDE-IN CUMULATIVE count, one per stage --
        # and NOT the single hierarchy-wide `n_residuals(...)` total. Do NOT "simplify"
        # this to one number for the whole model. MEASURED against `hnet.py:121-147`: the
        # reference threads `parent_residuals` inward, so `hnet_1stage_L` scales by
        # (8, 52) and `hnet_2stage_XL` by (8, 20, 74). The two definitions coincide ONLY
        # at the innermost stage, which is exactly why a 1-stage config cannot tell them
        # apart -- and why the guard for this is a TWO-stage one
        # (test_model.py::TestDepthScaledInit::
        # test_a_two_stage_model_scales_each_stage_by_its_own_cumulative_count).
        # Using the total on a 2-stage model makes the outer sandwich's init
        # sqrt(74 / 8) = 3.04x too small, with no shape symptom and no exception.
        #
        # The initializer is REPLACED before the variables exist rather than assigned
        # after they do: an `.assign()` performed inside a parent-triggered `build()` is
        # recorded by Keras' StatelessScope and DISCARDED (measured here: a nested model's
        # post-build assign reverted to the initializer value while the same assign on a
        # directly-built model stuck). Setting the initializer has no such failure mode.
        # Rationale: decisions.md D-021.
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
            sub.kernel_initializer = keras.initializers.RandomNormal(
                mean=0.0, stddev=stddev
            )

        if n_scaled == 0:
            # A rename of `out_proj` / `w_o` / `down_proj` in any of the three upstream
            # layers would otherwise silently leave the whole stack unscaled. A guard
            # keyed on a name goes blind the moment the name moves, so it says so.
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
        """Build the embedding, the backbone and the untied head.

        :param input_shape: ``(batch, seq_len)`` -- the shape of the int32 byte ids.
            Both entries may be ``None``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: if the input is not rank 2.
        """
        if self.built:
            return
        if len(input_shape) != 2:
            raise ValueError(
                f"HNet expects rank-2 (batch, seq_len) integer input, got shape "
                f"{input_shape}"
            )

        self.embeddings.build(input_shape)
        hidden_shape = (input_shape[0], input_shape[1], self.d_embed)
        self.backbone.build(hidden_shape)
        if self.lm_head is not None:
            self.lm_head.build(hidden_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: Any,
            padding_mask: Optional[Any] = None,
            training: Optional[bool] = None,
    ) -> Any:
        """Map byte ids to next-token logits, contributing the ratio loss on the way.

        :param inputs: ``(B, L)`` integer byte ids in ``[0, vocab_size)``.
        :type inputs: Any
        :param padding_mask: Optional ``(B, L)`` validity mask, truthy at a real token.
            Excluded from boundary selection, chunking, attention and the ratio loss.
        :type padding_mask: Optional[Any]
        :param training: Training-mode flag, forwarded to every sub-layer.
        :type training: Optional[bool]
        :returns: ``(B, L, vocab_size)`` logits.
        :rtype: Any
        """
        hidden = self.embeddings(inputs)

        hidden, routing_records = self.backbone(
            hidden, padding_mask=padding_mask, training=training
        )

        if routing_records:
            # `add_loss`, never a custom `train_step`: stock `fit()` already sums
            # `model.losses` into the compiled loss AND applies `scale_loss` under
            # `mixed_float16`, which a hand-written training step silently skips.
            ratio = total_ratio_loss(routing_records, self.target_ratios)
            alpha = keras.ops.cast(self.ratio_loss_alpha, ratio.dtype)
            self.add_loss(keras.ops.cast(alpha * ratio, "float32"))

        if self.tie_word_embeddings:
            # `mixer_seq.py:51-53` via the house idiom (`gpt2.py:378-386`, D-006):
            # logits = hidden @ embeddings.T
            embedding_weights = self.embeddings.embeddings
            return keras.ops.matmul(
                hidden, keras.ops.transpose(keras.ops.cast(embedding_weights, hidden.dtype))
            )
        return self.lm_head(hidden)

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """:param input_shape: ``(batch, seq_len)``.
        :type input_shape: Tuple[Optional[int], ...]
        :returns: ``(batch, seq_len, vocab_size)``.
        :rtype: Tuple[Optional[int], ...]
        """
        return (input_shape[0], input_shape[1], self.vocab_size)

    # -----------------------------------------------------------------
    # variants
    # -----------------------------------------------------------------

    @classmethod
    def from_variant(
            cls,
            variant: str,
            pretrained: bool = False,
            **kwargs: Any,
    ) -> "HNet":
        """Build one of the six shipped variants.

        :param variant: A key of
            :data:`~dl_techniques.models.language.hnet.config.MODEL_VARIANTS`.
        :type variant: str
        :param pretrained: Must be ``False``. No H-Net weights are distributed with this
            repository.
        :type pretrained: bool
        :param kwargs: Forwarded to :meth:`__init__` -- ``max_chunks``, ``max_seq_len``,
            ``target_ratios`` and so on.
        :type kwargs: Any
        :returns: The model, randomly initialised.
        :rtype: HNet
        :raises ValueError: if ``variant`` is not a known name; the message lists every
            available name.
        :raises NotImplementedError: if ``pretrained`` is truthy.
        """
        config = get_variant_config(variant)
        if pretrained:
            raise NotImplementedError(
                f"no pretrained weights are distributed for H-Net variant {variant!r}: "
                f"this repository ships no H-Net checkpoint and the reference "
                f"checkpoints are not loadable here (the RoPE pairing and the chunk-cap "
                f"layout both diverge). Build with pretrained=False and train, or pass "
                f"a local .keras file to keras.models.load_model."
            )
        return cls(arch_config=config, **kwargs)

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
        pretrained: bool = False,
        **kwargs: Any,
) -> HNet:
    """Create an H-Net from a shipped variant name.

    Pure delegation to :meth:`HNet.from_variant` -- it holds no defaulting, no validation
    and no construction logic of its own, so there is exactly one place where a variant
    becomes a model.

    :param variant: A key of
        :data:`~dl_techniques.models.language.hnet.config.MODEL_VARIANTS`.
    :type variant: str
    :param pretrained: Must be ``False``; see :meth:`HNet.from_variant`.
    :type pretrained: bool
    :param kwargs: Forwarded to :meth:`HNet.from_variant`.
    :type kwargs: Any
    :returns: The model.
    :rtype: HNet

    Example:
        >>> model = create_hnet("hnet_1stage_L", max_seq_len=1024)
    """
    return HNet.from_variant(variant, pretrained=pretrained, **kwargs)
