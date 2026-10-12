"""Zamba2 decoder-stack assembly: the full causal language model.

Composes the four building blocks in ``layers.py`` -- :class:`LoRAAdapter`
(wrapped inside :class:`Zamba2SharedMLPBlock`), :class:`Zamba2SharedAttentionBlock`,
:class:`Zamba2SharedMLPBlock`, and :class:`Zamba2MambaBlock` -- into
:class:`Zamba2Model`, a ``keras.Model`` causal LM. The decoder walks a
``layer_mapping`` list of ``'m'``/``'g'`` tokens: every ``'m'`` position gets a
freshly-built, independently-weighted :class:`Zamba2MambaBlock`; every
``'g'`` position invokes one of ``num_mem_blocks`` physical
attention+MLP mem-block pairs, round-robin, with a monotonically increasing
global occurrence index that selects the MLP block's per-occurrence LoRA
delta. See ``plans/plan-2026-09-12T075714-035fd488/plan.md`` Step 5 and
``findings.md`` Key Constraints for the resolved design.

References:
    - Glorioso, P. et al., 2024. Zamba2: A Compact and Fast Hybrid Model.
      (https://arxiv.org/abs/2411.15242)
"""

from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import keras

# ---------------------------------------------------------------------
# variant tables
# ---------------------------------------------------------------------

#: Default vocabulary size -- Tiktoken ``cl100k_base``, matching this repo's
#: other subword-tokenized LMs (``gpt2``, ``colbert``).
DEFAULT_VOCAB_SIZE: int = 100277


def _build_layer_mapping(num_mamba_blocks: int, num_shared_occurrences: int) -> List[str]:
    """
    Build a ``'m'``/``'g'`` token list: mostly ``'m'`` with ``'g'`` inserted
    at evenly-spaced intervals, matching Zamba2's own ``layer_mapping``
    convention (predominantly Mamba2 blocks, a small number of shared
    attention+MLP mem-block invocations sprinkled through the stack) scaled
    down from the original config's roughly-every-6th-position density to
    this repo's smaller variant depths.

    :param num_mamba_blocks: Total number of ``'m'`` entries to place.
    :type num_mamba_blocks: int
    :param num_shared_occurrences: Total number of ``'g'`` entries to place,
        evenly distributed among the ``'m'`` entries.
    :type num_shared_occurrences: int
    :return: A list of length ``num_mamba_blocks + num_shared_occurrences``.
    :rtype: List[str]
    """
    group_size = max(num_mamba_blocks // num_shared_occurrences, 1)
    mapping: List[str] = []
    m_remaining = num_mamba_blocks
    for i in range(num_shared_occurrences):
        is_last = i == num_shared_occurrences - 1
        take = m_remaining if is_last else min(group_size, m_remaining)
        mapping.extend(["m"] * take)
        mapping.append("g")
        m_remaining -= take
    # Any leftover 'm' entries (only possible if num_shared_occurrences
    # exceeds num_mamba_blocks) are appended unsplit.
    mapping.extend(["m"] * m_remaining)
    return mapping


#: Public variant registry: repo-scale (mini/small/base), not the Zamba2
#: paper's own 2.7B/7B table (explicit non-goal, decisions.md D-002).
#: Keyed by short variant name; each value is a kwargs dict accepted by
#: :class:`Zamba2Model`'s constructor (minus ``layer_mapping``, which is
#: derived below from ``num_mamba_blocks``/``num_mem_block_occurrences`` so
#: the counts can never drift apart) plus the table-only
#: ``num_mem_block_occurrences`` key, popped by :meth:`Zamba2Model.from_variant`
#: before construction (see D-009).
#:
#: ``num_mem_blocks`` is the number of PHYSICAL shared attention+MLP
#: mem-block pairs the model builds; ``num_mem_block_occurrences`` is the
#: number of ``'g'`` positions in the derived ``layer_mapping`` -- i.e. how
#: many times those physical blocks are INVOKED, round-robin, across the
#: stack. Every shipped variant below has ``num_mem_block_occurrences``
#: strictly greater than ``num_mem_blocks`` (2x here), so round-robin reuse
#: actually happens on the public factory path; passing the same value for
#: both (the original defect, D-009) means every physical block is invoked
#: exactly once and the round-robin mechanism this package exists to
#: implement is never exercised by any shipped variant.
# DECISION plan-2026-09-12T075714-035fd488/D-008
# WHAT NOT TO DO: do not bind this as a plain `Dict[str, Dict[str, Any]]`
# literal again. `Zamba2Model.MODEL_VARIANTS = MODEL_VARIANTS` below is a
# class-attribute ALIAS of this module-level object (matching `hnet`'s
# `HNet.MODEL_VARIANTS` convention) -- a plain dict there is the exact "class
# attribute aliasing a module-level mutable" shape
# `tests/test_models/test_package_api_contract.py::TestNoMutableDefaults`
# forbids (review-iter-1.md concern 1; that guard's own docstring records the
# waiver set as EMPTY). `MappingProxyType` closes it the same way
# `fastvit/model.py`'s `MCI_VARIANTS`/`FastVitImageEncoder.MODEL_VARIANTS`
# alias does (D-079) -- wrapping the OUTER dict is sufficient; the guard's
# AST sweep only flags a literal `{...}`/`[...]`/`dict()`/`list()` binding,
# and a `MappingProxyType(...)` call is not one. See decisions.md D-008.
MODEL_VARIANTS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
    "zamba2_mini": {
        "vocab_size": DEFAULT_VOCAB_SIZE,
        "hidden_size": 256,
        "num_mamba_blocks": 8,
        "num_mem_blocks": 2,
        "num_mem_block_occurrences": 4,
        "num_heads": 4,
        "max_seq_len": 512,
        "lora_rank": 4,
        "lora_alpha": 8.0,
        "description": "Zamba2 mini: ~256-dim, 8 Mamba2 blocks, 2 physical "
                        "mem-blocks reused at 4 occurrences -- fast "
                        "tests/smoke runs.",
    },
    "zamba2_small": {
        "vocab_size": DEFAULT_VOCAB_SIZE,
        "hidden_size": 512,
        "num_mamba_blocks": 18,
        "num_mem_blocks": 3,
        "num_mem_block_occurrences": 6,
        "num_heads": 8,
        "max_seq_len": 1024,
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "description": "Zamba2 small: ~512-dim, 18 Mamba2 blocks, 3 "
                        "physical mem-blocks reused at 6 occurrences.",
    },
    "zamba2_base": {
        "vocab_size": DEFAULT_VOCAB_SIZE,
        "hidden_size": 768,
        "num_mamba_blocks": 24,
        "num_mem_blocks": 4,
        "num_mem_block_occurrences": 8,
        "num_heads": 12,
        "max_seq_len": 2048,
        "lora_rank": 16,
        "lora_alpha": 32.0,
        "description": "Zamba2 base: ~768-dim, 24 Mamba2 blocks, 4 "
                        "physical mem-blocks reused at 8 occurrences.",
    },
    }
)

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.tied_embeddings import tied_embedding_logits
from dl_techniques.utils.weight_transfer import (
    load_backbone_from_checkpoint,
    pretrained_path_or_raise,
)
from dl_techniques.utils.sequence_length_guard import (
    check_sequence_length,
    refuse_static_sequence_length,
)
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
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.models.language.zamba2.layers import (
    Zamba2MambaBlock,
    Zamba2SharedAttentionBlock,
    Zamba2SharedMLPBlock,
)

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.zamba2.model")
class Zamba2Model(keras.Model):
    """
    Zamba2 hybrid Mamba2 + shared-attention causal language model.

    Embeds token ids once, walks ``layer_mapping`` (a list of ``'m'``/``'g'``
    tokens, one per depth position) to route through a per-depth
    :class:`Zamba2MambaBlock` at every ``'m'`` position or one of
    ``num_mem_blocks`` shared attention+MLP mem-block pairs at every ``'g'``
    position (round-robin, each carrying its own LoRA delta via a global
    occurrence counter), applies a final :class:`RMSNorm`, and projects to
    vocabulary logits through the **tied** embedding matrix -- no separate
    output ``Dense`` layer. With ``include_top=True`` and a ``task_config`` the
    model instead owns ONE task head from ``layers/heads/nlp``, named
    ``task_head``, fed the final-norm output and a padding mask; the tied
    logits are then not computed and the head's output is returned.

    Architecture:

    .. code-block:: text

        input_ids [B, S]
               │
               ▼ Embedding(vocab_size, hidden_size)
        original_embedding [B, S, H]  ───────────────────────┐
               │                                             │  (re-read at
               ▼                                             │   every 'g')
        hidden_state = original_embedding                    │
               │                                             │
               ▼  for token in layer_mapping:                │
          ┌────┴─────────────────────────────────┐           │
          │ 'm': hidden_state =                  │           │
          │   Zamba2MambaBlock_i(hidden_state)   │  (fresh instance per 'm')
          │ 'g': hidden_state =                  │           │
          │   Zamba2SharedAttentionBlock_{i%N}(  │◄──────────┘
          │     hidden_state, original_embedding)│  (one of N=num_mem_blocks
          │   hidden_state =                     │   shared instances,
          │   Zamba2SharedMLPBlock_{i%N}(        │   round-robin; occurrence
          │     hidden_state,                    │   counter increments once
          │     occurrence_idx=<g counter>)      │   per 'g', independent of
          └────┬─────────────────────────────────┘   which physical slot)
               ▼
        final RMSNorm
               │
               ▼
        logits = hidden_state @ embedding.weights[0]^T   (tied LM head)
        [B, S, vocab_size]

        include_top=True: the logits are replaced by
        ┌───────────────────┐
        │ task_head         │──► finalized head output (only output)
        │ (pools 'last')    │    (hidden_state + padding mask in)
        └───────────────────┘

    :param vocab_size: Size of the token vocabulary. Must be positive.
    :type vocab_size: int
    :param hidden_size: Width of the decoder's hidden state (``d_model`` for
        every sub-block). Must be positive.
    :type hidden_size: int
    :param layer_mapping: Per-depth-position token list, each entry either
        ``'m'`` (a fresh :class:`Zamba2MambaBlock`) or ``'g'`` (one of the
        ``num_mem_blocks`` shared mem-block pairs, round-robin). Must be
        non-empty; every entry must be ``'m'`` or ``'g'``.
    :type layer_mapping: Sequence[str]
    :param num_mem_blocks: Number of physical shared attention+MLP mem-block
        pairs built. ``'g'`` positions route to
        ``self.mem_attention_blocks[occurrence_counter % num_mem_blocks]``
        (and the paired MLP block at the same index), so ``num_mem_blocks``
        may be smaller than the number of ``'g'`` entries in
        ``layer_mapping`` -- that is the round-robin sharing this class
        exists to implement. Must be positive.
    :type num_mem_blocks: int
    :param num_heads: Attention heads for every shared attention mem-block.
        Must be positive and divide ``hidden_size``.
    :type num_heads: int
    :param max_seq_len: Largest sequence length the shared attention
        mem-blocks' RoPE tables are built for. Must be positive.
    :type max_seq_len: int
    :param rope_theta: RoPE frequency base, forwarded to every shared
        attention mem-block. Defaults to 10000.0.
    :type rope_theta: float
    :param rope_percentage: Fraction of the per-head dimension RoPE rotates,
        forwarded to every shared attention mem-block. Defaults to 1.0.
    :type rope_percentage: float
    :param attention_dropout_rate: Dropout rate inside every shared
        attention mem-block's attention computation. Defaults to 0.0.
    :type attention_dropout_rate: float
    :param mem_block_norm_epsilon: Epsilon for the pre-attention and pre-MLP
        :class:`RMSNorm` inside every shared mem-block. Defaults to 1e-6.
    :type mem_block_norm_epsilon: float
    :param mlp_hidden_dim: Explicit gate/up-projection width for every
        shared MLP mem-block. When ``None`` (the default), each block
        derives it from ``hidden_size`` via the PaLM 2/3-rule arithmetic --
        see :class:`Zamba2SharedMLPBlock`.
    :type mlp_hidden_dim: Optional[int]
    :param ffn_expansion_factor: Expansion factor in the MLP mem-block's 2/3
        rule. Ignored when ``mlp_hidden_dim`` is given. Defaults to 4.
    :type ffn_expansion_factor: int
    :param ffn_multiple_of: The MLP mem-block's derived hidden width is
        rounded up to a multiple of this. Ignored when ``mlp_hidden_dim`` is
        given. Defaults to 256.
    :type ffn_multiple_of: int
    :param lora_rank: Bottleneck width of every shared MLP mem-block's owned
        :class:`LoRAAdapter`. Defaults to 8.
    :type lora_rank: int
    :param lora_alpha: LoRA scaling numerator of every shared MLP mem-block's
        owned :class:`LoRAAdapter`. Defaults to 16.0.
    :type lora_alpha: float
    :param d_state: SSM latent-state width, forwarded to every
        :class:`Zamba2MambaBlock`. Defaults to 128.
    :type d_state: int
    :param d_conv: Causal-conv kernel size, forwarded to every
        :class:`Zamba2MambaBlock`. Defaults to 4.
    :type d_conv: int
    :param expand: SSM internal expansion factor, forwarded to every
        :class:`Zamba2MambaBlock`. Defaults to 2.
    :type expand: int
    :param headdim: Per-SSM-head width, forwarded to every
        :class:`Zamba2MambaBlock`. Defaults to 64.
    :type headdim: int
    :param d_ssm: Number of dims the SSM scan runs on; forwarded to every
        :class:`Zamba2MambaBlock`. ``None`` resolves per-instance to
        ``hidden_size * expand``. Defaults to None.
    :type d_ssm: Optional[int]
    :param ngroups: Forwarded to every :class:`Zamba2MambaBlock`. Defaults
        to 1.
    :type ngroups: int
    :param mamba_norm_epsilon: Forwarded to every :class:`Zamba2MambaBlock`
        as its own internal pre-norm epsilon. Defaults to 1e-5.
    :type mamba_norm_epsilon: float
    :param mamba_rmsnorm: Forwarded to every :class:`Zamba2MambaBlock` as
        ``rmsnorm``. Defaults to True.
    :type mamba_rmsnorm: bool
    :param norm_before_gate: Forwarded to every :class:`Zamba2MambaBlock`.
        Defaults to False.
    :type norm_before_gate: bool
    :param dt_min: Forwarded to every :class:`Zamba2MambaBlock`. Defaults to
        0.001.
    :type dt_min: float
    :param dt_max: Forwarded to every :class:`Zamba2MambaBlock`. Defaults to
        0.1.
    :type dt_max: float
    :param dt_init_floor: Forwarded to every :class:`Zamba2MambaBlock`.
        Defaults to 1e-4.
    :type dt_init_floor: float
    :param mamba_bias: Forwarded to every :class:`Zamba2MambaBlock` as
        ``bias``. Defaults to False.
    :type mamba_bias: bool
    :param conv_bias: Forwarded to every :class:`Zamba2MambaBlock`. Defaults
        to True.
    :type conv_bias: bool
    :param use_bias: Whether the shared mem-blocks' internal ``Dense``
        projections use a bias term. Defaults to False.
    :type use_bias: bool
    :param kernel_initializer: Initializer for every ``Dense``/LoRA ``A``
        weight built by this model's sub-blocks. Defaults to
        'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param embeddings_initializer: Initializer for the token embedding
        table. Defaults to 'glorot_uniform'.
    :type embeddings_initializer: Union[str, keras.initializers.Initializer]
    :param final_norm_epsilon: Epsilon for the final :class:`RMSNorm`
        applied before the tied LM head. Defaults to 1e-6.
    :type final_norm_epsilon: float
    :param pretrained: If True, raises ``NotImplementedError`` -- no
        pretrained Zamba2 checkpoint is distributed with ``dl_techniques``.
        Must be False (the default) to construct a model. A local checkpoint
        is restored through :meth:`from_variant` (``pretrained=<path>``).
    :type pretrained: bool
    :param pad_token_id: Id treated as padding by the task head when the input
        carries no ``attention_mask``. Read only with ``include_top=True``.
        Defaults to None, which derives no padding (the head sees every
        position): id 0 is a real token, so with right-padded batches pass the
        tokenizer's pad id or an ``attention_mask``.
    :type pad_token_id: Optional[int]
    :param include_top: Whether the model owns a task head. Defaults to False:
        no head layer and no head weight exist, and ``call`` returns the tied
        logits exactly as before.
    :type include_top: bool
    :param task_config: The head's task, as an ``NLPTaskConfig`` or a dict of
        its fields. Required when ``include_top`` is True, refused otherwise;
        stored as a plain dict. ``MULTIPLE_CHOICE`` is refused. A
        generation-type head's ``vocabulary_size`` is checked against
        ``vocab_size``.
    :type task_config: Optional[Union[NLPTaskConfig, Dict[str, Any]]]
    :param head_config_overrides: Extra keyword arguments for the head
        constructor (e.g. ``{"pooling_type": "mean"}``, which replaces the
        ``'last'`` default). Only valid with ``include_top=True``.
    :type head_config_overrides: Optional[Dict[str, Any]]
    :param kwargs: Additional keyword arguments for ``keras.Model``.
    :type kwargs: Any

    :ivar layer_mapping: The stored, validated ``'m'``/``'g'`` token list.
    :vartype layer_mapping: List[str]
    :ivar embedding: The token embedding table; its single weight is also
        the tied LM head's projection matrix.
    :vartype embedding: keras.layers.Embedding
    :ivar mamba_blocks: One freshly-built :class:`Zamba2MambaBlock` per
        ``'m'`` entry in ``layer_mapping``, in depth order.
    :vartype mamba_blocks: List[Zamba2MambaBlock]
    :ivar mem_attention_blocks: ``num_mem_blocks`` physical
        :class:`Zamba2SharedAttentionBlock` instances, reused round-robin
        across every ``'g'`` position.
    :vartype mem_attention_blocks: List[Zamba2SharedAttentionBlock]
    :ivar mem_mlp_blocks: ``num_mem_blocks`` physical
        :class:`Zamba2SharedMLPBlock` instances, paired index-for-index with
        ``mem_attention_blocks``.
    :vartype mem_mlp_blocks: List[Zamba2SharedMLPBlock]
    :ivar final_norm: The pre-LM-head :class:`RMSNorm`.
    :vartype final_norm: RMSNorm
    :ivar task_head: The task head, or None without ``include_top``.
    :vartype task_head: Optional[keras.layers.Layer]

    :raises ValueError: If ``vocab_size``, ``hidden_size``, ``num_mem_blocks``,
        ``num_heads``, or ``max_seq_len`` is not positive; if
        ``num_heads`` does not divide ``hidden_size``; if ``layer_mapping``
        is empty or contains an entry other than ``'m'``/``'g'``; if
        ``include_top=True`` comes without a ``task_config``, or a
        ``task_config`` or ``head_config_overrides`` comes with
        ``include_top=False``.
    :raises NotImplementedError: If ``pretrained`` is True.

    Input shape:
        ``input_ids``: ``(batch_size, seq_len)``, integer token ids, or a
        dictionary holding them under ``'input_ids'`` (and optionally an
        ``'attention_mask'``, read only by a task head).

    Output shape:
        With ``include_top=False``, ``(batch_size, seq_len, vocab_size)``,
        next-token logits (a bare tensor, not a dictionary). With
        ``include_top=True``, the head's output: a bare tensor, or a dict of
        two or more tensors (question answering).

    Example:
        .. code-block:: python

            model = Zamba2Model(
                vocab_size=1000,
                hidden_size=256,
                layer_mapping=["m", "m", "g", "m", "m", "g"],
                num_mem_blocks=2,
                num_heads=8,
                max_seq_len=128,
            )
            input_ids = keras.random.randint((2, 32), 0, 1000, dtype="int32")
            logits = model(input_ids)
            print(logits.shape)
            # (2, 32, 1000)

    Note:
        Every sub-block is created in :meth:`__init__` (unconditionally,
        per the repo's "create unconditionally, use conditionally" rule)
        even though each depth position's :meth:`call` only ever exercises
        one of the ``'m'``/``'g'`` branches at that position. The
        occurrence counter that selects each ``'g'`` position's LoRA pair
        is a Python-level loop variable fixed at construction time (static
        per depth position, never a data-dependent tensor) -- it is NOT the
        same value as the physical mem-block slot index, which cycles
        round-robin through ``num_mem_blocks`` separately. Collapsing the
        two onto one counter would be exactly the failure this plan's
        Pre-Mortem names: "the stack reads only the last block."
    """

    _SEQUENCE_LENGTH_MESSAGE = (
        "seq_len={seq_len} exceeds max_seq_len={limit}. Zamba2's shared "
        "attention precomputes its RoPE tables for {limit} positions, so "
        "there is no rotation for position {limit}. Truncate the input to "
        "{limit} tokens, or build the model with max_seq_len>={seq_len}."
    )

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        layer_mapping: Sequence[str],
        num_mem_blocks: int,
        num_heads: int = 8,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        rope_percentage: float = 1.0,
        attention_dropout_rate: float = 0.0,
        mem_block_norm_epsilon: float = 1e-6,
        mlp_hidden_dim: Optional[int] = None,
        ffn_expansion_factor: int = 4,
        ffn_multiple_of: int = 256,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        d_ssm: Optional[int] = None,
        ngroups: int = 1,
        mamba_norm_epsilon: float = 1e-5,
        mamba_rmsnorm: bool = True,
        norm_before_gate: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        mamba_bias: bool = False,
        conv_bias: bool = True,
        use_bias: bool = False,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        embeddings_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        final_norm_epsilon: float = 1e-6,
        pretrained: bool = False,
        pad_token_id: Optional[int] = None,
        include_top: bool = False,
        task_config: Optional[Union[NLPTaskConfig, Dict[str, Any]]] = None,
        head_config_overrides: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Validate the configuration, resolve the ``layer_mapping`` dispatch
        table, and create every sub-block (unbuilt).

        :raises ValueError: If ``vocab_size``, ``hidden_size``,
            ``num_mem_blocks``, ``num_heads``, or ``max_seq_len`` is not
            positive; if ``num_heads`` does not divide ``hidden_size``; if
            ``layer_mapping`` is empty or contains an entry other than
            ``'m'``/``'g'``; or the three task-head knobs are inconsistent.
        :raises NotImplementedError: If ``pretrained`` is True.
        """
        super().__init__(**kwargs)

        if pretrained:
            raise NotImplementedError(
                "No pretrained Zamba2 checkpoint is distributed with "
                "dl_techniques. Construct with pretrained=False (the "
                "default) and warm-start a built model via "
                "model.load_weights('/path/to/weights.keras') instead."
            )

        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_mem_blocks <= 0:
            raise ValueError(f"num_mem_blocks must be positive, got {num_mem_blocks}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        validate_task_head_knobs(include_top, task_config, head_config_overrides)

        layer_mapping = list(layer_mapping)
        if not layer_mapping:
            raise ValueError("layer_mapping must be non-empty")
        invalid = [token for token in layer_mapping if token not in ("m", "g")]
        if invalid:
            raise ValueError(
                f"layer_mapping entries must be 'm' or 'g', got {invalid}"
            )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.layer_mapping = layer_mapping
        self.num_mem_blocks = num_mem_blocks
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.rope_percentage = rope_percentage
        self.attention_dropout_rate = attention_dropout_rate
        self.mem_block_norm_epsilon = mem_block_norm_epsilon
        self.mlp_hidden_dim = mlp_hidden_dim
        self.ffn_expansion_factor = ffn_expansion_factor
        self.ffn_multiple_of = ffn_multiple_of
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.d_ssm = d_ssm
        self.ngroups = ngroups
        self.mamba_norm_epsilon = mamba_norm_epsilon
        self.mamba_rmsnorm = mamba_rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        self.mamba_bias = mamba_bias
        self.conv_bias = conv_bias
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.embeddings_initializer = keras.initializers.get(embeddings_initializer)
        self.final_norm_epsilon = final_norm_epsilon
        self.pad_token_id = None if pad_token_id is None else int(pad_token_id)
        self.include_top = bool(include_top)
        self.task_config = (
            task_config_to_dict(task_config, vocab_size=vocab_size)
            if task_config is not None else None
        )
        self.head_config_overrides = (
            dict(head_config_overrides) if head_config_overrides is not None else None
        )

        # Number of independent LoRA occurrences every shared MLP mem-block
        # must allocate: one per 'g' position in layer_mapping. `max(..., 1)`
        # keeps LoRAAdapter's positive-occurrence-count validation satisfied
        # even for the zero-'g' edge case (plan.md Problem Statement edge
        # cases) -- the allocated pair is simply never selected.
        num_g_positions = layer_mapping.count("g")
        self._num_occurrences = max(num_g_positions, 1)

        self.embedding = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.hidden_size,
            embeddings_initializer=self.embeddings_initializer,
            name="embedding",
        )

        self.mamba_blocks: List[Zamba2MambaBlock] = []
        self.mem_attention_blocks: List[Zamba2SharedAttentionBlock] = []
        self.mem_mlp_blocks: List[Zamba2SharedMLPBlock] = []

        for slot in range(self.num_mem_blocks):
            self.mem_attention_blocks.append(
                Zamba2SharedAttentionBlock(
                    d_model=self.hidden_size,
                    num_heads=self.num_heads,
                    max_seq_len=self.max_seq_len,
                    rope_theta=self.rope_theta,
                    rope_percentage=self.rope_percentage,
                    attention_dropout_rate=self.attention_dropout_rate,
                    norm_epsilon=self.mem_block_norm_epsilon,
                    use_bias=self.use_bias,
                    kernel_initializer=self.kernel_initializer,
                    name=f"mem_attention_block_{slot}",
                )
            )
            self.mem_mlp_blocks.append(
                Zamba2SharedMLPBlock(
                    d_model=self.hidden_size,
                    num_occurrences=self._num_occurrences,
                    hidden_dim=self.mlp_hidden_dim,
                    ffn_expansion_factor=self.ffn_expansion_factor,
                    ffn_multiple_of=self.ffn_multiple_of,
                    lora_rank=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                    norm_epsilon=self.mem_block_norm_epsilon,
                    use_bias=self.use_bias,
                    kernel_initializer=self.kernel_initializer,
                    name=f"mem_mlp_block_{slot}",
                )
            )

        # DECISION plan-2026-09-12T075714-035fd488/D-005: every 'm' position
        # gets its OWN Zamba2MambaBlock instance -- never shared -- while
        # 'g' positions route to one of the `num_mem_blocks` shared pairs
        # built above, round-robin. Do not route 'm' positions through the
        # mem-block lists, and do not give two 'm' positions the same
        # Zamba2MambaBlock instance. See decisions.md D-005.
        #
        # `_position_info` is the per-depth dispatch table `call()` walks:
        # each entry is (kind, ref_idx, occurrence_idx) where `ref_idx`
        # indexes `mamba_blocks` for 'm' or `mem_attention_blocks`/
        # `mem_mlp_blocks` for 'g', and `occurrence_idx` (only meaningful
        # for 'g') is the GLOBAL occurrence counter -- independent of which
        # physical slot `ref_idx` names -- selecting the MLP block's LoRA
        # pair (findings.md Key Constraints: occurrences cycle round-robin
        # through the physical blocks, e.g. ABAB for num_mem_blocks=2).
        self._position_info: List[Tuple[str, int, Optional[int]]] = []
        occurrence_counter = 0
        for token in self.layer_mapping:
            if token == "m":
                ref_idx = len(self.mamba_blocks)
                self.mamba_blocks.append(
                    Zamba2MambaBlock(
                        d_model=self.hidden_size,
                        d_state=self.d_state,
                        d_conv=self.d_conv,
                        expand=self.expand,
                        headdim=self.headdim,
                        d_ssm=self.d_ssm,
                        ngroups=self.ngroups,
                        norm_epsilon=self.mamba_norm_epsilon,
                        rmsnorm=self.mamba_rmsnorm,
                        norm_before_gate=self.norm_before_gate,
                        dt_min=self.dt_min,
                        dt_max=self.dt_max,
                        dt_init_floor=self.dt_init_floor,
                        bias=self.mamba_bias,
                        conv_bias=self.conv_bias,
                        name=f"mamba_block_{ref_idx}",
                    )
                )
                self._position_info.append(("m", ref_idx, None))
            else:
                slot = occurrence_counter % self.num_mem_blocks
                self._position_info.append(("g", slot, occurrence_counter))
                occurrence_counter += 1

        self.final_norm = RMSNorm(epsilon=self.final_norm_epsilon, name="final_norm")

        # Created after the backbone and under its own fixed name, so the
        # backbone's layer names and weight paths are the same with or without
        # a head. Its random initial VALUES are not: the LoRA ``A`` slices draw
        # their seeds when they are built, after the head's initializers drew
        # theirs.
        self.task_head = None
        if self.include_top:
            # DECISION plan-2026-10-11T170958-dc4d2202/D-005: headless stays the tied-logits tensor
            # and the head reads the FINAL-NORM output; do NOT feed it the logits, flip the headless
            # output to a hidden-state dict (breaks every logits test and the skip_head trainer), or keep
            # the head's 'cls' default -- position 0 of a causal decoder has seen only itself. See decisions.md.
            self.task_head = build_model_task_head(
                self.task_config,
                input_dim=self.hidden_size,
                head_config_overrides=self.head_config_overrides,
                vocab_size=self.vocab_size,
                default_pooling="last",
            )

        logger.info(
            f"Initialized Zamba2Model with vocab_size={vocab_size}, "
            f"hidden_size={hidden_size}, depth={len(self.layer_mapping)}, "
            f"num_mem_blocks={num_mem_blocks}, num_occurrences={self._num_occurrences}, "
            f"num_mamba_blocks={len(self.mamba_blocks)}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the embedding, every sub-block, the final norm and the task head
        (when there is one) in order.

        :param input_shape: Shape of ``input_ids``, ``(batch_size, seq_len)``,
            or a dict holding it under ``'input_ids'``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        ids_shape = ids_shape_of(input_shape)
        # First, so the cached step functions are dropped and no variable exists.
        if "g" in self.layer_mapping:
            refuse_static_sequence_length(
                self, ids_shape[-1], self.max_seq_len,
                self._SEQUENCE_LENGTH_MESSAGE.format(
                    seq_len=ids_shape[-1], limit=self.max_seq_len
                ),
            )
        self.embedding.build(ids_shape)

        hidden_shape = ids_shape + (self.hidden_size,)
        for mamba_block in self.mamba_blocks:
            mamba_block.build(hidden_shape)
        for attention_block, mlp_block in zip(
            self.mem_attention_blocks, self.mem_mlp_blocks
        ):
            attention_block.build(hidden_shape)
            mlp_block.build(hidden_shape)
        self.final_norm.build(hidden_shape)
        if self.task_head is not None:
            self.task_head.build(
                {"hidden_states": hidden_shape, "attention_mask": ids_shape}
            )

        super().build(input_shape)

    def call(
        self,
        inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
        training: Optional[bool] = None,
    ) -> Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """
        Run the full decoder stack, then project to tied-embedding logits or
        pool the final hidden states through the task head.

        :param inputs: Token ids, shape ``(batch_size, seq_len)``, or a
            dictionary holding them under ``'input_ids'`` (and optionally an
            ``'attention_mask'``, read only by a task head).
        :type inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :param training: Whether in training mode, forwarded to every
            sub-block and the task head.
        :type training: Optional[bool]
        :return: With ``include_top=False``, next-token logits, shape
            ``(batch_size, seq_len, vocab_size)``. With ``include_top=True``,
            only the head's output reduced by ``finalize_head_outputs``.
        :rtype: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        :raises ValueError: If a dictionary input has no ``'input_ids'`` key.
        """
        attention_mask = None
        if isinstance(inputs, dict):
            if "input_ids" not in inputs:
                raise ValueError("Dictionary input must contain 'input_ids' key")
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")
        else:
            input_ids = inputs

        if "g" in self.layer_mapping:  # only the shared attention has RoPE tables
            input_ids = check_sequence_length(
                self, input_ids, self.max_seq_len, "max_seq_len",
                self._SEQUENCE_LENGTH_MESSAGE,
            )
        original_embedding = self.embedding(input_ids)
        hidden_states = original_embedding

        for kind, ref_idx, occurrence_idx in self._position_info:
            if kind == "m":
                hidden_states = self.mamba_blocks[ref_idx](
                    hidden_states, training=training
                )
            else:
                hidden_states = self.mem_attention_blocks[ref_idx](
                    hidden_states, original_embedding, training=training
                )
                hidden_states = self.mem_mlp_blocks[ref_idx](
                    hidden_states, occurrence_idx=occurrence_idx, training=training
                )

        hidden_states = self.final_norm(hidden_states, training=training)

        if self.include_top:
            head_outputs = self.task_head(
                {
                    "hidden_states": hidden_states,
                    "attention_mask": head_attention_mask(
                        input_ids, attention_mask, self.pad_token_id,
                        hidden_states.dtype,
                    ),
                },
                training=training,
            )
            return finalize_head_outputs(head_outputs)

        embedding_weights = self.embedding.weights[0]
        return tied_embedding_logits(hidden_states, embedding_weights)

    def compute_output_shape(self, input_shape: Any) -> Any:
        """
        Compute the shapes of what :meth:`call` returns.

        :param input_shape: Shape of ``input_ids``, ``(batch_size, seq_len)``,
            or a dict holding it under ``'input_ids'``.
        :type input_shape: Any
        :return: Headless, ``(batch_size, seq_len, vocab_size)``. With a head,
            the head's output shape after the reduction ``call`` applies: one
            shape, or a mapping of two or more.
        :rtype: Any
        """
        ids_shape = ids_shape_of(input_shape)
        if self.include_top:
            return head_output_shape(self.task_head, ids_shape, self.hidden_size)
        return (ids_shape[0], ids_shape[1], self.vocab_size)

    #: Re-exported, not redefined: the module-level :data:`MODEL_VARIANTS`,
    #: aliased onto the class so ``from_variant``/``create_zamba2`` and the
    #: API-contract tests can reach the table through the class, matching
    #: ``hnet``'s ``HNet.MODEL_VARIANTS`` convention. The module-level table
    #: is a ``MappingProxyType`` (not a plain dict), which is the real
    #: remedy this alias relies on -- see D-008, and contrast the stale
    #: claim this comment used to make (the review's concern 12: "matching
    #: hnet's convention" is not itself what keeps the alias off the
    #: mutable-default guard; the proxy is).
    MODEL_VARIANTS: Mapping[str, Mapping[str, Any]] = MODEL_VARIANTS

    @classmethod
    def from_variant(
        cls,
        variant: str,
        pretrained: Union[bool, str] = False,
        **overrides: Any,
    ) -> "Zamba2Model":
        """
        Build one of the shipped :data:`MODEL_VARIANTS` (``'zamba2_mini'``,
        ``'zamba2_small'``, ``'zamba2_base'``).

        ``layer_mapping`` is derived from the variant's ``num_mamba_blocks``/
        ``num_mem_block_occurrences`` entries via :func:`_build_layer_mapping`
        rather than stored literally in the table, so the counts can never
        drift out of sync with the list's actual composition. Pass
        ``layer_mapping=...`` in ``overrides`` to bypass this derivation and
        supply an explicit mapping instead.

        ``num_mem_block_occurrences`` (the number of ``'g'`` positions to
        place) is a table-only key, popped here and never forwarded to
        :meth:`__init__` -- the constructor's own ``num_mem_blocks`` is the
        number of PHYSICAL shared blocks, a different, smaller number (see
        the module-level :data:`MODEL_VARIANTS` docstring comment and
        decisions.md D-009). Passing ``num_mem_blocks`` itself as the
        occurrence count, as an earlier revision of this method did, makes
        every physical block invoked exactly once and defeats round-robin
        reuse entirely.

        :param variant: A key of :data:`MODEL_VARIANTS`.
        :type variant: str
        :param pretrained: If a string, the path of a local ``.keras``
            checkpoint: the backbone (the embedding and every block, by layer
            name) is restored from it, headed or headless; a ``task_head`` in
            the checkpoint is never copied and the model's own head keeps its
            fresh initialization, so a checkpoint of any Zamba2, headless or
            trained under any head, warm-starts a model with any other head or
            none. The backbone loads completely or the call raises: a
            checkpoint of another depth or width is refused, never truncated;
            on a vocabulary change only the token table keeps its init. To
            restore a whole model, head included, use
            ``keras.models.load_model``. If True, raises
            ``NotImplementedError`` -- no Zamba2 checkpoint is distributed
            with ``dl_techniques``. Defaults to False (random init).
        :type pretrained: Union[bool, str]
        :param overrides: Override or extend any variant parameter, including
            ``include_top``, ``task_config``, ``head_config_overrides`` and
            ``pad_token_id``.
        :type overrides: Any
        :return: The constructed model.
        :rtype: Zamba2Model
        :raises ValueError: If ``variant`` is not a recognized key, or a
            ``pretrained`` checkpoint does not fit the backbone.
        :raises NotImplementedError: If ``pretrained`` is True.
        :raises FileNotFoundError: If a ``pretrained`` path does not exist.

        Example:
            .. code-block:: python

                model = Zamba2Model.from_variant("zamba2_mini")
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant {variant!r}. "
                f"Available: {list(cls.MODEL_VARIANTS.keys())}"
            )

        config = dict(cls.MODEL_VARIANTS[variant])
        config.pop("description", None)
        # DECISION plan-2026-09-12T075714-035fd488/D-009
        # WHAT NOT TO DO: do not pass `num_mem_blocks` (the PHYSICAL block
        # count) to `_build_layer_mapping` as the occurrence count again.
        # That was the original defect -- it made `count('g') ==
        # num_mem_blocks` for every shipped variant, so each physical block
        # was invoked exactly once and the round-robin reuse this package
        # exists to implement was never exercised by any public factory
        # call (review-iter-1.md concern 2). `num_mem_block_occurrences` is
        # the table-only key carrying the occurrence count; it is popped
        # here and never forwarded to `cls(**config)`. See decisions.md D-009.
        num_mem_block_occurrences = config.pop("num_mem_block_occurrences")

        if "layer_mapping" not in overrides:
            num_mamba_blocks = config.pop("num_mamba_blocks")
            config["layer_mapping"] = _build_layer_mapping(
                num_mamba_blocks, num_mem_block_occurrences
            )
        else:
            config.pop("num_mamba_blocks", None)

        config.update(overrides)
        weights_path = pretrained_path_or_raise(pretrained, "Zamba2")
        model = cls(**config)
        if weights_path is not None:
            # A build, not a forward: no RNG is drawn and no weight is touched.
            if not model.built:
                model.build((None, None))
            # DECISION plan-2026-10-11T170958-dc4d2202/D-005: by name and strict through the shared
            # rule; do NOT use model.load_weights (restores nothing on a mismatch and reports success)
            # nor drop TASK_HEAD_NAME (a head of equal shape would be copied). See decisions.md.
            load_backbone_from_checkpoint(
                model,
                weights_path,
                head_prefixes=(TASK_HEAD_NAME,),
                vocab_layers=("embedding",),
                tables=("embedding/embeddings",),
            )
        return model

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration for serialization.

        :return: Dictionary containing every constructor argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "layer_mapping": list(self.layer_mapping),
                "num_mem_blocks": self.num_mem_blocks,
                "num_heads": self.num_heads,
                "max_seq_len": self.max_seq_len,
                "rope_theta": self.rope_theta,
                "rope_percentage": self.rope_percentage,
                "attention_dropout_rate": self.attention_dropout_rate,
                "mem_block_norm_epsilon": self.mem_block_norm_epsilon,
                "mlp_hidden_dim": self.mlp_hidden_dim,
                "ffn_expansion_factor": self.ffn_expansion_factor,
                "ffn_multiple_of": self.ffn_multiple_of,
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "d_state": self.d_state,
                "d_conv": self.d_conv,
                "expand": self.expand,
                "headdim": self.headdim,
                "d_ssm": self.d_ssm,
                "ngroups": self.ngroups,
                "mamba_norm_epsilon": self.mamba_norm_epsilon,
                "mamba_rmsnorm": self.mamba_rmsnorm,
                "norm_before_gate": self.norm_before_gate,
                "dt_min": self.dt_min,
                "dt_max": self.dt_max,
                "dt_init_floor": self.dt_init_floor,
                "mamba_bias": self.mamba_bias,
                "conv_bias": self.conv_bias,
                "use_bias": self.use_bias,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "embeddings_initializer": keras.initializers.serialize(
                    self.embeddings_initializer
                ),
                "final_norm_epsilon": self.final_norm_epsilon,
                "pad_token_id": self.pad_token_id,
                "include_top": self.include_top,
                "task_config": (
                    dict(self.task_config) if self.task_config is not None else None
                ),
                "head_config_overrides": (
                    dict(self.head_config_overrides)
                    if self.head_config_overrides is not None else None
                ),
            }
        )
        return config


# ---------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------


def create_zamba2(
    variant: str = "zamba2_small",
    pretrained: Union[bool, str] = False,
    **overrides: Any,
) -> Zamba2Model:
    """
    Create a Zamba2 model from a shipped :data:`MODEL_VARIANTS` name.

    Delegates to :meth:`Zamba2Model.from_variant` and holds no defaulting,
    validation, or construction logic of its own (matching ``create_hnet``'s
    pure-delegation shape).

    :param variant: A key of :data:`MODEL_VARIANTS` --  ``'zamba2_mini'``,
        ``'zamba2_small'`` (default), or ``'zamba2_base'``.
    :type variant: str
    :param pretrained: If a string, loads the backbone from that local
        ``.keras`` checkpoint (a model with a head receives the backbone only;
        see :meth:`Zamba2Model.from_variant`). If ``True``, raises
        ``NotImplementedError`` -- no pretrained Zamba2 checkpoint is
        distributed with ``dl_techniques`` (decisions.md D-004). Defaults to
        ``False``.
    :type pretrained: Union[bool, str]
    :param overrides: Forwarded to :meth:`Zamba2Model.from_variant`,
        overriding any variant parameter, including ``include_top=True`` with
        a ``task_config`` for a model with a task head.
    :type overrides: Any
    :return: The constructed model.
    :rtype: Zamba2Model
    :raises ValueError: If ``variant`` is not a recognized key, or a forwarded
        argument is invalid.
    :raises NotImplementedError: If ``pretrained`` is True.

    Example:
        .. code-block:: python

            model = create_zamba2("zamba2_mini")
            input_ids = keras.random.randint((2, 32), 0, 100277, dtype="int32")
            logits = model(input_ids)
    """
    return Zamba2Model.from_variant(variant, pretrained=pretrained, **overrides)
