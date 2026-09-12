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
#: derived below from ``num_mamba_blocks``/``num_mem_blocks`` so the two
#: counts can never drift apart).
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
        "num_heads": 4,
        "max_seq_len": 512,
        "lora_rank": 4,
        "lora_alpha": 8.0,
        "description": "Zamba2 mini: ~256-dim, 8 Mamba2 blocks, 2 mem-block "
                        "occurrences -- fast tests/smoke runs.",
    },
    "zamba2_small": {
        "vocab_size": DEFAULT_VOCAB_SIZE,
        "hidden_size": 512,
        "num_mamba_blocks": 18,
        "num_mem_blocks": 6,
        "num_heads": 8,
        "max_seq_len": 1024,
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "description": "Zamba2 small: ~512-dim, 18 Mamba2 blocks, 6 "
                        "mem-block occurrences.",
    },
    "zamba2_base": {
        "vocab_size": DEFAULT_VOCAB_SIZE,
        "hidden_size": 768,
        "num_mamba_blocks": 24,
        "num_mem_blocks": 8,
        "num_heads": 12,
        "max_seq_len": 2048,
        "lora_rank": 16,
        "lora_alpha": 32.0,
        "description": "Zamba2 base: ~768-dim, 24 Mamba2 blocks, 8 "
                        "mem-block occurrences.",
    },
    }
)

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique
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
    output ``Dense`` layer.

    Architecture:

    .. code-block:: text

        input_ids [B, S]
               │
               ▼ Embedding(vocab_size, hidden_size)
        original_embedding [B, S, H]  ──────────────────────┐
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
          │   Zamba2SharedMLPBlock_{i%N}(         │   round-robin; occurrence
          │     hidden_state,                     │   counter increments once
          │     occurrence_idx=<g counter>)       │   per 'g', independent of
          └────┬─────────────────────────────────┘   which physical slot)
               ▼
        final RMSNorm
               │
               ▼
        logits = hidden_state @ embedding.weights[0]^T   (tied LM head)
        [B, S, vocab_size]

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
        Must be False (the default) to construct a model; warm-start a
        built model via ``model.load_weights(path)`` instead.
    :type pretrained: bool
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

    :raises ValueError: If ``vocab_size``, ``hidden_size``, ``num_mem_blocks``,
        ``num_heads``, or ``max_seq_len`` is not positive; if
        ``num_heads`` does not divide ``hidden_size``; if ``layer_mapping``
        is empty or contains an entry other than ``'m'``/``'g'``.
    :raises NotImplementedError: If ``pretrained`` is True.

    Input shape:
        ``input_ids``: ``(batch_size, seq_len)``, integer token ids.

    Output shape:
        ``(batch_size, seq_len, vocab_size)``, next-token logits.

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
        **kwargs: Any,
    ) -> None:
        """Validate the configuration, resolve the ``layer_mapping`` dispatch
        table, and create every sub-block (unbuilt).

        :raises ValueError: If ``vocab_size``, ``hidden_size``,
            ``num_mem_blocks``, ``num_heads``, or ``max_seq_len`` is not
            positive; if ``num_heads`` does not divide ``hidden_size``; if
            ``layer_mapping`` is empty or contains an entry other than
            ``'m'``/``'g'``.
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

        logger.info(
            f"Initialized Zamba2Model with vocab_size={vocab_size}, "
            f"hidden_size={hidden_size}, depth={len(self.layer_mapping)}, "
            f"num_mem_blocks={num_mem_blocks}, num_occurrences={self._num_occurrences}, "
            f"num_mamba_blocks={len(self.mamba_blocks)}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the embedding, every sub-block, and the final norm in order.

        :param input_shape: Shape of ``input_ids``, ``(batch_size, seq_len)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        ids_shape = tuple(input_shape)
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

        super().build(input_shape)

    def call(
        self,
        input_ids: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """
        Run the full decoder stack and project to tied-embedding logits.

        :param input_ids: Token ids, shape ``(batch_size, seq_len)``.
        :type input_ids: keras.KerasTensor
        :param training: Whether in training mode, forwarded to every
            sub-block.
        :type training: Optional[bool]
        :return: Next-token logits, shape ``(batch_size, seq_len, vocab_size)``.
        :rtype: keras.KerasTensor
        """
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

        embedding_weights = self.embedding.weights[0]
        logits = keras.ops.matmul(
            hidden_states, keras.ops.transpose(embedding_weights)
        )
        return logits

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the model.

        :param input_shape: Shape of ``input_ids``, ``(batch_size, seq_len)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``(batch_size, seq_len, vocab_size)``.
        :rtype: Tuple[Optional[int], ...]
        """
        batch_size, seq_len = input_shape[0], input_shape[1]
        return (batch_size, seq_len, self.vocab_size)

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
        **overrides: Any,
    ) -> "Zamba2Model":
        """
        Build one of the shipped :data:`MODEL_VARIANTS` (``'zamba2_mini'``,
        ``'zamba2_small'``, ``'zamba2_base'``).

        ``layer_mapping`` is derived from the variant's ``num_mamba_blocks``/
        ``num_mem_blocks`` entries via :func:`_build_layer_mapping` rather
        than stored literally in the table, so the two counts can never
        drift out of sync with the list's actual composition. Pass
        ``layer_mapping=...`` in ``overrides`` to bypass this derivation and
        supply an explicit mapping instead.

        :param variant: A key of :data:`MODEL_VARIANTS`.
        :type variant: str
        :param overrides: Override or extend any variant parameter,
            including ``pretrained`` (which :meth:`__init__` raises
            ``NotImplementedError`` for when True -- no Zamba2 checkpoint is
            distributed with ``dl_techniques``).
        :type overrides: Any
        :return: The constructed model.
        :rtype: Zamba2Model
        :raises ValueError: If ``variant`` is not a recognized key.
        :raises NotImplementedError: If ``pretrained=True`` is passed in
            ``overrides``.

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

        if "layer_mapping" not in overrides:
            num_mamba_blocks = config.pop("num_mamba_blocks")
            num_mem_blocks = config["num_mem_blocks"]
            config["layer_mapping"] = _build_layer_mapping(
                num_mamba_blocks, num_mem_blocks
            )
        else:
            config.pop("num_mamba_blocks", None)

        config.update(overrides)
        return cls(**config)

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
            }
        )
        return config


# ---------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------


def create_zamba2(
    variant: str = "zamba2_small",
    pretrained: bool = False,
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
    :param pretrained: If ``True``, raises ``NotImplementedError`` -- no
        pretrained Zamba2 checkpoint is distributed with ``dl_techniques``
        (decisions.md D-004). Must be ``False`` (the default).
    :type pretrained: bool
    :param overrides: Forwarded to :meth:`Zamba2Model.from_variant`,
        overriding any variant parameter.
    :type overrides: Any
    :return: The constructed model.
    :rtype: Zamba2Model
    :raises ValueError: If ``variant`` is not a recognized key.
    :raises NotImplementedError: If ``pretrained`` is True.

    Example:
        .. code-block:: python

            model = create_zamba2("zamba2_mini")
            input_ids = keras.random.randint((2, 32), 0, 100277, dtype="int32")
            logits = model(input_ids)
    """
    return Zamba2Model.from_variant(variant, pretrained=pretrained, **overrides)
