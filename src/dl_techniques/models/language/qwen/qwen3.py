"""
Qwen3 decoder-only language model with grouped-query attention, RoPE, and
mixture-of-experts feed-forward layers at selectable depths.

A dense transformer's parameter count and its per-token FLOP cost are the
same number, so raising quality always raises serving cost. Sparse
mixture-of-experts breaks that tie: a block's FFN becomes `num_experts`
independent FFNs plus a router that selects `num_experts_per_tok` of them
per token, so parameters scale with `num_experts` while arithmetic scales
with `num_experts_per_tok`. `moe_layers` names which block indices get this
treatment — the variants interleave dense and sparse blocks rather than
making every layer sparse, so a stable dense pathway exists at every depth.
Each expert's FFN is `moe_intermediate_size` wide, as in the Hugging Face
Qwen3-MoE config. Before 2026-10-13 it was a SwiGLU sized by
`max(1, moe_intermediate_size // hidden_size)` through the 2/3 rule, so an
MoE archive saved earlier no longer fits and `from_variant(pretrained=...)`
refuses it; dense layers are unchanged.

Attention is grouped-query with RoPE: GQA shares each key/value head across
several query heads, shrinking the KV cache by that ratio, which is what
lets the 30B-coder variant quote a 262144-token context. `rope_theta` scales
with context length across variants (10,000 at 4096 tokens up to
10,000,000 at 262144) so the slowest rotation's wavelength stays ahead of
the context — otherwise distant positions become numerically
indistinguishable from near ones.

Causality is imposed once at the model level: `call` builds a combined
causal-plus-padding mask via `create_causal_attend_mask` and passes the
same tensor to every block, since neither `TransformerLayer` nor the
attention layers it wraps manufacture one on their own. Blocks are pre-norm
RMSNorm with no biases; the LM head is an independent bias-free `Dense`, so
input and output embeddings are untied. With `return_dict=False` (the
default) the model returns logits.

With `include_top=True` and a `task_config` the model owns ONE task head from
`layers/heads/nlp`, named `task_head`, in place of the LM head (no `lm_head`
is created and no vocabulary logits are computed), fed the final-normed
hidden states and pooling `'last'` by default, as `models/language/gpt2`
does. `create_qwen3` returns this class itself, headed or headless, and
takes the head only through `include_top` and `task_config`; there is no
per-task factory and no functional wrapper. Before this, the classifier pooled
the vocab_size-wide logits (and since c6f5a13f2 could not even be built), so no
classifier checkpoint written earlier is portable. `from_variant(...,
pretrained=<path>)` (and `create_qwen3`, which forwards to it) restores the
backbone of a local `.keras` checkpoint, including an old two-input wrapper
archive, and `pretrained=True` raises `NotImplementedError` (no weights are
distributed with this package).

References:
    - Qwen Team, 2025. Qwen3 Technical Report.
    - Ainslie et al., 2023. GQA: Training Generalized Multi-Query Transformer Models
      from Multi-Head Checkpoints. (https://arxiv.org/abs/2305.13245)
    - Su et al., 2021. RoFormer: Enhanced Transformer with Rotary Position Embedding.
      (https://arxiv.org/abs/2104.09864)
    - Shazeer et al., 2017. Outrageously Large Neural Networks: The Sparsely-Gated
      Mixture-of-Experts Layer. (https://arxiv.org/abs/1701.06538)
    - Fedus et al., 2021. Switch Transformers: Scaling to Trillion Parameter Models
      with Simple and Efficient Sparsity. (https://arxiv.org/abs/2101.03961)
    - Shazeer, 2020. GLU Variants Improve Transformer.
      (https://arxiv.org/abs/2002.05202)
    - Zhang and Sennrich, 2019. Root Mean Square Layer Normalization.
      (https://arxiv.org/abs/1910.07467)
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
"""

import keras
from typing import Optional, Union, Any, Dict, List

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.utils.masking import create_causal_attend_mask
from dl_techniques.utils.model_build import concretize_axes, materialize_sublayers
from dl_techniques.utils.sequence_length_guard import (
    RecordedBuildShapeMixin,
    check_sequence_length,
)
from dl_techniques.utils.weight_transfer import (
    load_backbone_from_checkpoint,
    pretrained_path_or_raise,
)
from dl_techniques.layers.transformers import TransformerLayer
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.heads.nlp import (
    NLPTaskConfig,
    TASK_HEAD_NAME,
    build_model_task_head,
    finalize_head_outputs,
    head_attention_mask,
    head_output_shape,
    ids_shape_of,
    task_config_to_dict,
    validate_task_head_knobs,
)
from dl_techniques.layers.moe import MoEConfig, ExpertConfig, GatingConfig
from dl_techniques.layers.ffn import assemble_ffn_config

from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.qwen.qwen3")
class Qwen3(RecordedBuildShapeMixin, keras.Model):
    """
    Qwen3 model with standard transformer architecture and optional MoE layers.

    Uses standard TransformerLayer blocks rather than the custom
    Qwen3NextBlock. Some layers can use Mixture of Experts while others use
    a standard dense FFN.

    Architecture:

    .. code-block:: text

        Input (input_ids)
               │
               ▼
        Token Embeddings [vocab_size, hidden_size]
               │
               ▼
        TransformerLayer x num_layers:
            GroupedQueryAttention (RoPE) → RMSNorm → SwiGLU/MoE → Residual
               │
               ▼
        Final RMSNorm
               │
               ▼
        Linear Projection → Logits [vocab_size]

        include_top=True: the projection is replaced by
        task_head (pools 'last') → finalized head output (only output)

    :param vocab_size: Size of the vocabulary. Defaults to 151936.
    :param hidden_size: Dimensionality of encoder layers. Defaults to 2048.
    :param num_layers: Number of transformer blocks. Defaults to 12.
    :param num_attention_heads: Number of attention heads. Defaults to 16.
    :param num_key_value_heads: Number of key-value heads for GQA. Defaults to 4.
    :param max_seq_len: Maximum sequence length. Defaults to 8192.
    :param moe_layers: Layer indices that use MoE. Defaults to empty list.
    :param num_experts: Total number of experts in MoE layers. Defaults to 64.
    :param num_experts_per_tok: Number of experts activated per token. Defaults to 8.
    :param moe_intermediate_size: Hidden (intermediate) width of each expert's
        FFN, used verbatim as the FFN's `hidden_dim`, as in the Hugging Face
        Qwen3-MoE config. Read only by the `moe_layers`: a dense layer's FFN
        is sized from `4 * hidden_size` (for SwiGLU, its 2/3 rule rounded up
        to a multiple of 256: 1536 wide at `hidden_size=512`), whatever this
        value is. Defaults to 1408.
    :param norm_eps: Epsilon for normalization layers. Defaults to 1e-6.
        Applies to `final_norm` and to every one of the `2 * num_layers`
        in-block norms (decisions.md D-007). Weight shapes are unchanged, so
        existing `.keras` files still load.
    :param dropout_rate: Dropout rate for regularization. Defaults to 0.0.
    :param initializer_range: Standard deviation for weight initialization. Defaults to 0.02.
    :param normalization_type: Type of normalization layer. Defaults to "rms_norm".
    :param ffn_type: Type of feed-forward network, dense and in experts. An
        expert gets `hidden_dim=moe_intermediate_size`, so in MoE layers it
        must be a type with a `hidden_dim` argument. Defaults to "swiglu".
    :param use_stochastic_depth: Whether to enable stochastic depth. Defaults to False.
    :param stochastic_depth_rate: Drop path rate for stochastic depth. Defaults to 0.1.
    :param pad_token_id: Id treated as padding by the task head when the input
        carries no `attention_mask`. Read only with `include_top=True`; the
        blocks never derive a mask from it. Defaults to None, which derives no
        padding (the head sees every position): id 0 is a real token, so with
        right-padded batches pass the tokenizer's pad id or an `attention_mask`.
        An explicit int is stored and serialized as given.
    :param include_top: Whether the model owns a task head (and no LM head).
        Defaults to False: no head layer and no head weight exist.
    :param task_config: The head's task, as an `NLPTaskConfig` or a dict of its
        fields. Required when `include_top` is True, refused otherwise; stored
        as a plain dict. `MULTIPLE_CHOICE` is refused.
    :param head_config_overrides: Extra keyword arguments for the head
        constructor (e.g. `{"pooling_type": "mean"}`, which replaces the
        `'last'` default). Only valid with `include_top=True`.
    :param kwargs: Additional keyword arguments for the `keras.Model` base class.
    :raises ValueError: On an invalid size, if `include_top=True` comes without
        a `task_config`, or if a `task_config` or `head_config_overrides` comes
        with `include_top=False`.
    """

    # Model variant configurations following Qwen3 specifications
    MODEL_VARIANTS = {
        "30b-coder": {
            "vocab_size": 151936,
            "hidden_size": 2048,
            "num_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "max_seq_len": 262144,
            "moe_layers": list(range(0, 48, 3)),  # Every 3rd layer uses MoE
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 768,
            "rope_theta": 10_000_000.0,
            "description": "Qwen3 Coder 30B-A3B: 48L transformer with MoE every 3rd layer"
        },
        "medium": {
            "vocab_size": 100000,
            "hidden_size": 1024,
            "num_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "max_seq_len": 16384,
            "moe_layers": list(range(6, 24, 4)),  # MoE starting from layer 6
            "num_experts": 16,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 2048,
            "rope_theta": 100_000.0,
            "description": "Qwen3 Medium: 24L transformer with selective MoE layers"
        },
        "small": {
            "vocab_size": 32000,
            "hidden_size": 768,
            "num_layers": 12,
            "num_attention_heads": 12,
            "num_key_value_heads": 4,
            "max_seq_len": 4096,
            "moe_layers": [3, 6, 9],
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 1024,
            "rope_theta": 10_000.0,
            "description": "Qwen3 Small: 12L transformer for experimentation"
        },
        "tiny": {
            "vocab_size": 32000,
            "hidden_size": 512,
            "num_layers": 6,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "max_seq_len": 2048,
            "moe_layers": [],  # No MoE layers
            "num_experts": 1,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 2048,
            "rope_theta": 10_000.0,
            "description": "Qwen3 Tiny: 6L dense transformer for mobile/edge deployment"
        },
    }

    _SEQUENCE_LENGTH_MESSAGE = (
        "seq_len={seq_len} exceeds max_seq_len={limit}. Qwen3 "
        "precomputes its RoPE tables for {limit} positions, so there is no "
        "rotation for position {limit}. Truncate the input to {limit} "
        "tokens, or build the model with max_seq_len>={seq_len}."
    )

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        num_layers: int = 12,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        max_seq_len: int = 8192,
        moe_layers: Optional[List[int]] = None,
        num_experts: int = 64,
        num_experts_per_tok: int = 8,
        moe_intermediate_size: int = 1408,
        rope_theta: float = 10_000_000.0,
        norm_eps: float = 1e-6,
        dropout_rate: float = 0.0,
        initializer_range: float = 0.02,
        normalization_type: str = "rms_norm",
        ffn_type: str = "swiglu",
        use_stochastic_depth: bool = False,
        stochastic_depth_rate: float = 0.1,
        pad_token_id: Optional[int] = None,
        include_top: bool = False,
        task_config: Optional[Union[NLPTaskConfig, Dict[str, Any]]] = None,
        head_config_overrides: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        # CRITICAL: Call super() FIRST for Keras models
        super().__init__(**kwargs)

        # Set defaults
        if moe_layers is None:
            moe_layers = []

        # Validate configuration parameters
        self._validate_config(
            vocab_size, hidden_size, num_layers, num_attention_heads,
            num_key_value_heads, num_experts, num_experts_per_tok, moe_layers
        )
        validate_task_head_knobs(include_top, task_config, head_config_overrides)

        # Store all configuration parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_seq_len = max_seq_len
        self.moe_layers = moe_layers
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.rope_theta = rope_theta
        self.norm_eps = norm_eps
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.normalization_type = normalization_type
        self.ffn_type = ffn_type
        self.use_stochastic_depth = use_stochastic_depth
        self.stochastic_depth_rate = stochastic_depth_rate
        self.pad_token_id = pad_token_id
        self.include_top = include_top
        self.task_config = (
            task_config_to_dict(task_config, vocab_size=vocab_size)
            if task_config is not None else None
        )
        self.head_config_overrides = (
            dict(head_config_overrides) if head_config_overrides is not None else None
        )

        # Calculate head dimension
        self.head_dim = self.hidden_size // self.num_attention_heads

        # Build the model architecture
        self._build_architecture()

        # Created after the backbone and under its own fixed name, so the
        # backbone's layer names, weight paths and RNG draws are the same with
        # or without a head.
        self.task_head = None
        if self.include_top:
            # DECISION plan-2026-10-11T004932-b83935e1/D-012: pool 'last' by default; do NOT keep the head's
            # 'cls' default -- position 0 of a causal decoder has seen only itself. See decisions.md.
            self.task_head = build_model_task_head(
                self.task_config,
                input_dim=self.hidden_size,
                head_config_overrides=self.head_config_overrides,
                vocab_size=self.vocab_size,
                default_pooling="last",
            )

        # Log model creation
        moe_info = f"{len(self.moe_layers)} MoE layers" if self.moe_layers else "dense model"
        active_params_pct = (self.num_experts_per_tok / self.num_experts) * 100 if self.num_experts > 1 else 100.0
        logger.info(
            f"Created Qwen3 model: {self.num_layers} layers, "
            f"hidden_size={self.hidden_size}, {moe_info}"
        )
        if self.moe_layers:
            logger.info(
                f"MoE configuration: experts={self.num_experts}, "
                f"active={self.num_experts_per_tok} ({active_params_pct:.1f}%)"
            )

    def _validate_config(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_layers: List[int],
    ) -> None:
        """Validate model configuration parameters."""
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive, got {num_attention_heads}")
        if num_key_value_heads <= 0:
            raise ValueError(f"num_key_value_heads must be positive, got {num_key_value_heads}")
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({num_key_value_heads})"
            )
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        if num_experts_per_tok <= 0:
            raise ValueError(f"num_experts_per_tok must be positive, got {num_experts_per_tok}")
        if num_experts_per_tok > num_experts:
            raise ValueError(
                f"num_experts_per_tok ({num_experts_per_tok}) cannot exceed "
                f"num_experts ({num_experts})"
            )
        if any(layer_idx < 0 or layer_idx >= num_layers for layer_idx in moe_layers):
            raise ValueError(f"All MoE layer indices must be between 0 and {num_layers-1}")

    def _build_architecture(self) -> None:
        """Build all model components following modern Keras 3 patterns."""

        # Token embedding layer
        self.embeddings = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.hidden_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name="token_embedding"
        )

        # Create MoE configuration
        moe_config = None
        if self.moe_layers:
            moe_config = MoEConfig(
                num_experts=self.num_experts,
                expert_config=ExpertConfig(
                    # DECISION plan-2026-07-30T140922-8af1028f/D-037: keep the
                    # assemble_ffn_config wrapper — an unfiltered literal fails
                    # for the 15 of 21 FFN types that reject ffn_expansion_factor.
                    ffn_config=assemble_ffn_config(self.ffn_type, {
                        "type": self.ffn_type,
                        "output_dim": self.hidden_size,
                        # DECISION plan-2026-10-13T035756-5dd4d94d/D-014: the explicit
                        # width, never ffn_expansion_factor = moe_intermediate_size //
                        # hidden_size (SwiGLU's 2/3 rule and 256 rounding ignore the
                        # configured value). See decisions.md.
                        "hidden_dim": self.moe_intermediate_size,
                    })
                ),
                gating_config=GatingConfig(
                    top_k=self.num_experts_per_tok,
                    gating_type="linear"
                )
            )

        # Create a linear schedule for the drop path rate
        dpr = linear_drop_path_rates(self.num_layers, self.stochastic_depth_rate)

        # Create transformer blocks
        self.blocks = []
        for i in range(self.num_layers):
            is_moe_layer = i in self.moe_layers

            attention_args = {
                'dim': self.hidden_size,
                'num_heads': self.num_attention_heads,
                'num_kv_heads': self.num_key_value_heads,
                'max_seq_len': self.max_seq_len,
                'dropout_rate': self.dropout_rate,
                'rope_theta': self.rope_theta
            }

            # An MoE layer gets moe_config only: TransformerLayer never reads
            # ffn_type/ffn_args beside it, and warns when they are passed.
            if is_moe_layer:
                ffn_kwargs = {'moe_config': moe_config}
            else:
                # DECISION plan-2026-07-30T140922-8af1028f/D-037: pre-filter through
                # assemble_ffn_config — a bare dict literal fails Qwen3(ffn_type=...)
                # for 11 of 12 previously-working types. See decisions.md.
                ffn_args = assemble_ffn_config(self.ffn_type, {
                    'output_dim': self.hidden_size,
                    'ffn_expansion_factor': 4,  # Standard 4x expansion
                    'dropout_rate': self.dropout_rate,
                    'use_bias': False
                })
                # Keyword form, not a dict literal: the FFN kwarg sweep in
                # tests/test_layers/test_ffn/test_factory.py finds this site by
                # its `ffn_args=` keyword.
                ffn_kwargs = dict(ffn_type=self.ffn_type, ffn_args=ffn_args)

            # DECISION plan-2026-08-19T070627-a616f581/D-007: norm_eps must reach
            # the block norms via attention_norm_args/ffn_norm_args, not just
            # final_norm — default norm_eps equals the factory default (1e-6),
            # which hid this for any test at default. See decisions.md.
            block = TransformerLayer(
                hidden_size=self.hidden_size,
                num_heads=self.num_attention_heads,
                intermediate_size=self.hidden_size * 4,
                attention_type='group_query',
                attention_args=attention_args,
                normalization_type=self.normalization_type,
                attention_norm_args={'epsilon': self.norm_eps},
                ffn_norm_args={'epsilon': self.norm_eps},
                normalization_position='pre',
                **ffn_kwargs,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.dropout_rate,
                use_stochastic_depth=self.use_stochastic_depth,
                stochastic_depth_rate=dpr[i],
                use_bias=False,
                name=f"transformer_block_{i}"
            )
            self.blocks.append(block)

        # Final normalization layer
        self.final_norm = create_normalization_layer(
            self.normalization_type,
            epsilon=self.norm_eps,
            name='final_norm'
        )

        # Language modeling head. A task head replaces it: call() never runs
        # lm_head then, and build() must not create what call() does not run.
        self.lm_head = None
        if not self.include_top:
            self.lm_head = keras.layers.Dense(
                units=self.vocab_size,
                use_bias=False,
                kernel_initializer=keras.initializers.TruncatedNormal(
                    stddev=self.initializer_range
                ),
                name='lm_head'
            )

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer from an explicit `build` call.

        Without this method an explicit `build((None, L))` inherits
        `Layer.build`, which marks the model built with 0 weights; the shared
        helper traces `call()` on symbolic inputs, so what gets built cannot
        drift from what gets called (the task head included).

        :param input_shape: Shape, or nest of shapes, of `call`'s `inputs`.
        """
        if self.built:
            return
        # The sequence axis is traced as a concrete 1: the causal mask cannot
        # be built on a None length, and no weight shape depends on it (the
        # same coercion, and reason, as Qwen3Next.build).
        materialize_sublayers(self, concretize_axes(input_shape, {1: 1}))
        super().build(input_shape)

    def compute_output_shape(self, input_shape: Any) -> Any:
        """Compute the shape of what ``call`` returns.

        Keras builds a symbolic call of this model from this method, never
        from ``call``, so it must report what ``call`` returns in both modes.
        It describes the default ``return_dict=False`` form.

        :param input_shape: Shape of ``input_ids`` ``(batch, seq_len)``, or a
            dict holding it under ``'input_ids'``.
        :type input_shape: Any
        :return: The LM logits shape ``(batch, seq_len, vocab_size)``. With a
            head, the head's output shapes after the reduction ``call`` applies:
            one shape, or a mapping of two or more.
        :rtype: Any
        """
        ids_shape = ids_shape_of(input_shape)
        if self.include_top:
            return head_output_shape(
                self.task_head, ids_shape, self.hidden_size)
        return (*ids_shape, self.vocab_size)

    def call(
        self,
        inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        return_dict: bool = False
    ) -> Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """
        Forward pass of the Qwen3 model.

        :param inputs: Input token IDs or dictionary containing inputs.
        :param attention_mask: Mask to avoid attention on padding tokens.
        :param training: Whether the model is in training mode.
        :param return_dict: Whether to return outputs as a dictionary.
        :return: With `return_dict=False`, the logits tensor, shape
            (batch, seq_len, vocab_size). With `return_dict=True`, a
            dictionary with key `logits` and optionally others. With
            `include_top=True`, the head's output reduced by
            `finalize_head_outputs` in place of the LM logits: a bare tensor
            (under `logits` when `return_dict=True`), or a dict of two or more
            tensors (`start_logits` / `end_logits` for QA), returned as is.
        """
        # Parse inputs
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError("Dictionary input must contain 'input_ids' key")
            attention_mask = inputs.get("attention_mask", attention_mask)
        else:
            input_ids = inputs

        input_ids = check_sequence_length(
            self, input_ids, self.max_seq_len, "max_seq_len",
            self._SEQUENCE_LENGTH_MESSAGE,
        )

        # Token embeddings
        hidden_states = self.embeddings(input_ids)

        # Causal (+ padding) mask. Qwen3 is a decoder-only causal LM; without
        # this every token attended to every future token.
        causal_attend_mask = create_causal_attend_mask(
            hidden_states, attention_mask)

        # Pass through all transformer blocks
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=causal_attend_mask,
                training=training
            )

        # Final normalization
        hidden_states = self.final_norm(hidden_states)

        if self.include_top:
            # The blocks saw only the caller's mask; the head pools under it,
            # else under input_ids != pad_token_id (all ones if None).
            outputs = finalize_head_outputs(self.task_head(
                {
                    "hidden_states": hidden_states,
                    "attention_mask": head_attention_mask(
                        input_ids, attention_mask, self.pad_token_id,
                        hidden_states.dtype,
                    ),
                },
                training=training,
            ))
            if return_dict and not isinstance(outputs, dict):
                return {"logits": outputs}
            return outputs

        # Language modeling head
        logits = self.lm_head(hidden_states)

        # Return in requested format
        if return_dict:
            return {"logits": logits}
        else:
            return logits

    @classmethod
    def from_variant(
        cls,
        variant: str,
        pretrained: Union[bool, str] = False,
        **kwargs: Any
    ) -> "Qwen3":
        """
        Create a Qwen3 model from a predefined variant.

        :param variant: One of "30b-coder", "medium", "small", "tiny".
        :param pretrained: `False` (default) or any other falsy value: random
            initialization. A `str`: the path of a local `.keras` checkpoint
            whose backbone is restored into the model by layer name, headed or
            headless. The checkpoint may be a `Qwen3` or an old two-input
            functional wrapper around one; both are read through the same rule.
            A task head in the checkpoint is never copied, and the model's own
            head (if any) keeps its fresh initialization. The backbone loads
            completely or the call raises (another depth or width is refused);
            on a vocabulary change only the token table and the `lm_head` keep
            their initialization. To restore a whole model, head included, use
            `keras.models.load_model`. `True` raises `NotImplementedError`: no
            weights are distributed with this package.
        :param kwargs: Additional arguments passed to the constructor,
            including `include_top`, `task_config`, `head_config_overrides`
            and `pad_token_id`.
        :return: Qwen3 model instance.
        :raises ValueError: If `variant` is not a recognized key, or a
            `pretrained` checkpoint does not fit the backbone.
        :raises NotImplementedError: If `pretrained` is True.
        :raises FileNotFoundError: If a `pretrained` path does not exist.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        config.pop("description", None)

        logger.info(f"Creating Qwen3-{variant.upper()} model")
        logger.info(f"Configuration: {cls.MODEL_VARIANTS[variant]['description']}")

        # DECISION plan-2026-08-17T183311-79c63e38/D-025: MERGE, do not splat.
        # `cls(**config, **kwargs)` raises `TypeError: got multiple values for
        # keyword argument` for ANY override of a variant key -- and every
        # overridable key is already in the variant dict, so that was every
        # documented use of `**kwargs` here. Copied from `models/language/gpt2/gpt2.py:464`.
        config.update(kwargs)
        weights_path = pretrained_path_or_raise(pretrained, "Qwen3")
        model = cls(**config)
        if weights_path is not None:
            # A build, not a forward: no RNG is drawn and no weight is touched.
            if not model.built:
                model.build((None, None))
            # DECISION plan-2026-10-11T170958-dc4d2202/D-015: the shared load rule on the
            # wrapper's core, never `model.load_weights` (it restores nothing on a
            # name or shape mismatch and reports success); do NOT drop TASK_HEAD_NAME
            # (a head of equal shape would be copied). See decisions.md.
            # Superseded 2026-10-13: the factories return the class, not a wrapper; the
            # wrapper case applies only to old archives, which the shared rule unwraps.
            load_backbone_from_checkpoint(
                model,
                weights_path,
                head_prefixes=(TASK_HEAD_NAME,),
                vocab_layers=("token_embedding",),
                tables=("token_embedding/embeddings", "lm_head/kernel"),
                tops=("lm_head",),
            )
        return model

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_seq_len": self.max_seq_len,
            "moe_layers": self.moe_layers,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "moe_intermediate_size": self.moe_intermediate_size,
            "rope_theta": self.rope_theta,
            "norm_eps": self.norm_eps,
            "dropout_rate": self.dropout_rate,
            "initializer_range": self.initializer_range,
            "normalization_type": self.normalization_type,
            "ffn_type": self.ffn_type,
            "use_stochastic_depth": self.use_stochastic_depth,
            "stochastic_depth_rate": self.stochastic_depth_rate,
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
    def from_config(cls, config: Dict[str, Any]) -> "Qwen3":
        """Create model from configuration."""
        return cls(**config)

    def summary(self, **kwargs) -> None:
        """Print model summary with additional Qwen3-specific information."""
        super().summary(**kwargs)

        # Calculate statistics
        total_layers = self.num_layers
        moe_layers_count = len(self.moe_layers)
        dense_layers_count = total_layers - moe_layers_count

        logger.info("Qwen3 Model Configuration:")
        logger.info(f"  - Architecture: {total_layers} transformer layers")
        logger.info(f"    - {dense_layers_count} Dense layers")
        logger.info(f"    - {moe_layers_count} MoE layers")
        logger.info(f"  - Hidden size: {self.hidden_size}")
        logger.info(f"  - Attention heads: {self.num_attention_heads} (KV heads: {self.num_key_value_heads})")
        logger.info(f"  - Vocabulary: {self.vocab_size:,} tokens")
        logger.info(f"  - Max sequence length: {self.max_seq_len:,}")
        if self.moe_layers:
            logger.info(f"  - MoE Configuration:")
            logger.info(f"    - MoE layer indices: {self.moe_layers}")
            logger.info(f"    - Experts per layer: {self.num_experts}")
            logger.info(f"    - Active per token: {self.num_experts_per_tok}")
            sparsity_ratio = self.num_experts / self.num_experts_per_tok
            logger.info(f"    - Sparsity ratio: {sparsity_ratio:.1f}:1")
        logger.info(f"  - Normalization: {self.normalization_type}")
        logger.info(f"  - FFN type: {self.ffn_type}")
        if self.use_stochastic_depth:
            logger.info(f"  - Stochastic depth: {self.stochastic_depth_rate}")


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def create_qwen3(
    variant: str,
    pretrained: Union[bool, str] = False,
    **kwargs: Any,
) -> Qwen3:
    """
    Create a Qwen3 model from a shipped `MODEL_VARIANTS` name.

    Delegates to :meth:`Qwen3.from_variant` and holds no logic of its own, as
    `create_gpt2` and `create_zamba2` do. A task head comes from the
    constructor (`include_top=True` with a `task_config`), never from a
    per-task factory.

    :param variant: A key of `Qwen3.MODEL_VARIANTS`: "30b-coder", "medium",
        "small" or "tiny".
    :param pretrained: `False` (default) for random initialization; a `str`
        path of a local `.keras` checkpoint restores the backbone (see
        :meth:`Qwen3.from_variant`); `True` raises `NotImplementedError`.
    :param kwargs: Forwarded to :meth:`Qwen3.from_variant`, overriding any
        variant parameter, including `include_top=True` with a `task_config`
        (and optional `head_config_overrides`) for a model with a task head.
    :return: The `Qwen3` model itself, unbuilt unless `pretrained` is a path.
    :raises ValueError: If `variant` is not a recognized key, or a forwarded
        argument is invalid.
    :raises NotImplementedError: If `pretrained` is True.

    Example:
        .. code-block:: python

            # A 'tiny' language model (returns logits)
            model = create_qwen3("tiny")
            logits = model({"input_ids": ids, "attention_mask": mask})

            # A 'small' sequence classifier with 5 labels (pools 'last')
            clf = create_qwen3(
                "small",
                include_top=True,
                task_config={"name": "sentiment",
                             "task_type": "text_classification",
                             "num_classes": 5},
            )

            # Fewer layers than the variant
            custom = create_qwen3("tiny", num_layers=2)
    """
    return Qwen3.from_variant(variant, pretrained=pretrained, **kwargs)

# ---------------------------------------------------------------------
