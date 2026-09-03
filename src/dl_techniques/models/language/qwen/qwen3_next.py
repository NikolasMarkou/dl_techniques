"""
Qwen3-Next hybrid decoder: blocks of three gated linear-attention sublayers
followed by one full gated-attention sublayer, each with its own
normalization and optional mixture-of-experts feed-forward.

Softmax attention can retrieve any earlier token exactly but its compute
and cache both grow with sequence length. A gated linear-attention layer
folds the past into a fixed-size matrix state instead, giving `O(L)` time
and `O(1)` state at the cost of exact recall. Each `Qwen3NextBlock` runs
three linear-attention sublayers and then one softmax-attention sublayer,
so the cheap mixers do most of the contextualization and the expensive one
supplies exact global lookup. Only one sublayer in four holds a KV cache, so
cache memory falls to roughly a quarter of a uniformly attentive stack of
the same effective depth. `num_layers` counts blocks, not sublayers — the
`80b_a3b` variant's 12 blocks are 48 effective layers.

Mixture-of-experts, when configured, replaces the FFN with `num_experts`
experts of which `num_experts_per_tok` are routed, decoupling parameter
count from per-token arithmetic; the `80b` variant sets `num_experts=1` to
express a dense configuration instead of using a separate flag.

`call` builds a combined causal-plus-padding mask once and passes it only
to the gated-attention sublayer. The three linear-attention sublayers get
no mask: their scan is a strictly left-to-right recurrence with causal
depthwise convolutions and cannot read forward, so causality holds
structurally, but padding does not — padded positions enter the recurrent
state, so left-padding contaminates a token's summary while right-padding
does not.

There is no model-level positional embedding; position enters through RoPE
inside the gated-attention sublayer and through the causal convolutions and
ordered recurrence of the linear-attention sublayers. `from_variant`
exposes no `pretrained` argument. With `return_dict=False` (the default)
the model returns logits, and `create_qwen3_next_classification` pools that
vocab_size-wide tensor rather than hidden states, so a classifier trained
one way does not transfer to the other.

References:
    - Qwen Team, 2025. Qwen3 Technical Report.
    - Yang et al., 2024. Gated Linear Attention Transformers with Hardware-Efficient
      Training. (https://arxiv.org/abs/2312.06635)
    - Yang et al., 2024. Parallelizing Linear Transformers with the Delta Rule over
      Sequence Length. (https://arxiv.org/abs/2406.06484)
    - Katharopoulos et al., 2020. Transformers are RNNs: Fast Autoregressive
      Transformers with Linear Attention. (https://arxiv.org/abs/2006.16236)
    - Ainslie et al., 2023. GQA: Training Generalized Multi-Query Transformer Models
      from Multi-Head Checkpoints. (https://arxiv.org/abs/2305.13245)
    - Su et al., 2021. RoFormer: Enhanced Transformer with Rotary Position Embedding.
      (https://arxiv.org/abs/2104.09864)
    - Shazeer et al., 2017. Outrageously Large Neural Networks: The Sparsely-Gated
      Mixture-of-Experts Layer. (https://arxiv.org/abs/1701.06538)
    - Zhang and Sennrich, 2019. Root Mean Square Layer Normalization.
      (https://arxiv.org/abs/1910.07467)
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
"""

import keras
from typing import Optional, Union, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.utils.model_build import concretize_axes, materialize_sublayers
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.sequence_pooling import SequencePooling
from dl_techniques.layers.moe import MoEConfig, ExpertConfig, GatingConfig
from dl_techniques.layers.ffn import assemble_ffn_config

from .components import Qwen3NextBlock, build_causal_attention_mask
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.qwen.qwen3_next")
class Qwen3Next(keras.Model):
    """
    Qwen3-Next model: N hybrid blocks of gated linear attention and gated attention.

    Architecture:

    .. code-block:: text

        Input (input_ids)
               │
               ▼
        Token Embeddings [vocab_size, dim]
               │
               ▼   (no model-level positional stage: position enters inside
               │    the blocks, via RoPE in the gated-attention sublayer and
               │    the causal convolutions of the linear-attention ones)
        Qwen3NextBlock x num_layers:
            3x [Zero-Centered RMSNorm → GatedLinearAttentionBlock → MoE → Residual]
            1x [Zero-Centered RMSNorm → GatedAttention → MoE → Residual]
               │
               ▼
        Final Zero-Centered RMSNorm
               │
               ▼
        Linear Projection → Logits [vocab_size]

    :param vocab_size: Size of the vocabulary. Defaults to 151936.
    :param hidden_size: Dimensionality of encoder layers. Defaults to 2048.
    :param num_layers: Number of transformer blocks. Defaults to 12.
    :param num_attention_heads: Number of attention heads. Defaults to 16.
    :param num_key_value_heads: Number of key/value heads for grouped-query
        attention in each block's gated-attention sublayer. Must divide
        num_attention_heads. Defaults to 4, so the default model's KV cache
        is num_attention_heads // num_key_value_heads times smaller than plain MHA.
    :param max_seq_len: Maximum sequence length. Defaults to 8192.
    :param num_experts: Total number of experts in MoE layers. Defaults to 64.
    :param num_experts_per_tok: Number of experts activated per token. Defaults to 8.
    :param moe_intermediate_size: Individual expert intermediate size. Defaults to 1408.
    :param norm_eps: Epsilon for normalization layers. Defaults to 1e-6.
    :param dropout_rate: Dropout rate for regularization. Defaults to 0.0.
    :param initializer_range: Standard deviation for weight initialization. Defaults to 0.02.
    :param normalization_type: Type of normalization layer. Defaults to "zero_centered_rms_norm".
    :param ffn_type: Type of feed-forward network in experts. Defaults to "swiglu".
    :param use_stochastic_depth: Whether to enable stochastic depth. Defaults to False.
    :param stochastic_depth_rate: Drop path rate for stochastic depth. Defaults to 0.1.
    :param kwargs: Additional keyword arguments for the `keras.Model` base class.
    """

    # Model variant configurations following Qwen3 Next specifications
    MODEL_VARIANTS = {
        # DECISION plan-2026-08-22T035419-a11304c8/D-112 and D-204: every value
        # below is the released Qwen3-Next-80B-A3B config, fetched 2026-08-22;
        # do not shrink for feasibility, use from_variant(num_experts=...) instead.
        # See decisions.md.
        "80b_a3b": {
            "vocab_size": 151936,
            "hidden_size": 2048,
            "num_layers": 12,  # 12 blocks, each with 3 delta + 1 attn = 48 layers total
            "num_attention_heads": 16,
            "head_dim": 256,  # RELEASED value; NOT hidden_size // num_attention_heads
            "num_key_value_heads": 2,
            "max_seq_len": 262144,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 512,
            "description": "Qwen3 Next 80B-A3B: 12 blocks × (3 delta + 1 attn) = 48 effective layers"
        },
        "80b": {
            "vocab_size": 151936,
            "hidden_size": 2048,
            "num_layers": 12,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "max_seq_len": 8192,
            "num_experts": 1,  # Dense model
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 5632,
            "description": "Qwen3 Next 80B Dense: 12 blocks without MoE"
        },
        "small": {
            "vocab_size": 151936,
            "hidden_size": 1024,
            "num_layers": 6,  # 6 blocks
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "max_seq_len": 2048,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 704,
            "description": "Qwen3 Next Small: 6 blocks for experimentation"
        },
        "tiny": {
            "vocab_size": 151936,
            "hidden_size": 512,
            "num_layers": 3,  # 3 blocks
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "max_seq_len": 1024,
            "num_experts": 4,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 352,
            "description": "Qwen3 Next Tiny: 3 blocks for mobile/edge deployment"
        },
    }

    def __init__(
            self,
            vocab_size: int = 151936,
            hidden_size: int = 2048,
            num_layers: int = 12,
            num_attention_heads: int = 16,
            num_key_value_heads: int = 4,
            head_dim: Optional[int] = None,
            max_seq_len: int = 8192,
            num_experts: int = 64,
            num_experts_per_tok: int = 8,
            moe_intermediate_size: int = 1408,
            norm_eps: float = 1e-6,
            dropout_rate: float = 0.0,
            initializer_range: float = 0.02,
            normalization_type: str = "zero_centered_rms_norm",
            ffn_type: str = "swiglu",
            use_stochastic_depth: bool = False,
            stochastic_depth_rate: float = 0.1,
            **kwargs: Any
    ) -> None:
        # CRITICAL FIX: Call super() FIRST. This is mandatory for Keras models.
        super().__init__(**kwargs)

        # Validate configuration parameters
        self._validate_config(
            vocab_size, hidden_size, num_layers, num_attention_heads,
            num_key_value_heads, num_experts, num_experts_per_tok
        )

        # Store all configuration parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_seq_len = max_seq_len
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.norm_eps = norm_eps
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.normalization_type = normalization_type
        self.ffn_type = ffn_type
        self.use_stochastic_depth = use_stochastic_depth
        self.stochastic_depth_rate = stochastic_depth_rate

        # DECISION plan-2026-08-22T035419-a11304c8/D-204: head_dim is decoupled
        # upstream (released 256 vs quotient 128) and must stay an explicit
        # override; None keeps the quotient for every other variant. See decisions.md.
        self.head_dim = (
            int(head_dim)
            if head_dim is not None
            else self.hidden_size // self.num_attention_heads
        )
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {self.head_dim}")

        # Build the model architecture
        self._build_architecture()

        # Log model creation
        total_effective_layers = num_layers * 4  # Each block has 3 delta + 1 attn
        active_params_pct = (self.num_experts_per_tok / self.num_experts) * 100 if self.num_experts > 1 else 100.0
        logger.info(
            f"Created Qwen3 Next model: {self.num_layers} blocks "
            f"({total_effective_layers} effective layers), "
            f"hidden_size={self.hidden_size}, experts={self.num_experts}, "
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

        # CRITICAL FIX: Removed unused `self.rope_embedding`.
        # RoPE is handled within each `GatedAttention` layer, not at the model level.

        # Create MoE configuration
        moe_config = None
        if self.num_experts > 1:
            moe_config = MoEConfig(
                num_experts=self.num_experts,
                expert_config=ExpertConfig(
                    # DECISION plan-2026-07-30T140922-8af1028f/D-037: pre-filter
                    # against ffn_type — the factory raises on an unaccepted key.
                    # Full reasoning at the twin anchor in qwen3.py. See decisions.md.
                    ffn_config=assemble_ffn_config(self.ffn_type, {
                        "type": self.ffn_type,
                        "output_dim": self.hidden_size,
                        "ffn_expansion_factor": max(1, self.moe_intermediate_size // self.hidden_size)
                    })
                ),
                gating_config=GatingConfig(
                    top_k=self.num_experts_per_tok,
                    gating_type="linear"
                )
            )

        # Create a linear schedule for the drop path rate
        dpr = linear_drop_path_rates(self.num_layers, self.stochastic_depth_rate)

        # Create Qwen3Next blocks
        self.blocks = []
        for i in range(self.num_layers):
            block = Qwen3NextBlock(
                dim=self.hidden_size,
                num_heads=self.num_attention_heads,
                # DECISION plan-2026-08-14T233721-d4f9beb2/D-071: wiring this in is
                # a checkpoint break — never pass skip_mismatch=True loading an old
                # checkpoint, it silently randomizes K/V projections. See decisions.md.
                num_kv_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                max_seq_len=self.max_seq_len,
                moe_config=moe_config,
                normalization_type=self.normalization_type,
                norm_eps=self.norm_eps,
                dropout_rate=self.dropout_rate,
                use_stochastic_depth=self.use_stochastic_depth,
                stochastic_depth_rate=dpr[i],
                name=f"qwen3_next_block_{i}"
            )
            self.blocks.append(block)

        # Final normalization layer
        self.final_norm = create_normalization_layer(
            self.normalization_type,
            epsilon=self.norm_eps,
            name='final_norm'
        )

        # Language modeling head
        self.lm_head = keras.layers.Dense(
            units=self.vocab_size,
            use_bias=False,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name='lm_head'
        )

    # DECISION plan-2026-08-23T091307-9a110062/D-421: coerce the sequence axis
    # to a concrete 1 before tracing — materialize_sublayers calls `call` on
    # KerasTensor placeholders where seq_len is None. No weight shape here
    # depends on seq_len. Do not generalize this into materialize_sublayers
    # itself; other models on that helper have not been checked. See decisions.md.
    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer from an explicit `build` call.

        :param input_shape: Shape, or nest of shapes, of `call`'s `inputs`.
        """
        if self.built:
            return
        materialize_sublayers(self, concretize_axes(input_shape, {1: 1}))
        super().build(input_shape)

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None,
            return_dict: bool = False
    ) -> Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """
        Forward pass of the Qwen3 Next model.

        :param inputs: Input token IDs or dictionary containing inputs.
        :param attention_mask: Mask to avoid attention on padding tokens.
        :param training: Whether the model is in training mode.
        :param return_dict: Whether to return outputs as a dictionary.
        :return: With `return_dict=False`, the logits tensor, shape
            (batch, seq_len, vocab_size). With `return_dict=True`, a
            dictionary with key `logits` and optionally others.
        """
        # Parse inputs
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError("Dictionary input must contain 'input_ids' key")
            attention_mask = inputs.get("attention_mask", attention_mask)
        else:
            input_ids = inputs

        # Token embeddings
        hidden_states = self.embeddings(input_ids)

        # Causal (+ padding) mask. Qwen3Next is a decoder-only causal LM;
        # without this every token attended to every future token.
        causal_attend_mask = build_causal_attention_mask(
            hidden_states, attention_mask)

        # Pass through all Qwen3Next blocks
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=causal_attend_mask,
                training=training
            )

        # Final normalization
        hidden_states = self.final_norm(hidden_states)

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
            **kwargs: Any
    ) -> "Qwen3Next":
        """
        Create a Qwen3 Next model from a predefined variant.

        :param variant: One of "80b_a3b", "80b", "small", "tiny".
        :param kwargs: Additional arguments passed to the constructor.
        :return: Qwen3Next model instance.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        config.pop("description", None)

        logger.info(f"Creating Qwen3Next-{variant.upper()} model")
        logger.info(f"Configuration: {cls.MODEL_VARIANTS[variant]['description']}")

        # DECISION plan-2026-08-17T183311-79c63e38/D-025: MERGE, do not splat.
        # See the identical note in `qwen3.py::Qwen3.from_variant`.
        config.update(kwargs)
        return cls(**config)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "max_seq_len": self.max_seq_len,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "moe_intermediate_size": self.moe_intermediate_size,
            "norm_eps": self.norm_eps,
            "dropout_rate": self.dropout_rate,
            "initializer_range": self.initializer_range,
            "normalization_type": self.normalization_type,
            "ffn_type": self.ffn_type,
            "use_stochastic_depth": self.use_stochastic_depth,
            "stochastic_depth_rate": self.stochastic_depth_rate,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Qwen3Next":
        """Create model from configuration."""
        return cls(**config)

    def summary(self, **kwargs) -> None:
        """Print model summary with additional Qwen3-specific information."""
        super().summary(**kwargs)

        # Calculate statistics
        total_blocks = self.num_layers
        total_delta_layers = total_blocks * 3
        total_attention_layers = total_blocks * 1
        total_effective_layers = total_delta_layers + total_attention_layers
        total_experts = self.num_experts * total_effective_layers if self.num_experts > 1 else 0
        active_experts_per_token = self.num_experts_per_tok * total_effective_layers if self.num_experts > 1 else 0
        sparsity_ratio = self.num_experts / self.num_experts_per_tok if self.num_experts > 1 else 1

        logger.info("Qwen3 Next Model Configuration:")
        logger.info(f"  - Architecture: {total_blocks} blocks → {total_effective_layers} effective layers")
        logger.info(f"    - {total_delta_layers} Gated DeltaNet layers")
        logger.info(f"    - {total_attention_layers} Gated Attention layers")
        logger.info(f"  - Hidden size: {self.hidden_size}")
        logger.info(f"  - Attention heads: {self.num_attention_heads} (KV heads: {self.num_key_value_heads})")
        logger.info(f"  - Vocabulary: {self.vocab_size:,} tokens")
        logger.info(f"  - Max sequence length: {self.max_seq_len:,}")
        if self.num_experts > 1:
            logger.info(f"  - MoE Configuration:")
            logger.info(f"    - Experts per layer: {self.num_experts}")
            logger.info(f"    - Active per token: {self.num_experts_per_tok}")
            logger.info(f"    - Sparsity ratio: {sparsity_ratio:.1f}:1")
            logger.info(f"    - Total experts: {total_experts:,}")
            logger.info(f"    - Active experts per token: {active_experts_per_token}")
        logger.info(f"  - Normalization: {self.normalization_type}")
        logger.info(f"  - Expert FFN: {self.ffn_type}")
        if self.use_stochastic_depth:
            logger.info(f"  - Stochastic depth: {self.stochastic_depth_rate}")


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def create_qwen3_next_generation(config: Dict[str, Any]) -> keras.Model:
    """
    Create a Qwen3 Next model optimized for text generation tasks.

    This factory builds a Keras model that takes `input_ids` and an
    `attention_mask` and returns token logits, suitable for autoregressive
    text generation.

    :param config: Complete configuration for the `Qwen3Next` base model.
    :return: A compiled Keras `Model` ready for generation tasks.
    """
    logger.info("Creating Qwen3 Next model for text generation.")
    logger.debug(f"Generation model config: {config}")

    qwen3_next_backbone = Qwen3Next(**config, name="qwen3_next_backbone")
    input_ids = keras.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = keras.Input(shape=(None,), dtype="int32", name="attention_mask")

    logits = qwen3_next_backbone(
        inputs={"input_ids": input_ids, "attention_mask": attention_mask}
    )

    model = keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=logits,
        name="qwen3_next_for_generation"
    )

    param_count = model.count_params()
    logger.info(
        f"Created Qwen3 Next generation model with {param_count:,} parameters."
    )
    return model

# ---------------------------------------------------------------------

def create_qwen3_next_classification(
    config: Dict[str, Any],
    num_labels: int,
    pooling_strategy: str = "last",
    classifier_dropout_rate: Optional[float] = None,
) -> keras.Model:
    """
    Create a Qwen3 Next model for sequence classification tasks.

    This factory adds a classification head on top of the Qwen3 Next model.
    It supports different pooling strategies for aggregating sequence
    information.

    :param config: Complete configuration for the `Qwen3Next` base model.
    :param num_labels: Number of output labels for the classification task.
    :param pooling_strategy: Method to pool the sequence output. "last" uses
        the output at the last position kept by `attention_mask` — the only
        position that has attended to the whole sequence under this
        backbone's causal mask. "mean" averages all token outputs,
        respecting the attention mask. "cls" uses the first token's output;
        under a causal mask that position attends only to itself, so the
        pooled vector is a function of the first token id alone, and this
        mode is kept only for bidirectional-era checkpoints. Defaults to "last".
    :param classifier_dropout_rate: Dropout rate for the classification
        head. If None, defaults to the `dropout_rate` from `config`.
    :return: A compiled Keras `Model` ready for classification tasks.
    """
    if num_labels <= 0:
        raise ValueError(f"num_labels must be positive, got {num_labels}")
    # DECISION plan-2026-08-14T233721-d4f9beb2/D-029: default "last", not "cls" —
    # this backbone is strictly causally masked, so `cls` pools a position that
    # attended only to itself. Same mechanism, same measurement and the same
    # do-not-restore instruction as `qwen3.py`. See decisions.md D-029.
    if pooling_strategy not in ["last", "cls", "mean"]:
        raise ValueError(
            f"pooling_strategy must be 'last', 'cls' or 'mean', got "
            f"'{pooling_strategy}'"
        )

    logger.info(f"Creating Qwen3 Next classification model with {num_labels} labels.")
    logger.info(f"Using pooling strategy: '{pooling_strategy}'")
    logger.debug(f"Classification model config: {config}")

    qwen3_next_backbone = Qwen3Next(**config, name="qwen3_next_backbone")
    input_ids = keras.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = keras.Input(shape=(None,), dtype="int32", name="attention_mask")

    sequence_output = qwen3_next_backbone(
        inputs={"input_ids": input_ids, "attention_mask": attention_mask}
    )

    # Apply the selected pooling strategy via the shared SequencePooling layer
    # (byte-identical cls/mean; see qwen3.py DECISION D-001).
    pooled_output = SequencePooling(strategy=pooling_strategy, name="pooler")(
        sequence_output, mask=attention_mask
    )

    # Determine classifier dropout
    dropout_rate = classifier_dropout_rate if classifier_dropout_rate is not None else config.get("dropout_rate", 0.1)
    if dropout_rate > 0.0:
        logger.info(f"Applying classifier dropout with rate: {dropout_rate}")
        pooled_output = keras.layers.Dropout(
            dropout_rate, name="classifier_dropout"
        )(pooled_output)

    # Final classification layer
    initializer_range = config.get("initializer_range", 0.02)
    logits = keras.layers.Dense(
        units=num_labels,
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=initializer_range),
        name="classifier_head",
    )(pooled_output)

    model = keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=logits,
        name="qwen3_next_for_classification"
    )

    param_count = model.count_params()
    logger.info(
        f"Created Qwen3 Next classification model with {param_count:,} parameters."
    )
    return model

# ---------------------------------------------------------------------

def create_qwen3_next(
    config_or_variant: Union[str, Dict[str, Any]],
    task_type: str = "generation",
    **kwargs: Any,
) -> keras.Model:
    """
    High-level factory to create Qwen3 Next models for common tasks.

    This function provides a single, convenient entry point for creating
    different types of Qwen3 Next models. It allows specifying a model by
    a predefined variant string or a custom configuration dictionary, and
    supports overriding any parameter via keyword arguments.

    Configuration Precedence:
    1. Predefined variant defaults.
    2. Overridden by `config_or_variant` if it's a dictionary.
    3. Finally, overridden by any explicit `**kwargs`.

    :param config_or_variant: Either a variant string (e.g., "tiny", "small")
        or a dictionary with a custom model configuration.
    :param task_type: Type of model to create: "generation" for
        autoregressive language modeling, or "classification" for sequence
        classification.
    :param kwargs: Additional keyword arguments to override configuration
        parameters (e.g. `hidden_size`, `num_layers`) or provide
        task-specific settings for classification (`num_labels`,
        `pooling_strategy`, `classifier_dropout_rate`).
    :return: A Keras `Model` configured for the specified task.

    Example:
        .. code-block:: python

            # Create a standard 'tiny' model for generation
            gen_model = create_qwen3_next("tiny")

            # Create a 'small' model for classification with 5 labels
            clf_model = create_qwen3_next("small", task_type="classification", num_labels=5)

            # Create a custom 'tiny' model with fewer layers for generation
            custom_gen = create_qwen3_next("tiny", num_layers=2)

            # Create a custom classification model from a dictionary with mean pooling
            my_config = {"hidden_size": 128, "num_layers": 2, ...}
            custom_clf = create_qwen3_next(
                my_config,
                task_type="classification",
                num_labels=10,
                pooling_strategy="mean"
            )
    """
    # 1. Determine base configuration from variant or dict
    if isinstance(config_or_variant, str):
        variant = config_or_variant
        if variant not in Qwen3Next.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {list(Qwen3Next.MODEL_VARIANTS.keys())}"
            )
        config = Qwen3Next.MODEL_VARIANTS[variant].copy()
        config.pop("description", None)
    elif isinstance(config_or_variant, dict):
        config = config_or_variant.copy()
    else:
        raise TypeError(
            "config_or_variant must be a string (variant name) or a dictionary."
        )

    # 2. Separate task-specific kwargs from model config kwargs
    task_kwargs = {}
    model_kwargs = {}
    task_specific_keys = ["num_labels", "pooling_strategy", "classifier_dropout_rate"]

    for key, value in kwargs.items():
        if key in task_specific_keys:
            task_kwargs[key] = value
        else:
            model_kwargs[key] = value

    # 3. Apply overrides to the base model config
    config.update(model_kwargs)

    # 4. Build the requested model based on task_type
    if task_type == "generation":
        if "num_labels" in task_kwargs:
            logger.warning("`num_labels` is ignored for 'generation' task type.")
        return create_qwen3_next_generation(config)

    elif task_type == "classification":
        num_labels = task_kwargs.pop("num_labels", None)
        if num_labels is None:
            raise ValueError(
                "`num_labels` must be provided for the 'classification' task."
            )
        return create_qwen3_next_classification(config, num_labels, **task_kwargs)

    else:
        raise ValueError(f"Unknown task_type '{task_type}'. Supported: 'generation', 'classification'.")

# ---------------------------------------------------------------------