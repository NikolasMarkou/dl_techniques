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

Attention is grouped-query with RoPE: GQA shares each key/value head across
several query heads, shrinking the KV cache by that ratio, which is what
lets the 30B-coder variant quote a 262144-token context. `rope_theta` scales
with context length across variants (10,000 at 4096 tokens up to
10,000,000 at 262144) so the slowest rotation's wavelength stays ahead of
the context — otherwise distant positions become numerically
indistinguishable from near ones.

Causality is imposed once at the model level: `call` builds a combined
causal-plus-padding mask via `build_causal_attention_mask` and passes the
same tensor to every block, since neither `TransformerLayer` nor the
attention layers it wraps manufacture one on their own. Blocks are pre-norm
RMSNorm with no biases; the LM head is an independent bias-free `Dense`, so
input and output embeddings are untied. With `return_dict=False` (the
default) the model returns logits, and `create_qwen3_classification` pools
that vocab_size-dimensional tensor rather than the hidden_size hidden
states — a checkpoint's classifier head is not portable to a model that
pools hidden states instead. `from_variant` takes no `pretrained` argument;
no pretrained-weight path exists in this package.

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
from dl_techniques.layers.transformers import TransformerLayer
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.sequence_pooling import SequencePooling
from dl_techniques.layers.moe import MoEConfig, ExpertConfig, GatingConfig
from dl_techniques.layers.ffn import assemble_ffn_config

from .components import build_causal_attention_mask
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.qwen.qwen3")
class Qwen3(keras.Model):
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

    :param vocab_size: Size of the vocabulary. Defaults to 151936.
    :param hidden_size: Dimensionality of encoder layers. Defaults to 2048.
    :param num_layers: Number of transformer blocks. Defaults to 12.
    :param num_attention_heads: Number of attention heads. Defaults to 16.
    :param num_key_value_heads: Number of key-value heads for GQA. Defaults to 4.
    :param max_seq_len: Maximum sequence length. Defaults to 8192.
    :param moe_layers: Layer indices that use MoE. Defaults to empty list.
    :param num_experts: Total number of experts in MoE layers. Defaults to 64.
    :param num_experts_per_tok: Number of experts activated per token. Defaults to 8.
    :param moe_intermediate_size: Individual expert intermediate size. Defaults to 1408.
    :param norm_eps: Epsilon for normalization layers. Defaults to 1e-6.
        Applies to `final_norm` and to every one of the `2 * num_layers`
        in-block norms (decisions.md D-007). Weight shapes are unchanged, so
        existing `.keras` files still load.
    :param dropout_rate: Dropout rate for regularization. Defaults to 0.0.
    :param initializer_range: Standard deviation for weight initialization. Defaults to 0.02.
    :param normalization_type: Type of normalization layer. Defaults to "rms_norm".
    :param ffn_type: Type of feed-forward network in experts. Defaults to "swiglu".
    :param use_stochastic_depth: Whether to enable stochastic depth. Defaults to False.
    :param stochastic_depth_rate: Drop path rate for stochastic depth. Defaults to 0.1.
    :param kwargs: Additional keyword arguments for the `keras.Model` base class.
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

        # Calculate head dimension
        self.head_dim = self.hidden_size // self.num_attention_heads

        # Build the model architecture
        self._build_architecture()

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

            # DECISION plan-2026-07-30T140922-8af1028f/D-037: pre-filter through
            # assemble_ffn_config — a bare dict literal fails Qwen3(ffn_type=...)
            # for 11 of 12 previously-working types. See decisions.md.
            ffn_args = assemble_ffn_config(self.ffn_type, {
                'output_dim': self.hidden_size,
                'ffn_expansion_factor': 4,  # Standard 4x expansion
                'dropout_rate': self.dropout_rate,
                'use_bias': False
            })

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
                moe_config=moe_config if is_moe_layer else None,
                ffn_type=self.ffn_type,
                ffn_args=ffn_args,
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

        # Language modeling head
        self.lm_head = keras.layers.Dense(
            units=self.vocab_size,
            use_bias=False,
            kernel_initializer=keras.initializers.TruncatedNormal(
                stddev=self.initializer_range
            ),
            name='lm_head'
        )

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

        # Causal (+ padding) mask. Qwen3 is a decoder-only causal LM; without
        # this every token attended to every future token.
        causal_attend_mask = build_causal_attention_mask(
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
    ) -> "Qwen3":
        """
        Create a Qwen3 model from a predefined variant.

        :param variant: One of "30b-coder", "medium", "small", "tiny".
        :param kwargs: Additional arguments passed to the constructor.
        :return: Qwen3 model instance.
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

def create_qwen3_generation(config: Dict[str, Any]) -> keras.Model:
    """
    Create a Qwen3 model optimized for text generation tasks.

    This factory builds a Keras model that takes `input_ids` and an
    `attention_mask` and returns token logits, suitable for autoregressive
    text generation.

    :param config: Complete configuration for the `Qwen3` base model.
    :return: A compiled Keras `Model` ready for generation tasks.
    """
    logger.info("Creating Qwen3 model for text generation.")
    logger.debug(f"Generation model config: {config}")

    qwen3_backbone = Qwen3(**config, name="qwen3_backbone")
    input_ids = keras.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = keras.Input(shape=(None,), dtype="int32", name="attention_mask")

    logits = qwen3_backbone(
        inputs={"input_ids": input_ids, "attention_mask": attention_mask}
    )

    model = keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=logits,
        name="qwen3_for_generation"
    )

    param_count = model.count_params()
    logger.info(
        f"Created Qwen3 generation model with {param_count:,} parameters."
    )
    return model

# ---------------------------------------------------------------------

def create_qwen3_classification(
    config: Dict[str, Any],
    num_labels: int,
    pooling_strategy: str = "last",
    classifier_dropout_rate: Optional[float] = None,
) -> keras.Model:
    """
    Create a Qwen3 model for sequence classification tasks.

    This factory adds a classification head on top of the Qwen3 model.
    It supports different pooling strategies for aggregating sequence
    information.

    :param config: Complete configuration for the `Qwen3` base model.
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
    # DECISION plan-2026-08-14T233721-d4f9beb2/D-029: default is "last", not
    # "cls" — Qwen3 is causally masked, so "cls" pools only the first token id.
    # Do not simplify "last" to inputs[:, -1, :]. See decisions.md.
    if pooling_strategy not in ["last", "cls", "mean"]:
        raise ValueError(
            f"pooling_strategy must be 'last', 'cls' or 'mean', got "
            f"'{pooling_strategy}'"
        )

    logger.info(f"Creating Qwen3 classification model with {num_labels} labels.")
    logger.info(f"Using pooling strategy: '{pooling_strategy}'")
    logger.debug(f"Classification model config: {config}")

    qwen3_backbone = Qwen3(**config, name="qwen3_backbone")
    input_ids = keras.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = keras.Input(shape=(None,), dtype="int32", name="attention_mask")

    sequence_output = qwen3_backbone(
        inputs={"input_ids": input_ids, "attention_mask": attention_mask}
    )

    # Apply the selected pooling strategy via the shared SequencePooling layer.
    # DECISION plan-2026-07-15T144225-5b25d9f1/D-001: pool via shipped
    # SequencePooling, not a hand-rolled cls/mean. mask=attention_mask is
    # required for "last" (D-029) to land on the last real token. See decisions.md.
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
        name="qwen3_for_classification"
    )

    param_count = model.count_params()
    logger.info(
        f"Created Qwen3 classification model with {param_count:,} parameters."
    )
    return model


def create_qwen3(
    config_or_variant: Union[str, Dict[str, Any]],
    task_type: str = "generation",
    **kwargs: Any,
) -> keras.Model:
    """
    High-level factory to create Qwen3 models for common tasks.

    This function provides a single, convenient entry point for creating
    different types of Qwen3 models. It allows specifying a model by
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
            gen_model = create_qwen3("tiny")

            # Create a 'small' model for classification with 5 labels
            clf_model = create_qwen3("small", task_type="classification", num_labels=5)

            # Create a custom 'tiny' model with fewer layers for generation
            custom_gen = create_qwen3("tiny", num_layers=2)

            # Create a custom classification model from a dictionary with mean pooling
            my_config = {"hidden_size": 128, "num_layers": 2, ...}
            custom_clf = create_qwen3(
                my_config,
                task_type="classification",
                num_labels=10,
                pooling_strategy="mean"
            )
    """
    # 1. Determine base configuration from variant or dict
    if isinstance(config_or_variant, str):
        variant = config_or_variant
        if variant not in Qwen3.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {list(Qwen3.MODEL_VARIANTS.keys())}"
            )
        config = Qwen3.MODEL_VARIANTS[variant].copy()
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
        return create_qwen3_generation(config)

    elif task_type == "classification":
        num_labels = task_kwargs.pop("num_labels", None)
        if num_labels is None:
            raise ValueError(
                "`num_labels` must be provided for the 'classification' task."
            )
        return create_qwen3_classification(config, num_labels, **task_kwargs)

    else:
        raise ValueError(f"Unknown task_type '{task_type}'. Supported: 'generation', 'classification'.")

# ---------------------------------------------------------------------