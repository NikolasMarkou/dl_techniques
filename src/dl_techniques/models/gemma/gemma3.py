"""
Gemma 3 decoder with interleaved sliding-window / global attention and sandwich
normalization, plus generation and classification task heads.

The problem Gemma 3 is built around is not modelling quality but the cost of
context. Attention is `O(n^2)` in time and, worse for deployment, the key-value
cache it must retain is `O(n * layers * kv_heads)` in memory — at a 32k context the
cache, not the weights, is what will not fit. The architecture attacks that cost on
two independent axes. Grouped-query attention shrinks the cache by the ratio
`num_attention_heads / num_key_value_heads`, since only the distinct KV heads are
stored and the repeat up to the query-head count happens at score time. Interleaved
attention shrinks it again by giving most layers a bounded window: with the 5:1
pattern the variants encode, only one layer in six ever needs keys older than
`sliding_window_size`.

Windowing would be a poor trade if it truly severed long-range dependence, but it
does not. A token's receptive field through a stack of windowed layers grows
roughly as `depth * sliding_window_size`, so information still propagates
backwards, just indirectly and with a hop count. The interleaved full-attention
layers then supply what stacking cannot: an exact, single-hop route to any earlier
position, at the price of a full cache in those layers alone. The `layer_types`
list is the knob for this trade and is validated element-by-element against
`{'sliding_window', 'full_attention'}` with its length pinned to `num_layers`. Note
that leaving `layer_types=None` yields *all* full attention — the interleaving is a
property of the variant tables, not of the constructor defaults.

Each block uses sandwich normalization: RMSNorm before the sublayer and RMSNorm
again on the sublayer's output, with the residual added afterwards, so the
computation is `x = x + PostNorm(Attn(PreNorm(x)))` and likewise for the FFN. The
pre-norm half is the usual conditioning fix. The post-norm half is the part worth
explaining: in ordinary pre-norm transformers the residual stream's variance
accumulates across depth without bound, because nothing ever rescales what each
branch contributes. Normalizing the branch output before the addition caps each
block's contribution while leaving the residual path itself free of any
normalization, so gradients still reach layer zero unattenuated. Four RMSNorm
layers per block is the cost.

Masking happens inside each block rather than once at the model, because the mask
depends on the block's own `attention_type`. `_create_attention_mask` builds it in
*block* semantics — `j > i` for the causal part, OR-ed with `(i - j) >= window` to
cut off the far past — and then inverts it once to the *attend* semantics the
attention layer expects. The inverted mask is explicitly expanded to `(1, q, k)`
before use: a rank-2 mask would be interpreted downstream as a padding mask rather
than a full attention bias, silently discarding causality, so the leading axis is
load-bearing and not merely cosmetic broadcasting. A caller-supplied
`attention_mask` (1 = attend, 0 = pad) is cast to boolean and AND-ed in as
`(batch, 1, k)`, which masks padded *keys* only; padded query rows are left to
produce garbage that the loss is expected to ignore.

Token embeddings are scaled by `sqrt(hidden_size)` before the first block. With a
`TruncatedNormal(0.02)` initializer the raw embeddings are far smaller than the
unit-scale activations the rest of the network is tuned for, and the scaling
restores that match; the factor is computed once in `__init__` against
`compute_dtype` rather than per call.

Several things here deliberately diverge from the published Gemma 3, and a reader
should not assume checkpoint compatibility. There is no QK normalization — the
grouped-query attention layer supports `qk_norm_type` but this block does not pass
it. A single RoPE base of 10000 is used for every layer, whereas the report uses a
much larger base in the global layers specifically so that they remain usable at
long context; consequently the long-context behaviour of the interleaved pattern
here is not the paper's. The LM head is an independent `Dense` rather than the
transposed embedding matrix, so input and output vocabularies are untied. And this
is a text-only decoder: the vision tower of the multimodal Gemma 3 sizes is not
part of this package.

Unlike most model packages here, `from_variant` exposes no `pretrained` argument at
all. That is the strongest form of the house rule that a request for pretrained
weights must never be answerable with a randomly initialized model: there is no
argument to pass, so there is no silent fallback to write.

The two task factories build functional models over the same backbone.
`create_gemma3_classification` re-traces the backbone's embedding, blocks and final
norm into the functional graph instead of calling the backbone as a unit, because
it needs the hidden states rather than the vocabulary logits; pooling then goes
through the shared `SequencePooling` layer so that every strategy behaves
identically to the other decoder packages. It defaults to `last` — the last
position kept by `attention_mask` — because the blocks are causally masked and
`cls` would pool a position that attended only to itself.

References:
    - Gemma Team, 2025. Gemma 3 Technical Report.
      (https://arxiv.org/abs/2503.19786)
    - Gemma Team, 2024. Gemma 2: Improving Open Language Models at a Practical Size.
      (https://arxiv.org/abs/2408.00118)
    - Ainslie et al., 2023. GQA: Training Generalized Multi-Query Transformer Models
      from Multi-Head Checkpoints. (https://arxiv.org/abs/2305.13245)
    - Shazeer, 2019. Fast Transformer Decoding: One Write-Head is All You Need.
      (https://arxiv.org/abs/1911.02150)
    - Beltagy et al., 2020. Longformer: The Long-Document Transformer.
      (https://arxiv.org/abs/2004.05150)
    - Su et al., 2021. RoFormer: Enhanced Transformer with Rotary Position Embedding.
      (https://arxiv.org/abs/2104.09864)
    - Zhang and Sennrich, 2019. Root Mean Square Layer Normalization.
      (https://arxiv.org/abs/1910.07467)
    - Shazeer, 2020. GLU Variants Improve Transformer.
      (https://arxiv.org/abs/2002.05202)
"""


import keras
from keras import initializers, layers, ops
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.sequence_pooling import SequencePooling

from .components import Gemma3TransformerBlock

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class Gemma3(keras.Model):
    """
    Gemma 3 Language Model with dual normalization and mixed attention patterns.

    This model implements Google's Gemma 3 architecture following Modern Keras 3
    best practices. It features token embeddings with scaling, a series of
    transformer blocks, and a final projection head.

    **Intent**: To provide a production-ready Gemma 3 implementation that is
    robust, serializable, and easily integrated with the dl_techniques framework
    for training, optimization, and analysis.

    **Architecture Overview**:
    ```
    Input(input_ids: [batch, seq_len])
           ↓
    Token Embeddings * √(hidden_size)
           ↓
    TransformerBlock₁ (Dual Norm, Mixed Attention)
           ↓
          ...
           ↓
    TransformerBlockₙ (Dual Norm, Mixed Attention)
           ↓
    Final RMSNorm
           ↓
    Linear Projection → Logits([batch, seq_len, vocab_size])
    ```

    Args:
        vocab_size: Integer, size of the vocabulary. Must be positive.
        hidden_size: Integer, dimensionality of encoder layers. Must be
            positive.
        num_layers: Integer, number of transformer blocks. Must be positive.
        num_attention_heads: Integer, number of attention heads.
        num_key_value_heads: Integer, number of key-value heads for GQA.
        ffn_hidden_size: Integer, FFN intermediate size. Must be positive.
        max_seq_len: Integer, maximum sequence length. Must be positive.
        sliding_window_size: Integer, sliding window size for local
            attention.
        layer_types: List of strings, attention type per layer
            ('sliding_window' or 'full_attention'). Length must match
            num_layers.
        norm_eps: Float, epsilon for normalization layers.
        dropout_rate: Float, dropout rate for regularization, in [0, 1].
        use_bias: Boolean, whether to use bias in linear layers.
        initializer_range: Float, stddev for TruncatedNormal weight
            initialization.
        **kwargs: Additional keyword arguments for the Model base class.

    Input shape:
        2D tensor with shape: `(batch_size, sequence_length)` of token IDs.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, vocab_size)`
        of logits.

    Attributes:
        embeddings: Token embedding layer.
        blocks: List of Gemma3TransformerBlock layers.
        final_norm: Final RMSNorm layer before output projection.
        lm_head: Language modeling head (Dense layer).
    """

    # Model variant configurations following Gemma 3 specifications
    MODEL_VARIANTS = {
        "270m": {
            "vocab_size": 262144,
            "hidden_size": 640,
            "num_layers": 18,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "ffn_hidden_size": 2048,
            "max_seq_len": 32768,
            "sliding_window_size": 512,
            "layer_types": [
                "sliding_window", "sliding_window", "sliding_window",
                "sliding_window", "sliding_window", "full_attention",
                "sliding_window", "sliding_window", "sliding_window",
                "sliding_window", "sliding_window", "full_attention",
                "sliding_window", "sliding_window", "sliding_window",
                "sliding_window", "sliding_window", "full_attention",
            ],
            "description": (
                "Gemma 3 270M: Original model with mixed attention patterns."
            ),
        },
        "small": {
            "vocab_size": 50000,
            "hidden_size": 512,
            "num_layers": 12,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "ffn_hidden_size": 1536,
            "max_seq_len": 8192,
            "sliding_window_size": 256,
            "layer_types": [
                "sliding_window", "sliding_window", "full_attention",
                "sliding_window", "sliding_window", "full_attention",
                "sliding_window", "sliding_window", "full_attention",
                "sliding_window", "sliding_window", "full_attention",
            ],
            "description": "Gemma 3 Small: Reduced model for experimentation.",
        },
        "tiny": {
            "vocab_size": 32000,
            "hidden_size": 384,
            "num_layers": 6,
            "num_attention_heads": 6,
            "num_key_value_heads": 2,
            "ffn_hidden_size": 1024,
            "max_seq_len": 4096,
            "sliding_window_size": 128,
            "layer_types": [
                "sliding_window", "sliding_window", "full_attention",
                "sliding_window", "sliding_window", "full_attention",
            ],
            "description": (
                "Gemma 3 Tiny: Minimal model for mobile/edge deployment."
            ),
        },
    }

    def __init__(
        self,
        vocab_size: int = 262144,
        hidden_size: int = 640,
        num_layers: int = 18,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 1,
        ffn_hidden_size: int = 2048,
        max_seq_len: int = 32768,
        sliding_window_size: int = 512,
        layer_types: Optional[List[str]] = None,
        norm_eps: float = 1e-6,
        dropout_rate: float = 0.0,
        use_bias: bool = False,
        initializer_range: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if layer_types is None:
            layer_types = ["full_attention"] * num_layers
        self._validate_config(
            vocab_size,
            hidden_size,
            num_layers,
            num_attention_heads,
            num_key_value_heads,
            ffn_hidden_size,
            max_seq_len,
            sliding_window_size,
            layer_types,
            norm_eps,
            dropout_rate,
            initializer_range,
        )

        # Store ALL configuration parameters for serialization
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.ffn_hidden_size = ffn_hidden_size
        self.max_seq_len = max_seq_len
        self.sliding_window_size = sliding_window_size
        self.layer_types = layer_types
        self.norm_eps = norm_eps
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.initializer_range = initializer_range

        # CREATE all sub-layers in __init__
        self._build_architecture()
        self.emb_scale = ops.sqrt(
            ops.cast(self.hidden_size, dtype=self.compute_dtype)
        )
        self._log_model_creation()

    def _validate_config(self, *args) -> None:
        """Comprehensive model configuration parameter validation."""
        (
            vocab_size, hidden_size, num_layers, num_attention_heads,
            num_key_value_heads, ffn_hidden_size, max_seq_len,
            sliding_window_size, layer_types, norm_eps, dropout_rate,
            initializer_range,
        ) = args
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, got {hidden_size}"
            )
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_attention_heads <= 0:
            raise ValueError(
                "num_attention_heads must be positive, got "
                f"{num_attention_heads}"
            )
        if num_key_value_heads <= 0:
            raise ValueError(
                "num_key_value_heads must be positive, got "
                f"{num_key_value_heads}"
            )
        if ffn_hidden_size <= 0:
            raise ValueError(
                f"ffn_hidden_size must be positive, got {ffn_hidden_size}"
            )
        if max_seq_len <= 0:
            raise ValueError(
                f"max_seq_len must be positive, got {max_seq_len}"
            )
        if sliding_window_size <= 0:
            raise ValueError(
                "sliding_window_size must be positive, got "
                f"{sliding_window_size}"
            )
        if norm_eps <= 0:
            raise ValueError(f"norm_eps must be positive, got {norm_eps}")
        if initializer_range <= 0:
            raise ValueError(
                f"initializer_range must be positive, got {initializer_range}"
            )

        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(
                f"dropout_rate must be in [0, 1], got {dropout_rate}"
            )
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be "
                f"divisible by num_key_value_heads ({num_key_value_heads})"
            )
        if len(layer_types) != num_layers:
            raise ValueError(
                f"layer_types length ({len(layer_types)}) must match "
                f"num_layers ({num_layers})"
            )
        if any(t not in {"sliding_window", "full_attention"} for t in layer_types):
            raise ValueError(
                "Invalid layer_type found. Must be one of 'sliding_window', "
                "'full_attention'."
            )

    def _build_architecture(self) -> None:
        """Create all model components."""
        initializer = initializers.TruncatedNormal(stddev=self.initializer_range)

        self.embeddings = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.hidden_size,
            embeddings_initializer=initializer,
            name="token_embedding",
        )

        self.blocks = [
            Gemma3TransformerBlock(
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                ffn_hidden_size=self.ffn_hidden_size,
                max_seq_len=self.max_seq_len,
                attention_type=self.layer_types[i],
                sliding_window_size=self.sliding_window_size,
                dropout_rate=self.dropout_rate,
                use_bias=self.use_bias,
                norm_eps=self.norm_eps,
                kernel_initializer=initializer,
                name=f"transformer_block_{i}",
            )
            for i in range(self.num_layers)
        ]

        self.final_norm = create_normalization_layer(
            "rms_norm", epsilon=self.norm_eps, name="final_norm"
        )
        self.lm_head = layers.Dense(
            units=self.vocab_size,
            use_bias=self.use_bias,
            kernel_initializer=initializer,
            name="lm_head",
        )

    def _log_model_creation(self) -> None:
        """Log comprehensive model creation information."""
        sliding_count = self.layer_types.count("sliding_window")
        full_count = self.num_layers - sliding_count
        logger.info(
            f"Created Gemma3 model with {self.num_layers} transformer layers:"
        )
        logger.info(
            f"  - Mixed Attention: {sliding_count} sliding window, "
            f"{full_count} full attention"
        )
        logger.info(
            f"  - Vocabulary: {self.vocab_size:,} | "
            f"Hidden Size: {self.hidden_size} | "
            f"FFN Hidden: {self.ffn_hidden_size}"
        )
        logger.info(
            f"  - Attention: {self.num_attention_heads} heads | "
            f"GQA: {self.num_key_value_heads} KV heads"
        )
        logger.info(
            f"  - Context: {self.max_seq_len:,} tokens | "
            f"Sliding Window: {self.sliding_window_size}"
        )

    def call(
        self,
        inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass of the Gemma3 model."""
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError(
                    "Dictionary input must contain 'input_ids' key."
                )
            attention_mask = inputs.get("attention_mask", attention_mask)
        else:
            input_ids = inputs

        hidden_states = self.embeddings(input_ids) * self.emb_scale

        for block in self.blocks:
            hidden_states = block(
                hidden_states, attention_mask=attention_mask, training=training
            )

        hidden_states = self.final_norm(hidden_states)
        return self.lm_head(hidden_states)

    @classmethod
    def from_variant(cls, variant: str, **kwargs: Any) -> "Gemma3":
        """Create a Gemma3 model from a predefined variant."""
        if variant not in cls.MODEL_VARIANTS:
            available = list(cls.MODEL_VARIANTS.keys())
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {available}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        description = config.pop("description", "")
        config.update(kwargs)

        logger.info(f"Creating Gemma3 model from variant: {variant.upper()}")
        logger.info(f"Description: {description}")
        return cls(**config)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "ffn_hidden_size": self.ffn_hidden_size,
                "max_seq_len": self.max_seq_len,
                "sliding_window_size": self.sliding_window_size,
                "layer_types": self.layer_types,
                "norm_eps": self.norm_eps,
                "dropout_rate": self.dropout_rate,
                "use_bias": self.use_bias,
                "initializer_range": self.initializer_range,
            }
        )
        return config


def create_gemma3_generation(config: Dict[str, Any]) -> keras.Model:
    """Creates a Gemma3 model for text generation tasks."""
    logger.info("Creating Gemma3 model for text generation.")
    gemma3_backbone = Gemma3(**config, name="gemma3_backbone")
    input_ids = keras.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = keras.Input(
        shape=(None,), dtype="int32", name="attention_mask"
    )
    logits = gemma3_backbone(
        inputs={"input_ids": input_ids, "attention_mask": attention_mask}
    )
    model = keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=logits,
        name="gemma3_for_generation",
    )
    logger.info(
        f"Created Gemma3 generation model with {model.count_params():,} "
        "parameters."
    )
    return model


def create_gemma3_classification(
    config: Dict[str, Any],
    num_labels: int,
    pooling_strategy: str = "last",
    classifier_dropout: Optional[float] = None,
) -> keras.Model:
    """Creates a Gemma3 model for sequence classification tasks.

    ``pooling_strategy`` is one of ``"last"`` (default — the last position kept
    by ``attention_mask``, the only one that has attended to the whole sequence
    under this backbone's causal mask), ``"mean"`` (mask-aware mean) or
    ``"cls"`` (first position; a function of the first token id alone here, kept
    only for bidirectional-era checkpoints).
    """
    if num_labels <= 0:
        raise ValueError(f"num_labels must be positive, got {num_labels}")
    # DECISION plan-2026-08-14T233721-d4f9beb2/D-029: default "last", not "cls" —
    # Gemma3's blocks are strictly causally masked (sliding-window and full
    # layers alike), so `cls` pools a position that attended only to itself.
    # Same mechanism, same measurement and the same do-not-restore instruction
    # as `qwen/qwen3.py`. See decisions.md D-029.
    if pooling_strategy not in ["last", "cls", "mean"]:
        raise ValueError(
            f"pooling_strategy must be 'last', 'cls' or 'mean', got "
            f"'{pooling_strategy}'"
        )

    logger.info(
        f"Creating Gemma3 classification model "
        f"(Labels: {num_labels}, Pooling: {pooling_strategy})."
    )
    gemma3_backbone = Gemma3(**config, name="gemma3_backbone")
    input_ids = keras.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = keras.Input(
        shape=(None,), dtype="int32", name="attention_mask"
    )

    # Trace the computation graph through the backbone's layers
    hidden_states = gemma3_backbone.embeddings(input_ids) * gemma3_backbone.emb_scale
    for block in gemma3_backbone.blocks:
        hidden_states = block(hidden_states, attention_mask=attention_mask)
    base_output = gemma3_backbone.final_norm(hidden_states)

    # Apply the selected pooling strategy via the shared SequencePooling layer
    # (byte-identical cls/mean; see qwen3.py DECISION D-001).
    pooled_output = SequencePooling(strategy=pooling_strategy, name="pooler")(
        base_output, mask=attention_mask
    )

    dropout_rate = (
        classifier_dropout
        if classifier_dropout is not None
        else config.get("dropout_rate", 0.0)
    )
    if dropout_rate > 0.0:
        pooled_output = layers.Dropout(
            dropout_rate, name="classifier_dropout"
        )(pooled_output)

    initializer = initializers.TruncatedNormal(
        stddev=config.get("initializer_range", 0.02)
    )
    logits = layers.Dense(
        units=num_labels,
        kernel_initializer=initializer,
        name="classifier_head",
    )(pooled_output)

    model = keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=logits,
        name="gemma3_for_classification",
    )
    logger.info(
        f"Created Gemma3 classification model with {model.count_params():,} "
        "parameters."
    )
    return model


def create_gemma3(
    config_or_variant: Union[str, Dict[str, Any]],
    task_type: str = "generation",
    **kwargs: Any,
) -> keras.Model:
    """High-level factory to create Gemma3 models for common tasks."""
    if isinstance(config_or_variant, str):
        if config_or_variant not in Gemma3.MODEL_VARIANTS:
            available = list(Gemma3.MODEL_VARIANTS.keys())
            raise ValueError(
                f"Unknown variant '{config_or_variant}'. "
                f"Available: {available}"
            )
        config = Gemma3.MODEL_VARIANTS[config_or_variant].copy()
        config.pop("description", None)
        logger.info(f"Using Gemma3 variant: {config_or_variant}")
    elif isinstance(config_or_variant, dict):
        config = config_or_variant.copy()
        logger.info("Using custom Gemma3 configuration")
    else:
        raise TypeError("config_or_variant must be a string or a dictionary.")

    task_keys = ["num_labels", "pooling_strategy", "classifier_dropout"]
    task_kwargs = {k: kwargs.pop(k) for k in task_keys if k in kwargs}
    config.update(kwargs)  # The rest are model overrides

    if task_type == "generation":
        return create_gemma3_generation(config)
    if task_type == "classification":
        if "num_labels" not in task_kwargs:
            raise ValueError(
                "num_labels must be provided for the 'classification' task"
            )
        return create_gemma3_classification(config, **task_kwargs)

    raise ValueError(
        f"Unknown task_type '{task_type}'. "
        "Supported: ['generation', 'classification']"
    )