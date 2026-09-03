"""
Gemma 3 decoder with interleaved sliding-window / global attention and sandwich
normalization, plus generation and classification task heads.

Gemma 3 targets inference memory rather than raw quality: attention costs
O(n^2) in time, and its key-value cache costs O(n * layers * kv_heads) in
memory, which is what fails to fit at long context. Grouped-query attention
shrinks the cache by storing only `num_key_value_heads` distinct KV heads and
repeating them at score time. Most layers also use a bounded attention window
(`sliding_window_size`), with the `layer_types` list controlling how often a
layer instead gets full, unbounded attention. A token's receptive field through
the windowed layers still grows with depth; the full-attention layers add an
exact single-hop route to any earlier position.

Each block uses sandwich normalization: RMSNorm before and after the sublayer,
`x = x + PostNorm(Attn(PreNorm(x)))`, so each branch's contribution is capped
before it joins the residual stream, while the residual path itself stays
unnormalized. Masking is built per block, since it depends on that block's
attention type, then inverted once to the attend-semantics the attention layer
expects. Token embeddings are scaled by `sqrt(hidden_size)` before the first
block to match the unit-scale activations the rest of the network expects.

This implementation omits QK normalization and uses one shared RoPE base for
every layer rather than a larger base in the global layers, so long-context
behavior differs from the published report; a reader should not assume
checkpoint compatibility. The LM head is an independent `Dense`, not the tied
embedding matrix. This package is text-only; the multimodal vision tower is
not included. `from_variant` takes no `pretrained` argument, so there is no
path to a silently random-initialized "pretrained" model.

`create_gemma3_classification` re-traces the backbone into a functional graph
to reach hidden states before the LM head, then pools with the shared
`SequencePooling` layer. It defaults to `last` pooling, the last position kept
by `attention_mask`, because the causally masked blocks make `cls` pool a
position that attended only to itself.

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
from dl_techniques.utils.model_build import materialize_sublayers
from dl_techniques.layers.transformers.gemma3_transformer import Gemma3TransformerBlock
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.gemma.gemma3")
class Gemma3(keras.Model):
    """Gemma 3 language model with sandwich normalization and mixed attention.

    Token embeddings, scaled by sqrt(hidden_size), feed a stack of transformer
    blocks that alternate sliding-window and full attention per `layer_types`,
    then a final RMSNorm and a Dense projection to vocabulary logits.

    Architecture:

        .. code-block:: text

            input_ids [B, S]
                 │
                 ▼
            Embedding * sqrt(hidden_size)
                 │
                 ▼
            ┌────────────────────────────┐
            │ Gemma3TransformerBlock x N  │  sandwich norm, per-layer
            │ (sliding_window|full)       │  attention_type from layer_types
            └────────────────────────────┘
                 │
                 ▼
            final_norm (RMSNorm)
                 │
                 ▼
            lm_head (Dense, untied)
                 │
                 ▼
            logits [B, S, vocab_size]

    :param vocab_size: Vocabulary size. Must be positive.
    :type vocab_size: int
    :param hidden_size: Dimensionality of encoder layers. Must be positive.
    :type hidden_size: int
    :param num_layers: Number of transformer blocks. Must be positive.
    :type num_layers: int
    :param num_attention_heads: Number of attention heads.
    :type num_attention_heads: int
    :param num_key_value_heads: Number of key-value heads for GQA.
    :type num_key_value_heads: int
    :param ffn_hidden_size: FFN intermediate size. Must be positive.
    :type ffn_hidden_size: int
    :param max_seq_len: Maximum sequence length. Must be positive.
    :type max_seq_len: int
    :param sliding_window_size: Window size for local attention layers.
    :type sliding_window_size: int
    :param layer_types: Attention type per layer, each ``'sliding_window'``
        or ``'full_attention'``. Length must match ``num_layers``. ``None``
        (default) yields all ``'full_attention'``.
    :type layer_types: Optional[List[str]]
    :param norm_eps: Epsilon for normalization layers.
    :type norm_eps: float
    :param dropout_rate: Dropout rate, in [0, 1].
    :type dropout_rate: float
    :param use_bias: Whether to use bias in linear layers.
    :type use_bias: bool
    :param initializer_range: Stddev for TruncatedNormal weight initialization.
    :type initializer_range: float
    :param kwargs: Additional keyword arguments for ``keras.Model``.

    :ivar embeddings: Token embedding layer.
    :ivar blocks: List of ``Gemma3TransformerBlock`` layers.
    :ivar final_norm: Final RMSNorm layer before output projection.
    :ivar lm_head: Language modeling head (Dense layer).

    Input shape:
        2D tensor of shape ``(batch_size, sequence_length)`` of token IDs.

    Output shape:
        3D tensor of shape ``(batch_size, sequence_length, vocab_size)`` of
        logits.
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

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer from ``input_shape``.

        Without this method Gemma3 inherits ``Layer.build``, which marks the
        model built while every sub-layer is still unbuilt -- Keras warns about
        exactly that at ``layers/layer.py:393``. The shared helper traces
        ``call()`` on symbolic inputs, so what gets built cannot drift from what
        gets called.

        :param input_shape: Shape (or nest of shapes) of the input to ``call``.
        """
        if self.built:
            return
        materialize_sublayers(self, input_shape)
        super().build(input_shape)

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
    classifier_dropout_rate: Optional[float] = None,
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
    # causal masking makes 'cls' pool a position that attended only to itself. See decisions.md.
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
        classifier_dropout_rate
        if classifier_dropout_rate is not None
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

    task_keys = ["num_labels", "pooling_strategy", "classifier_dropout_rate"]
    task_kwargs = {k: kwargs.pop(k) for k in task_keys if k in kwargs}
    # The rest of kwargs are model overrides.
    config.update(kwargs)

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