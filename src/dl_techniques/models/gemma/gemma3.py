"""
A complete implementation of the Gemma 3 architecture following Modern Keras 3
best practices for custom models, ensuring robustness and serializability.
"""


import keras
from keras import initializers, layers, ops
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms import create_normalization_layer

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
    pooling_strategy: str = "cls",
    classifier_dropout: Optional[float] = None,
) -> keras.Model:
    """Creates a Gemma3 model for sequence classification tasks."""
    if num_labels <= 0:
        raise ValueError(f"num_labels must be positive, got {num_labels}")
    if pooling_strategy not in ["cls", "mean"]:
        raise ValueError(
            f"pooling_strategy must be 'cls' or 'mean', got "
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

    if pooling_strategy == "cls":
        pooled_output = base_output[:, 0]
    else:  # "mean"
        mask = ops.expand_dims(ops.cast(attention_mask, base_output.dtype), axis=-1)
        summed = ops.sum(base_output * mask, axis=1)
        count = ops.maximum(
            ops.sum(ops.cast(attention_mask, "float32"), axis=1, keepdims=True),
            1.0,
        )
        pooled_output = summed / count

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