import keras
from keras import layers, ops
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformer import TransformerLayer
from dl_techniques.layers.nlp_heads import create_nlp_head, NLPTaskConfig

from .components import ModernBertEmbeddings

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ModernBERT(keras.Model):
    """
    ModernBERT (A Modern Bidirectional Encoder) foundation model.

    This model refactors the original BERT architecture to include modern
    techniques. It leverages a generic, configurable TransformerLayer to
    implement its encoder stack, allowing for easy swapping of components like
    attention and FFN types.

    **Intent**:
    To provide a powerful, configurable transformer encoder that serves as the
    backbone for various NLP tasks. Its responsibility ends at producing high-quality
    contextual representations.

    **Architectural Contract**:
    ```
    Input (input_ids, token_type_ids, attention_mask)
           ↓
    ModernBertEmbeddings
           ↓
    TransformerLayer x num_layers (configured for Pre-LN, GeGLU, and mixed attention)
           ↓
    Final Layer Normalization
           ↓
    Output Dictionary {
        "last_hidden_state": [batch, seq_len, hidden_size],
        "attention_mask": [batch, seq_len]
    }
    ```

    Args:
        vocab_size: Integer, size of the vocabulary. Defaults to 50368.
        hidden_size: Integer, dimensionality of the encoder layers. Defaults to 768.
        num_layers: Integer, number of hidden layers. Defaults to 22.
        num_heads: Integer, number of attention heads. Defaults to 12.
        intermediate_size: Integer, dimensionality of the feed-forward layer.
            Defaults to 1152.
        hidden_act: String, the non-linear activation function. Defaults to "gelu".
        hidden_dropout_prob: Float, dropout probability for embeddings and encoder.
            Defaults to 0.1.
        attention_probs_dropout_prob: Float, dropout ratio for attention.
            Defaults to 0.1.
        type_vocab_size: Integer, vocabulary size of `token_type_ids`. Defaults to 2.
        initializer_range: Float, stddev for weight initialization. Defaults to 0.02.
        layer_norm_eps: Float, epsilon for layer normalization. Defaults to 1e-12.
        use_bias: Boolean, whether to use bias vectors. Defaults to False.
        global_attention_interval: Integer, interval for global attention layers.
            Defaults to 3.
        local_attention_window_size: Integer, window size for local attention.
            Defaults to 128.
        **kwargs: Additional arguments for the `keras.Model` base class.

    """

    MODEL_VARIANTS = {
        "base": {
            "vocab_size": 50368,
            "hidden_size": 768,
            "num_layers": 22,
            "num_heads": 12,
            "intermediate_size": 1152,
            "use_bias": False,
            "global_attention_interval": 3,
            "local_attention_window_size": 128,
            "description": "ModernBERT-Base: Efficient base-sized model",
        },
        "large": {
            "vocab_size": 50368,
            "hidden_size": 1024,
            "num_layers": 28,
            "num_heads": 16,
            "intermediate_size": 2624,
            "use_bias": False,
            "global_attention_interval": 3,
            "local_attention_window_size": 128,
            "description": "ModernBERT-Large: High-performance large model",
        },
    }

    def __init__(
            self,
            vocab_size: int = 50368,
            hidden_size: int = 768,
            num_layers: int = 22,
            num_heads: int = 12,
            intermediate_size: int = 1152,
            hidden_act: str = "gelu",
            hidden_dropout_prob: float = 0.1,
            attention_probs_dropout_prob: float = 0.1,
            type_vocab_size: int = 2,
            initializer_range: float = 0.02,
            layer_norm_eps: float = 1e-12,
            use_bias: bool = False,
            global_attention_interval: int = 3,
            local_attention_window_size: int = 128,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_bias = use_bias
        self.global_attention_interval = global_attention_interval
        self.local_attention_window_size = local_attention_window_size

        self.embeddings = ModernBertEmbeddings(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            hidden_dropout_prob=self.hidden_dropout_prob,
            use_bias=self.use_bias,
            name="embeddings",
        )

        self.encoder_layers = []
        for i in range(self.num_layers):
            is_global = (i + 1) % self.global_attention_interval == 0

            if is_global:
                attention_type = "multi_head"
                attention_args = {}
            else:
                attention_type = "window"
                attention_args = {"window_size": self.local_attention_window_size}

            layer = TransformerLayer(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                attention_type=attention_type,
                attention_args=attention_args,
                normalization_position='pre',
                ffn_type='geglu',
                ffn_args={'activation': self.hidden_act},
                dropout_rate=self.hidden_dropout_prob,
                attention_dropout_rate=self.attention_probs_dropout_prob,
                use_bias=self.use_bias,
                name=f"encoder_layer_{i}",
            )
            self.encoder_layers.append(layer)

        self.final_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_eps,
            center=self.use_bias,
            name="final_layer_norm"
        )

    def call(
            self,
            inputs,
            attention_mask=None,
            token_type_ids=None,
            training=None,
    ) -> Dict[str, keras.KerasTensor]:
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
        else:
            input_ids = inputs
        if input_ids is None:
            raise ValueError("input_ids must be provided")

        embedding_output = self.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids, training=training
        )

        hidden_states = embedding_output
        for layer in self.encoder_layers:
            hidden_states = layer(
                hidden_states, attention_mask=attention_mask, training=training
            )

        sequence_output = self.final_norm(hidden_states, training=training)

        return {
            "last_hidden_state": sequence_output,
            "attention_mask": attention_mask,
        }

    @classmethod
    def from_variant(cls, variant: str, **kwargs: Any) -> "ModernBERT":
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        description = config.pop("description", "")
        logger.info(f"Creating ModernBERT-{variant.upper()} model")
        logger.info(f"Configuration: {description}")
        config.update(kwargs)
        return cls(**config)

    def get_config(self) -> Dict[str, Any]:
        """Returns the model's configuration for serialization."""
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_act": self.hidden_act,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "type_vocab_size": self.type_vocab_size,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "use_bias": self.use_bias,
            "global_attention_interval": self.global_attention_interval,
            "local_attention_window_size": self.local_attention_window_size,
        })
        return config

# ---------------------------------------------------------------------

def create_modern_bert_with_head(
        bert_variant: str,
        task_config: NLPTaskConfig,
        bert_config_overrides: Optional[Dict[str, Any]] = None,
        head_config_overrides: Optional[Dict[str, Any]] = None,
) -> keras.Model:
    bert_config_overrides = bert_config_overrides or {}
    head_config_overrides = head_config_overrides or {}

    logger.info(f"Creating ModernBERT-{bert_variant} with a '{task_config.name}' head.")

    bert_encoder = ModernBERT.from_variant(bert_variant, **bert_config_overrides)
    task_head = create_nlp_head(
        task_config=task_config,
        input_dim=bert_encoder.hidden_size,
        **head_config_overrides,
    )

    inputs = {
        "input_ids": keras.Input(shape=(None,), dtype="int32", name="input_ids"),
        "attention_mask": keras.Input(shape=(None,), dtype="int32", name="attention_mask"),
        "token_type_ids": keras.Input(shape=(None,), dtype="int32", name="token_type_ids"),
    }

    encoder_outputs = bert_encoder(inputs)

    # --- FIX: Cast attention_mask to float32 to match the head's expected dtype ---
    # This patches the dtype mismatch bug within the QuestionAnsweringHead layer.
    attention_mask_float = ops.cast(
        encoder_outputs["attention_mask"], dtype=encoder_outputs["last_hidden_state"].dtype
    )

    head_inputs = {
        "hidden_states": encoder_outputs["last_hidden_state"],
        "attention_mask": attention_mask_float,
    }
    task_outputs = task_head(head_inputs)

    model = keras.Model(
        inputs=inputs,
        outputs=task_outputs,
        name=f"modern_bert_{bert_variant}_with_{task_config.name}_head",
    )

    logger.info(f"Successfully created model with {model.count_params():,} parameters.")
    return model

# ---------------------------------------------------------------------