import keras
from keras import initializers
from typing import Optional, Any, Dict, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .components import ModernBertEmbeddings, ModernBertTransformerLayer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ModernBERT(keras.Model):
    """
    ModernBERT (A Modern Bidirectional Encoder) model implementation.

    This model integrates several modern transformer architecture features:
    - **Rotary Position Embeddings (RoPE)** for relative position encoding.
    - **Hybrid Global/Local Attention**: Alternates between global (full) and
      local (sliding window) attention layers to balance performance and efficiency.
    - **GeGLU Feed-Forward Networks**: Uses Gated Linear Units for the FFN block.
    - **Pre-Layer Normalization**: Applies normalization before sub-layer execution.

    **Intent**: Provide a complete, serializable, and high-performance BERT-like
    model that incorporates recent architectural improvements.

    **Architecture**:
    ```
    Input(input_ids, token_type_ids)
           │
           ▼
    ModernBertEmbeddings
           │
           ▼
    ModernBertTransformerLayer₁ (Local Attention)
           │
           ▼
    ModernBertTransformerLayer₂ (Local Attention)
           │
           ▼
    ModernBertTransformerLayer₃ (Global Attention)
           │
           ▼
          ... (repeats)
           │
           ▼
    Final LayerNorm
           │
           ▼
    Sequence Output ───> (Optional) Pooler ───> Pooled Output
    ```
    """

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
            pad_token_id: int = 0,
            classifier_dropout: Optional[float] = None,
            use_bias: bool = False,
            rope_theta_local: float = 10000.0,
            rope_theta_global: float = 160000.0,
            global_attention_interval: int = 3,
            local_attention_window_size: int = 128,
            add_pooling_layer: bool = True,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        # Store all configuration parameters
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
        self.pad_token_id = pad_token_id
        self.classifier_dropout = classifier_dropout
        self.use_bias = use_bias
        self.rope_theta_local = rope_theta_local
        self.rope_theta_global = rope_theta_global
        self.global_attention_interval = global_attention_interval
        self.local_attention_window_size = local_attention_window_size
        self.add_pooling_layer = add_pooling_layer

        # CREATE all sub-layers in __init__
        self.embeddings = ModernBertEmbeddings(
            vocab_size=self.vocab_size, hidden_size=self.hidden_size, type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range, layer_norm_eps=self.layer_norm_eps,
            hidden_dropout_prob=self.hidden_dropout_prob, use_bias=self.use_bias, name="embeddings"
        )
        self.encoder_layers: List[ModernBertTransformerLayer] = []
        for i in range(self.num_layers):
            is_global = (i + 1) % self.global_attention_interval == 0
            layer = ModernBertTransformerLayer(
                hidden_size=self.hidden_size, num_heads=self.num_heads, intermediate_size=self.intermediate_size,
                hidden_act=self.hidden_act, hidden_dropout_prob=self.hidden_dropout_prob,
                attention_probs_dropout_prob=self.attention_probs_dropout_prob, use_bias=self.use_bias,
                initializer_range=self.initializer_range, layer_norm_eps=self.layer_norm_eps,
                is_global=is_global, rope_theta_local=self.rope_theta_local,
                rope_theta_global=self.rope_theta_global,
                local_attention_window_size=self.local_attention_window_size,
                name=f"encoder_layer_{i}"
            )
            self.encoder_layers.append(layer)

        self.final_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps, center=self.use_bias, name="final_layer_norm"
        )

        self.pooler = None
        if self.add_pooling_layer:
            self.pooler = keras.layers.Dense(
                units=self.hidden_size, activation="tanh", use_bias=self.use_bias,
                kernel_initializer=initializers.TruncatedNormal(stddev=self.initializer_range), name="pooler"
            )

    def call(self, inputs, attention_mask=None, token_type_ids=None, training=None, return_dict=False):
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", attention_mask)
            token_type_ids = inputs.get("token_type_ids", token_type_ids)
        else:
            input_ids = inputs

        if input_ids is None:
            raise ValueError("input_ids must be provided")

        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids, training=training)
        hidden_states = embedding_output
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask, training=training)

        sequence_output = self.final_norm(hidden_states, training=training)
        pooled_output = None
        if self.pooler is not None:
            first_token_tensor = sequence_output[:, 0]
            pooled_output = self.pooler(first_token_tensor)

        if return_dict:
            outputs = {"last_hidden_state": sequence_output}
            if pooled_output is not None:
                outputs["pooler_output"] = pooled_output
            return outputs
        else:
            if pooled_output is not None:
                return sequence_output, pooled_output
            return sequence_output

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size, 'hidden_size': self.hidden_size, 'num_layers': self.num_layers,
            'num_heads': self.num_heads, 'intermediate_size': self.intermediate_size, 'hidden_act': self.hidden_act,
            'hidden_dropout_prob': self.hidden_dropout_prob,
            'attention_probs_dropout_prob': self.attention_probs_dropout_prob,
            'type_vocab_size': self.type_vocab_size, 'initializer_range': self.initializer_range,
            'layer_norm_eps': self.layer_norm_eps, 'pad_token_id': self.pad_token_id,
            'classifier_dropout': self.classifier_dropout, 'use_bias': self.use_bias,
            'rope_theta_local': self.rope_theta_local, 'rope_theta_global': self.rope_theta_global,
            'global_attention_interval': self.global_attention_interval,
            'local_attention_window_size': self.local_attention_window_size,
            'add_pooling_layer': self.add_pooling_layer,
        })
        return config

# ---------------------------------------------------------------------

def create_modern_bert_base() -> Dict[str, Any]:
    """Return configuration dictionary for ModernBERT-base model."""
    logger.info("Creating ModernBERT-base configuration")
    return {
        "vocab_size": 50368, "hidden_size": 768, "num_layers": 22, "num_heads": 12,
        "intermediate_size": 1152, "hidden_act": "gelu", "use_bias": False,
        "global_attention_interval": 3, "local_attention_window_size": 128
    }

# ---------------------------------------------------------------------

def create_modern_bert_large() -> Dict[str, Any]:
    """Return configuration dictionary for ModernBERT-large model."""
    logger.info("Creating ModernBERT-large configuration")
    return {
        "vocab_size": 50368, "hidden_size": 1024, "num_layers": 28, "num_heads": 16,
        "intermediate_size": 2624, "hidden_act": "gelu", "use_bias": False,
        "global_attention_interval": 3, "local_attention_window_size": 128
    }

# ---------------------------------------------------------------------

def create_modern_bert_for_classification(
        config: Dict[str, Any],
        num_labels: int,
        classifier_dropout: Optional[float] = None
) -> keras.Model:
    """Create a ModernBERT model with a classification head."""
    logger.info(f"Creating ModernBERT classification model with {num_labels} labels")
    input_ids = keras.layers.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = keras.layers.Input(shape=(None,), dtype="int32", name="attention_mask")
    token_type_ids = keras.layers.Input(shape=(None,), dtype="int32", name="token_type_ids")

    modern_bert = ModernBERT(**config, add_pooling_layer=True, name="modern_bert")
    bert_outputs = modern_bert(
        inputs={"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids},
        return_dict=True
    )
    pooled_output = bert_outputs["pooler_output"]

    final_dropout = classifier_dropout
    if final_dropout is None:
        final_dropout = config.get("classifier_dropout") or config.get("hidden_dropout_prob", 0.1)

    if final_dropout > 0.0:
        pooled_output = keras.layers.Dropout(final_dropout, name="classifier_dropout")(pooled_output)

    logits = keras.layers.Dense(
        units=num_labels,
        kernel_initializer=initializers.TruncatedNormal(stddev=config.get("initializer_range", 0.02)),
        name="classifier"
    )(pooled_output)

    model = keras.Model(
        inputs=[input_ids, attention_mask, token_type_ids], outputs=logits, name="modern_bert_for_classification"
    )
    logger.info(f"Created ModernBERT classification model with {model.count_params()} parameters")
    return model

# ---------------------------------------------------------------------

def create_modern_bert_for_sequence_output(
        config: Dict[str, Any]
) -> keras.Model:
    """Create a ModernBERT model for sequence-level tasks."""
    logger.info("Creating ModernBERT model for sequence output")
    input_ids = keras.layers.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = keras.layers.Input(shape=(None,), dtype="int32", name="attention_mask")
    token_type_ids = keras.layers.Input(shape=(None,), dtype="int32", name="token_type_ids")

    modern_bert = ModernBERT(**config, add_pooling_layer=False, name="modern_bert")
    sequence_output = modern_bert(
        inputs={"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
    )
    model = keras.Model(
        inputs=[input_ids, attention_mask, token_type_ids],
        outputs=sequence_output, name="modern_bert_for_sequence_output"
    )
    logger.info(f"Created ModernBERT sequence model with {model.count_params()} parameters")
    return model

# ---------------------------------------------------------------------
