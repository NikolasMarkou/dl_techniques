import keras
from keras import initializers, layers
from typing import Optional, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .components import ModernBertEmbeddings, ModernBertEncoderLayer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ModernBERT(keras.Model):
    """
    ModernBERT (A Modern Bidirectional Encoder) model implementation.

    This model refactors the original BERT architecture to include modern
    techniques like Rotary Position Embeddings (RoPE), GeGLU activations,
    and a mixture of global and local attention.
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
        use_bias: bool = False,
        rope_theta_local: float = 10000.0,
        rope_theta_global: float = 160000.0,
        max_seq_len: int = 8192,
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
        self.use_bias = use_bias
        self.rope_theta_local = rope_theta_local
        self.rope_theta_global = rope_theta_global
        self.max_seq_len = max_seq_len
        self.global_attention_interval = global_attention_interval
        self.local_attention_window_size = local_attention_window_size
        self.add_pooling_layer = add_pooling_layer

        # Create sub-layers
        self.embeddings = ModernBertEmbeddings(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            hidden_dropout_prob=self.hidden_dropout_prob,
            use_bias=self.use_bias,
            name="embeddings"
        )

        # Group encoder layer args into a dict for cleaner instantiation
        encoder_args = {
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'intermediate_size': self.intermediate_size,
            'hidden_act': self.hidden_act,
            'hidden_dropout_prob': self.hidden_dropout_prob,
            'attention_probs_dropout_prob': self.attention_probs_dropout_prob,
            'initializer_range': self.initializer_range,
            'layer_norm_eps': self.layer_norm_eps,
            'use_bias': self.use_bias,
            'rope_theta_local': self.rope_theta_local,
            'rope_theta_global': self.rope_theta_global,
            'max_seq_len': self.max_seq_len,
            'local_attention_window_size': self.local_attention_window_size,
        }

        self.encoder_layers = []
        for i in range(self.num_layers):
            is_global = (i + 1) % self.global_attention_interval == 0
            layer = ModernBertEncoderLayer(
                is_global=is_global,
                name=f"encoder_layer_{i}",
                **encoder_args
            )
            self.encoder_layers.append(layer)

        self.final_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_eps,
            center=self.use_bias,
            name="final_layer_norm"
        )

        self.pooler = None
        if self.add_pooling_layer:
            self.pooler = layers.Dense(
                units=self.hidden_size,
                activation="tanh",
                use_bias=self.use_bias,
                kernel_initializer=initializers.TruncatedNormal(stddev=self.initializer_range),
                name="pooler"
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
            pooled_output = self.pooler(first_token_tensor, training=training)

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
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'intermediate_size': self.intermediate_size,
            'hidden_act': self.hidden_act,
            'hidden_dropout_prob': self.hidden_dropout_prob,
            'attention_probs_dropout_prob': self.attention_probs_dropout_prob,
            'type_vocab_size': self.type_vocab_size,
            'initializer_range': self.initializer_range,
            'layer_norm_eps': self.layer_norm_eps,
            'use_bias': self.use_bias,
            'rope_theta_local': self.rope_theta_local,
            'rope_theta_global': self.rope_theta_global,
            'max_seq_len': self.max_seq_len,
            'global_attention_interval': self.global_attention_interval,
            'local_attention_window_size': self.local_attention_window_size,
            'add_pooling_layer': self.add_pooling_layer,
        })
        return config


# ---------------------------------------------------------------------

def create_modern_bert_base() -> Dict[str, Any]:
    """
    Create configuration dictionary for ModernBERT-base model.
    Architecture: 22 layers, 768 hidden, 12 heads, 1152 intermediate
    """
    logger.info("Creating ModernBERT-base configuration dictionary")
    return {
        "vocab_size": 50368,
        "hidden_size": 768,
        "num_layers": 22,
        "num_heads": 12,
        "intermediate_size": 1152,
        "hidden_act": "gelu",
        "use_bias": False,
        "global_attention_interval": 3,
        "local_attention_window_size": 128,
        "max_seq_len": 8192,
    }


def create_modern_bert_large() -> Dict[str, Any]:
    """
    Create configuration dictionary for ModernBERT-large model.
    Architecture: 28 layers, 1024 hidden, 16 heads, 2624 intermediate
    """
    logger.info("Creating ModernBERT-large configuration dictionary")
    return {
        "vocab_size": 50368,
        "hidden_size": 1024,
        "num_layers": 28,
        "num_heads": 16,
        "intermediate_size": 2624,
        "hidden_act": "gelu",
        "use_bias": False,
        "global_attention_interval": 3,
        "local_attention_window_size": 128,
        "max_seq_len": 8192,
    }


def create_modern_bert_for_classification(
    config: Dict[str, Any],
    num_labels: int,
    classifier_dropout: Optional[float] = None
) -> keras.Model:
    """
    Create a ModernBERT model with a classification head.
    """
    logger.info(f"Creating ModernBERT classification model with {num_labels} labels")
    input_ids = layers.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = layers.Input(shape=(None,), dtype="int32", name="attention_mask")
    token_type_ids = layers.Input(shape=(None,), dtype="int32", name="token_type_ids")

    modern_bert = ModernBERT(add_pooling_layer=True, name="modern_bert", **config)

    bert_outputs = modern_bert(
        inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
        return_dict=True
    )
    pooled_output = bert_outputs["pooler_output"]

    dropout_rate = classifier_dropout if classifier_dropout is not None else config.get('hidden_dropout_prob', 0.1)
    if dropout_rate > 0.0:
        pooled_output = layers.Dropout(
            dropout_rate, name="classifier_dropout"
        )(pooled_output)

    logits = layers.Dense(
        units=num_labels,
        kernel_initializer=initializers.TruncatedNormal(
            stddev=config.get('initializer_range', 0.02)
        ),
        name="classifier"
    )(pooled_output)

    model = keras.Model(
        inputs=[input_ids, attention_mask, token_type_ids],
        outputs=logits,
        name="modern_bert_for_classification"
    )
    logger.info(f"Created ModernBERT classification model with {model.count_params()} parameters")
    return model


def create_modern_bert_for_sequence_output(
    config: Dict[str, Any]
) -> keras.Model:
    """
    Create a ModernBERT model for sequence-level tasks.
    """
    logger.info("Creating ModernBERT model for sequence output")
    input_ids = layers.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = layers.Input(shape=(None,), dtype="int32", name="attention_mask")
    token_type_ids = layers.Input(shape=(None,), dtype="int32", name="token_type_ids")

    modern_bert = ModernBERT(add_pooling_layer=False, name="modern_bert", **config)

    sequence_output = modern_bert(
        inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
    )

    model = keras.Model(
        inputs=[input_ids, attention_mask, token_type_ids],
        outputs=sequence_output,
        name="modern_bert_for_sequence_output"
    )
    logger.info(f"Created ModernBERT sequence model with {model.count_params()} parameters")
    return model

# ---------------------------------------------------------------------