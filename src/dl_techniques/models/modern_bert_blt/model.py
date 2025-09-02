"""
ModernBertBLT: A Modern BERT with Byte Latent Transformer Features

This module implements a state-of-the-art byte-level bidirectional encoder.
It fuses the modern architectural improvements of ModernBERT with the robust
byte-level processing and hash n-gram embeddings from BertBlt.

Key Features:
=============

1.  **Byte-Level Processing**: Operates directly on UTF-8 bytes, providing
    language-agnostic capabilities and enhanced robustness to noisy text.

2.  **Hash N-Gram Embeddings**: Captures local contextual byte patterns
    (e.g., n-grams) using hashing to enrich representations without a large
    subword vocabulary.

3.  **Rotary Position Embeddings (RoPE)**: Replaces absolute position
    embeddings with a more dynamic and flexible relative positioning mechanism,
    improving performance on long sequences.

4.  **Modern Transformer Blocks**: Utilizes advanced components like GeGLU
    activations in the feed-forward network and a pre-normalization architecture.

5.  **Mixed Global/Local Attention**: Efficiently processes long sequences by
    alternating between full global attention and a sliding-window local attention.

Architecture Overview:
=====================

Input Bytes → [Byte Embeddings + Hash N-gram Embeddings] → [LayerNorm + Dropout] →
[Stack of ModernBertEncoderLayers (RoPE, GeGLU, Pre-LN)] →
[Final LayerNorm] → [Optional Pooling]
"""

import keras
from keras import initializers, layers
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .components import ModernBertBltEmbeddings
from ..modern_bert.components import ModernBertEncoderLayer

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ModernBertBLT(keras.Model):
    """
    A Modern Bidirectional Encoder with Byte Latent Transformer (BLT) features.

    This model integrates RoPE, GeGLU FFNs, and mixed attention from ModernBERT
    with the byte-level tokenization and hash n-gram embeddings of BertBlt.

    Args:
        add_pooling_layer: Whether to add a pooling layer. Defaults to `True`.
        vocab_size: Size of the byte vocabulary. Defaults to 260.
        hidden_size: Dimensionality of the encoder layers. Defaults to 768.
        num_layers: Number of hidden layers. Defaults to 12.
        num_heads: Number of attention heads. Defaults to 12.
        intermediate_size: Dimensionality of the feed-forward layer. Defaults to 3072.
        hidden_act: The activation function in the encoder. Defaults to "gelu".
        hidden_dropout_prob: Dropout probability for hidden layers. Defaults to 0.1.
        attention_probs_dropout_prob: Dropout probability for attention scores. Defaults to 0.1.
        initializer_range: Stddev for weight initialization. Defaults to 0.02.
        layer_norm_eps: Epsilon for layer normalization. Defaults to 1e-12.
        use_hash_embeddings: Whether to use hash n-gram embeddings. Defaults to True.
        hash_vocab_size: Number of hash buckets for n-grams. Defaults to 500000.
        ngram_sizes: List of n-gram sizes to use.
        hash_embedding_dim: Dimensionality of hash embeddings.
        use_bias: Whether to use bias in layers. Defaults to False.
        rope_theta_local: RoPE theta for local attention. Defaults to 10000.0.
        rope_theta_global: RoPE theta for global attention. Defaults to 160000.0.
        max_seq_len: Maximum sequence length for RoPE. Defaults to 8192.
        global_attention_interval: Interval for global attention layers. Defaults to 3.
        local_attention_window_size: Window size for local attention. Defaults to 128.
        **kwargs: Additional arguments for `keras.Model`.
    """

    def __init__(
            self,
            add_pooling_layer: bool = True,
            # Core parameters
            vocab_size: int = 260,
            hidden_size: int = 768,
            num_layers: int = 12,
            num_heads: int = 12,
            intermediate_size: int = 3072,
            hidden_act: str = "gelu",
            hidden_dropout_prob: float = 0.1,
            attention_probs_dropout_prob: float = 0.1,
            initializer_range: float = 0.02,
            layer_norm_eps: float = 1e-12,
            # BLT-specific parameters
            use_hash_embeddings: bool = True,
            hash_vocab_size: int = 500000,
            ngram_sizes: Optional[List[int]] = None,
            hash_embedding_dim: Optional[int] = None,
            # ModernBERT architectural parameters
            use_bias: bool = False,
            rope_theta_local: float = 10000.0,
            rope_theta_global: float = 160000.0,
            max_seq_len: int = 8192,
            global_attention_interval: int = 3,
            local_attention_window_size: int = 128,
            **kwargs,
    ):
        super().__init__(**kwargs)

        # --- Store configuration ---
        self.add_pooling_layer = add_pooling_layer
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_hash_embeddings = use_hash_embeddings
        self.hash_vocab_size = hash_vocab_size
        self.ngram_sizes = (
            ngram_sizes if ngram_sizes is not None else [3, 4, 5, 6, 7, 8]
        )
        self.hash_embedding_dim = (
            hash_embedding_dim if hash_embedding_dim is not None else hidden_size
        )
        self.use_bias = use_bias
        self.rope_theta_local = rope_theta_local
        self.rope_theta_global = rope_theta_global
        self.max_seq_len = max_seq_len
        self.global_attention_interval = global_attention_interval
        self.local_attention_window_size = local_attention_window_size

        # --- Layer Definition ---

        # BLT Embeddings (without absolute position embeddings)
        self.embeddings = ModernBertBltEmbeddings(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            # -----------------------------------------------------------------
            # FIX: Pass a valid, positive integer to max_position_embeddings.
            # Using max_seq_len makes the layer validly constructible.
            # -----------------------------------------------------------------
            max_position_embeddings=self.max_seq_len,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            hidden_dropout_prob=self.hidden_dropout_prob,
            use_hash_embeddings=self.use_hash_embeddings,
            hash_vocab_size=self.hash_vocab_size,
            ngram_sizes=self.ngram_sizes,
            hash_embedding_dim=self.hash_embedding_dim,
            normalization_type="layer_norm", # Standard for this arch
            name="embeddings",
        )

        # ModernBERT Encoder Stack
        encoder_args = {
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_act": self.hidden_act,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "use_bias": self.use_bias,
            "rope_theta_local": self.rope_theta_local,
            "rope_theta_global": self.rope_theta_global,
            "max_seq_len": self.max_seq_len,
            "local_attention_window_size": self.local_attention_window_size,
        }

        self.encoder_layers = [
            ModernBertEncoderLayer(
                is_global=((i + 1) % self.global_attention_interval == 0),
                name=f"encoder_layer_{i}",
                **encoder_args,
            )
            for i in range(self.num_layers)
        ]

        self.final_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_eps, center=self.use_bias, name="final_layer_norm"
        )

        # Pooler for classification tasks
        self.pooler = None
        if self.add_pooling_layer:
            self.pooler = layers.Dense(
                units=self.hidden_size,
                activation="tanh",
                use_bias=self.use_bias,
                kernel_initializer=initializers.TruncatedNormal(
                    stddev=self.initializer_range
                ),
                name="pooler",
            )

        logger.info(
            f"Initialized ModernBertBLT model with {self.num_layers} layers."
        )

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None,
            return_dict: bool = False,
    ) -> Union[
        keras.KerasTensor,
        Tuple[keras.KerasTensor, keras.KerasTensor],
        Dict[str, keras.KerasTensor],
    ]:
        """Forward pass of the ModernBertBLT model."""
        if isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", attention_mask)
        else:
            input_ids = inputs

        # Note: We pass position_ids=None to prevent absolute pos embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=None, training=training
        )

        hidden_states = embedding_output
        for layer in self.encoder_layers:
            hidden_states = layer(
                hidden_states, attention_mask=attention_mask, training=training
            )

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

    def encode_text(
            self,
            text: str,
            max_length: Optional[int] = None,
            add_special_tokens: bool = True
    ) -> keras.KerasTensor:
        """Encodes a string of text into byte token IDs."""
        return self.embeddings.encode_text(text, max_length, add_special_tokens)

    def decode_tokens(self, token_ids: keras.KerasTensor) -> str:
        """Decodes a tensor of byte token IDs back into a string."""
        return self.embeddings.decode_tokens(token_ids)

    def get_config(self) -> Dict[str, Any]:
        """Serializes the model's configuration for saving."""
        config = super().get_config()
        config.update({
            "add_pooling_layer": self.add_pooling_layer,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_act": self.hidden_act,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "use_hash_embeddings": self.use_hash_embeddings,
            "hash_vocab_size": self.hash_vocab_size,
            "ngram_sizes": self.ngram_sizes,
            "hash_embedding_dim": self.hash_embedding_dim,
            "use_bias": self.use_bias,
            "rope_theta_local": self.rope_theta_local,
            "rope_theta_global": self.rope_theta_global,
            "max_seq_len": self.max_seq_len,
            "global_attention_interval": self.global_attention_interval,
            "local_attention_window_size": self.local_attention_window_size,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ModernBertBLT":
        """Creates a model from its configuration."""
        return cls(**config)


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def create_modern_bert_blt_base_config() -> Dict[str, Any]:
    """Creates the configuration for a ModernBertBLT-base model."""
    logger.info("Creating ModernBertBLT-base configuration dictionary")
    return {
        "vocab_size": 260,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "intermediate_size": 3072,
        "use_hash_embeddings": True,
        "hash_vocab_size": 500000,
        "ngram_sizes": [3, 4, 5, 6, 7, 8],
        "use_bias": False,
        "global_attention_interval": 3,
        "local_attention_window_size": 128,
        "max_seq_len": 8192,
    }


def create_modern_bert_blt_for_classification(
        config_dict: Dict[str, Any],
        num_labels: int,
        classifier_dropout: Optional[float] = None,
) -> keras.Model:
    """Builds a Keras Model for sequence classification using ModernBertBLT."""
    logger.info(
        f"Creating ModernBertBLT classification model with {num_labels} labels"
    )
    model_instance = ModernBertBLT(
        **config_dict, add_pooling_layer=True, name="modern_bert_blt"
    )
    input_ids = keras.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = keras.Input(
        shape=(None,), dtype="int32", name="attention_mask"
    )
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

    bert_outputs = model_instance(inputs, return_dict=True)
    pooled_output = bert_outputs["pooler_output"]

    dropout_rate = (
        classifier_dropout
        if classifier_dropout is not None
        else config_dict.get("hidden_dropout_prob", 0.1)
    )
    if dropout_rate > 0.0:
        pooled_output = keras.layers.Dropout(
            dropout_rate, name="classifier_dropout"
        )(pooled_output)

    initializer_range = config_dict.get("initializer_range", 0.02)
    logits = keras.layers.Dense(
        units=num_labels,
        kernel_initializer=initializers.TruncatedNormal(
            stddev=initializer_range
        ),
        name="classifier",
    )(pooled_output)

    model = keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=logits,
        name="modern_bert_blt_for_classification",
    )
    logger.info(
        f"Created ModernBertBLT classification model with {model.count_params():,} "
        "parameters"
    )
    return model