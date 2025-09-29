"""
ModernBertBLT: A Modern BERT with Byte Latent Transformer Features

This module implements a state-of-the-art byte-level bidirectional encoder.
It fuses the modern architectural improvements of ModernBERT with the robust
byte-level processing and hash n-gram embeddings from BertBlt.

This version has been refactored to align with the system's preferred
decoupled architecture, where the core model acts as a pure foundation
encoder.

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

    ## NOTE: The current implementation falls back to absolute position embeddings.
    ## RoPE is not applied in the TransformerLayer.

4.  **Modern Transformer Blocks**: Utilizes advanced components like GeGLU
    activations in the feed-forward network and a pre-normalization architecture,
    implemented via a generic and configurable `TransformerLayer`.

5.  **Mixed Global/Local Attention**: Efficiently processes long sequences by
    alternating between full global attention and a sliding-window local attention.

Architecture Overview:
=====================

Input Bytes → [Byte Embeddings + Hash N-gram Embeddings + Absolute Positional Embeddings] →
[LayerNorm + Dropout] →
[Stack of TransformerLayers (GeGLU, Pre-LN, Mixed Attention)] →
[Final LayerNorm] → Output Dictionary
"""

import keras
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformer import TransformerLayer
from dl_techniques.layers.nlp_heads import create_nlp_head, NLPTaskConfig
from .components import ModernBertBltEmbeddings

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ModernBertBLT(keras.Model):
    """
    A Modern Bidirectional Encoder with Byte Latent Transformer (BLT) features.

    This model integrates RoPE, GeGLU FFNs, and mixed attention with byte-level
    tokenization and hash n-gram embeddings. It is a pure foundation model,
    designed to be combined with task-specific heads.

    Args:
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
        rope_theta: RoPE theta value. Defaults to 10000.0.
        max_seq_len: Maximum sequence length for RoPE. Defaults to 8192.
        global_attention_interval: Interval for global attention layers. Defaults to 3.
        local_attention_window_size: Window size for local attention. Defaults to 128.
        **kwargs: Additional arguments for `keras.Model`.
    """

    MODEL_VARIANTS = {
        "base": {
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
            "description": "ModernBertBLT-Base: Byte-level model with RoPE",
        },
        "large": {
            "vocab_size": 260,
            "hidden_size": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "intermediate_size": 4096,
            "use_hash_embeddings": True,
            "hash_vocab_size": 500000,
            "ngram_sizes": [3, 4, 5, 6, 7, 8],
            "use_bias": False,
            "global_attention_interval": 4,
            # FIX: Reduce window size from 256 to 128 to prevent OOM on init.
            "local_attention_window_size": 128,
            "max_seq_len": 8192,
            "description": "ModernBertBLT-Large: High-performance byte model",
        },
    }

    def __init__(
            self,
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
            rope_theta: float = 10000.0,
            max_seq_len: int = 8192,
            global_attention_interval: int = 3,
            local_attention_window_size: int = 128,
            **kwargs,
    ):
        super().__init__(**kwargs)

        # --- Store configuration ---
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
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len
        self.global_attention_interval = global_attention_interval
        self.local_attention_window_size = local_attention_window_size

        # --- Layer Definition ---

        # ## FIX: Use absolute position embeddings as RoPE is not yet integrated
        self.embeddings = ModernBertBltEmbeddings(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            max_position_embeddings=self.max_seq_len, # For valid construction
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            hidden_dropout_prob=self.hidden_dropout_prob,
            use_hash_embeddings=self.use_hash_embeddings,
            hash_vocab_size=self.hash_vocab_size,
            ngram_sizes=self.ngram_sizes,
            hash_embedding_dim=self.hash_embedding_dim,
            normalization_type="layer_norm",
            name="embeddings",
        )

        # ModernBERT Encoder Stack using generic TransformerLayer
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

        self.final_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps,
            center=self.use_bias,
            name="final_layer_norm"
        )

        logger.info(
            f"Initialized ModernBertBLT model with {self.num_layers} layers."
        )

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass of the ModernBertBLT model."""
        if isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", attention_mask)
        else:
            input_ids = inputs

        # The ModernBertBltEmbeddings layer handles position_ids internally
        # when they are not provided, creating absolute position embeddings.
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=None, training=training
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
    def from_variant(cls, variant: str, **kwargs: Any) -> "ModernBertBLT":
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available: {list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        description = config.pop("description", "")
        logger.info(f"Creating ModernBertBLT-{variant.upper()} model")
        logger.info(f"Configuration: {description}")
        config.update(kwargs)
        return cls(**config)

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
            "rope_theta": self.rope_theta,
            "max_seq_len": self.max_seq_len,
            "global_attention_interval": self.global_attention_interval,
            "local_attention_window_size": self.local_attention_window_size,
        })
        return config


# ---------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------

def create_modern_bert_blt_with_head(
        bert_variant: str,
        task_config: NLPTaskConfig,
        bert_config_overrides: Optional[Dict[str, Any]] = None,
        head_config_overrides: Optional[Dict[str, Any]] = None,
) -> keras.Model:
    """
    Factory function to create a complete ModernBertBLT model with a task-specific head.
    """
    bert_config_overrides = bert_config_overrides or {}
    head_config_overrides = head_config_overrides or {}

    logger.info(f"Creating ModernBertBLT-{bert_variant} with a '{task_config.name}' head.")

    # 1. Create the foundational BERT model
    bert_encoder = ModernBertBLT.from_variant(bert_variant, **bert_config_overrides)

    # 2. Create the task head
    task_head = create_nlp_head(
        task_config=task_config,
        input_dim=bert_encoder.hidden_size,
        **head_config_overrides,
    )

    # 3. Define inputs and build the end-to-end model
    inputs = {
        "input_ids": keras.Input(shape=(None,), dtype="int32", name="input_ids"),
        "attention_mask": keras.Input(shape=(None,), dtype="int32", name="attention_mask"),
    }

    # Get hidden states from the encoder
    encoder_outputs = bert_encoder(inputs)

    # Pass encoder outputs to the task head
    head_inputs = {
        "hidden_states": encoder_outputs["last_hidden_state"],
        "attention_mask": encoder_outputs["attention_mask"],
    }
    task_outputs = task_head(head_inputs)

    # Create the final model
    model = keras.Model(
        inputs=inputs,
        outputs=task_outputs,
        name=f"modern_bert_blt_{bert_variant}_with_{task_config.name}_head"
    )

    logger.info(f"Successfully created model with {model.count_params():,} parameters.")
    return model