"""
Qwen3 Model Implementation for dl_techniques Framework (Refactored)

This module provides a clean, refined Keras 3 implementation of the Qwen3 architecture,
refactored to follow the modern backbone-and-head design pattern. The Qwen3Model class
represents the core transformer backbone, while task-specific factory functions add the
appropriate heads for generation or classification.

This design enhances modularity, reusability, and consistency with other models in the
dl_techniques framework, such as Qwen3Next.

Based on the Qwen3 architecture from:
- Qwen3: Think Deeper, Act Faster
- https://hface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct
"""

import keras
from typing import Optional, Dict, Any, List, Union

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformer import TransformerLayer
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.moe.config import MoEConfig, ExpertConfig, GatingConfig

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class Qwen3(keras.Model):
    """
    Core Qwen3 model backbone implementation using the dl_techniques TransformerLayer.

    This class implements the main transformer stack of the Qwen3 architecture,
    acting as a feature extractor that outputs final hidden states. It is designed
    to be used by task-specific factory functions that add the final output head
    (e.g., for language modeling or sequence classification).

    **Architecture**:
    ```
    Input Token IDs → Token Embedding → Stack of TransformerLayers → Final Norm → Final Hidden States
    ```

    Each TransformerLayer automatically handles:
    - GroupedQueryAttention with RoPE positioning
    - RMSNorm for normalization (pre-norm architecture)
    - SwiGLU FFN or integrated MixtureOfExperts
    - Proper residual connections and dropout

    Args:
        vocab_size: Integer, vocabulary size. Must be positive.
        hidden_size: Integer, model dimension (d_model).
        num_layers: Integer, number of transformer layers.
        num_attention_heads: Integer, number of attention heads.
        num_key_value_heads: Integer, number of key-value heads for GQA.
        intermediate_size: Integer, FFN hidden dimension for non-MoE layers.
        max_seq_len: Integer, maximum sequence length for RoPE.
        moe_layers: List of integers, indices of layers that use MoE.
        num_experts: Integer, number of MoE experts for MoE layers.
        num_experts_per_tok: Integer, experts activated per token for MoE layers.
        moe_intermediate_size: Integer, hidden size for MoE experts.
        rope_theta: Float, RoPE theta parameter for position encoding.
        dropout_rate: Float, dropout rate for regularization.
        vocab_padding_size: Integer, pads vocabulary in the embedding layer for
            hardware efficiency. If None, no padding is applied.
        **kwargs: Additional model arguments for Model base class.
    """

    MODEL_VARIANTS = {
        "30b-coder": {
            "vocab_size": 151_936,
            "hidden_size": 2048,
            "num_layers": 48,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "intermediate_size": 5504,
            "moe_layers": list(range(0, 48, 3)),
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 768,
            "max_seq_len": 262_144,
            "rope_theta": 10_000_000.0,
            "dropout_rate": 0.0,
            "description": "Qwen3 Coder 30B-A3B style model",
        },
        "medium": {
            "vocab_size": 100_000,
            "hidden_size": 1024,
            "num_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "intermediate_size": 4096,
            "moe_layers": list(range(6, 24, 4)),
            "num_experts": 16,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 2048,
            "max_seq_len": 16_384,
            "rope_theta": 100_000.0,
            "dropout_rate": 0.05,
            "description": "Medium-sized Qwen3 model (24L, 1024-dim)",
        },
        "small": {
            "vocab_size": 32_000,
            "hidden_size": 768,
            "num_layers": 12,
            "num_attention_heads": 12,
            "num_key_value_heads": 4,
            "intermediate_size": 2048,
            "moe_layers": [3, 6, 9],
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 1024,
            "max_seq_len": 4096,
            "rope_theta": 10_000.0,
            "dropout_rate": 0.1,
            "description": "Small Qwen3 model for experimentation (12L, 768-dim)",
        },
    }

    def __init__(
            self,
            vocab_size: int,
            hidden_size: int,
            num_layers: int,
            num_attention_heads: int,
            num_key_value_heads: Optional[int] = None,
            intermediate_size: Optional[int] = None,
            max_seq_len: int = 32768,
            moe_layers: Optional[List[int]] = None,
            num_experts: int = 8,
            num_experts_per_tok: int = 2,
            moe_intermediate_size: Optional[int] = None,
            rope_theta: float = 10_000_000.0,
            dropout_rate: float = 0.0,
            vocab_padding_size: Optional[int] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Set defaults similar to Qwen3Next style
        num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        intermediate_size = intermediate_size if intermediate_size is not None else 4 * hidden_size
        moe_layers = moe_layers or []
        moe_intermediate_size = moe_intermediate_size if moe_intermediate_size is not None else intermediate_size

        # Validate configuration
        self._validate_config(
            vocab_size, hidden_size, num_layers, num_attention_heads,
            num_key_value_heads, dropout_rate
        )

        # Store configuration
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.moe_layers = moe_layers
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.rope_theta = rope_theta
        self.dropout_rate = dropout_rate
        self.vocab_padding_size = vocab_padding_size

        # Derived parameters
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.final_vocab_size = vocab_padding_size if vocab_padding_size is not None else vocab_size

        # Build the model architecture
        self._build_architecture()

    def _validate_config(
        self, vocab_size, hidden_size, num_layers, num_attention_heads,
        num_key_value_heads, dropout_rate
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
        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})")
        if num_key_value_heads <= 0:
            raise ValueError(f"num_key_value_heads must be positive, got {num_key_value_heads}")
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(f"num_attention_heads ({num_attention_heads}) must be divisible by num_key_value_heads ({num_key_value_heads})")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0.0 and 1.0, got {dropout_rate}")

    def _build_architecture(self) -> None:
        """Build all model components."""
        self.token_embedding = keras.layers.Embedding(
            input_dim=self.final_vocab_size,
            output_dim=self.hidden_size,
            embeddings_initializer="normal",
            name="token_embedding"
        )

        self.moe_config = None
        if self.moe_layers:
            expert_config = ExpertConfig(ffn_config={
                "type": "swiglu",
                "output_dim": self.hidden_size,
                "ffn_expansion_factor": self.moe_intermediate_size // self.hidden_size,
                "dropout_rate": self.dropout_rate,
                "use_bias": False
            })
            gating_config = GatingConfig(
                top_k=self.num_experts_per_tok, aux_loss_weight=0.01
            )
            self.moe_config = MoEConfig(self.num_experts, expert_config, gating_config)

        self.transformer_blocks = []
        for i in range(self.num_layers):
            attention_args = {
                'dim': self.hidden_size,
                'num_heads': self.num_attention_heads,
                'num_kv_heads': self.num_key_value_heads,
                'max_seq_len': self.max_seq_len,
                'dropout_rate': self.dropout_rate,
                'rope_theta': self.rope_theta
            }
            ffn_args = {
                'output_dim': self.hidden_size,
                'ffn_expansion_factor': self.intermediate_size // self.hidden_size,
                'dropout_rate': self.dropout_rate,
                'use_bias': False
            }
            is_moe_layer = i in self.moe_layers
            block = TransformerLayer(
                hidden_size=self.hidden_size,
                num_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
                attention_type='group_query_attention',
                attention_args=attention_args,
                normalization_type='rms_norm',
                normalization_position='pre',
                moe_config=self.moe_config if is_moe_layer else None,
                ffn_type='swiglu',
                ffn_args=ffn_args,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.dropout_rate,
                use_bias=False,
                name=f"transformer_{'moe_' if is_moe_layer else ''}block_{i}"
            )
            self.transformer_blocks.append(block)

        self.final_norm = create_normalization_layer(
            'rms_norm', axis=-1, epsilon=1e-6, use_scale=True, name="final_norm"
        )

    def call(
            self,
            inputs: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]],
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None,
            return_dict: bool = False
    ) -> Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """
        Forward pass through the Qwen3 backbone.

        Args:
            inputs: Input token IDs or dictionary containing inputs.
            attention_mask: Optional attention mask.
            training: Training mode flag.
            return_dict: Boolean, whether to return outputs as a dictionary.

        Returns:
            Final hidden states tensor of shape (batch_size, seq_len, hidden_size).
        """
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError("Dictionary input must contain 'input_ids' key")
            attention_mask = inputs.get("attention_mask", attention_mask)
        else:
            input_ids = inputs

        x = self.token_embedding(input_ids)

        for block in self.transformer_blocks:
            x = block(x, attention_mask=attention_mask, training=training)

        hidden_states = self.final_norm(x, training=training)

        if return_dict:
            return {"last_hidden_state": hidden_states}
        return hidden_states

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for serialization."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_attention_heads': self.num_attention_heads,
            'num_key_value_heads': self.num_key_value_heads,
            'intermediate_size': self.intermediate_size,
            'max_seq_len': self.max_seq_len,
            'moe_layers': self.moe_layers,
            'num_experts': self.num_experts,
            'num_experts_per_tok': self.num_experts_per_tok,
            'moe_intermediate_size': self.moe_intermediate_size,
            'rope_theta': self.rope_theta,
            'dropout_rate': self.dropout_rate,
            'vocab_padding_size': self.vocab_padding_size,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Qwen3":
        """Create model from configuration."""
        return cls(**config)

    @classmethod
    def from_variant(cls, variant: str, **kwargs: Any) -> "Qwen3":
        """Create a Qwen3 model from a predefined variant."""
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Available: {list(cls.MODEL_VARIANTS.keys())}")
        config = cls.MODEL_VARIANTS[variant].copy()
        config.pop("description", None)
        logger.info(f"Creating Qwen3-{variant.upper()} model backbone")
        logger.info(f"Configuration: {cls.MODEL_VARIANTS[variant]['description']}")
        return cls(**config, **kwargs)

    def summary(self, **kwargs) -> None:
        """Print model summary with additional Qwen3-specific information."""
        super().summary(**kwargs)
        logger.info("Qwen3 Model Configuration:")
        logger.info(f"  - Total layers: {self.num_layers}")
        logger.info(f"  - Hidden size: {self.hidden_size}")
        logger.info(f"  - Attention heads: {self.num_attention_heads} (KV heads: {self.num_key_value_heads})")
        logger.info(f"  - Vocabulary: {self.vocab_size:,} tokens (padded to {self.final_vocab_size:,})")
        logger.info(f"  - Max sequence length: {self.max_seq_len:,}")
        if self.moe_layers:
            logger.info("  - MoE Configuration:")
            logger.info(f"    - MoE layers: {len(self.moe_layers)} (at indices {self.moe_layers})")
            logger.info(f"    - Experts per layer: {self.num_experts}")
            logger.info(f"    - Active per token: {self.num_experts_per_tok}")

    def get_auxiliary_loss(self) -> Optional[keras.KerasTensor]:
        """Get auxiliary loss from MoE layers."""
        if not self.moe_layers:
            return None
        aux_losses = [
            block.get_auxiliary_loss() for i, block in enumerate(self.transformer_blocks)
            if i in self.moe_layers and hasattr(block, 'get_auxiliary_loss') and block.get_auxiliary_loss() is not None
        ]
        return keras.ops.sum(keras.ops.stack(aux_losses)) if aux_losses else None

# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def create_qwen3_for_generation(config: Dict[str, Any]) -> keras.Model:
    """Creates a Qwen3 model with a language modeling head for text generation."""
    logger.info("Creating Qwen3 model for text generation.")

    backbone = Qwen3(**config, name="qwen3_backbone")

    # Extract params needed for the head from the *full* config dict
    use_weight_tying = config.get('use_weight_tying', True)
    final_vocab_size = backbone.final_vocab_size
    vocab_size = backbone.vocab_size

    # Define model inputs
    input_ids = keras.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = keras.Input(shape=(None,), dtype="int32", name="attention_mask")

    # Get hidden states from the backbone
    hidden_states = backbone(inputs={"input_ids": input_ids, "attention_mask": attention_mask})

    # Create the language modeling head
    if use_weight_tying:
        embedding_weights = backbone.token_embedding.embeddings
        if backbone.vocab_padding_size is not None:
            embedding_weights = embedding_weights[:vocab_size, :]
        logits = keras.ops.matmul(hidden_states, keras.ops.transpose(embedding_weights))
    else:
        lm_head = keras.layers.Dense(
            final_vocab_size, use_bias=False, name="lm_head"
        )
        logits = lm_head(hidden_states)

    model = keras.Model(inputs=[input_ids, attention_mask], outputs=logits, name="qwen3_for_generation")
    logger.info(f"Created Qwen3 generation model with {model.count_params():,} parameters.")
    return model

# ---------------------------------------------------------------------

def create_qwen3_for_classification(
    config: Dict[str, Any],
    num_labels: int,
    pooling_strategy: str = "cls",
    classifier_dropout: Optional[float] = None,
) -> keras.Model:
    """Creates a Qwen3 model with a classification head."""
    if num_labels <= 0:
        raise ValueError(f"num_labels must be positive, got {num_labels}")
    if pooling_strategy not in ["cls", "mean"]:
        raise ValueError(f"pooling_strategy must be 'cls' or 'mean', got '{pooling_strategy}'")

    logger.info(f"Creating Qwen3 classification model with {num_labels} labels (pooling: '{pooling_strategy}').")

    backbone = Qwen3(**config, name="qwen3_backbone")

    input_ids = keras.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = keras.Input(shape=(None,), dtype="int32", name="attention_mask")

    sequence_output = backbone(inputs={"input_ids": input_ids, "attention_mask": attention_mask})

    if pooling_strategy == "cls":
        pooled_output = sequence_output[:, 0]
    else: # "mean" pooling
        mask = keras.ops.expand_dims(keras.ops.cast(attention_mask, sequence_output.dtype), axis=-1)
        masked_output = sequence_output * mask
        summed_output = keras.ops.sum(masked_output, axis=1)
        num_tokens = keras.ops.maximum(keras.ops.sum(keras.ops.cast(attention_mask, 'float32'), axis=1, keepdims=True), 1.0)
        pooled_output = summed_output / num_tokens

    dropout_rate = classifier_dropout if classifier_dropout is not None else config.get("dropout_rate", 0.1)
    if dropout_rate > 0.0:
        pooled_output = keras.layers.Dropout(dropout_rate, name="classifier_dropout")(pooled_output)

    logits = keras.layers.Dense(units=num_labels, name="classifier_head")(pooled_output)

    model = keras.Model(inputs=[input_ids, attention_mask], outputs=logits, name="qwen3_for_classification")
    logger.info(f"Created Qwen3 classification model with {model.count_params():,} parameters.")
    return model

# ---------------------------------------------------------------------

def create_qwen3(
    config_or_variant: Union[str, Dict[str, Any]],
    task_type: str = "generation",
    **kwargs: Any,
) -> keras.Model:
    """
    High-level factory to create Qwen3 models for common tasks.

    Args:
        config_or_variant: A variant string ('small', 'medium', '30b-coder') or a custom config dict.
        task_type: The model type to create ('generation' or 'classification').
        **kwargs: Overrides for config parameters or task-specific settings like `num_labels`.

    Returns:
        A Keras Model configured for the specified task.

    Example:
        ```python
        # Create a small model for generation
        model = create_qwen3("small")

        # Create a medium model for classification with 10 labels
        clf_model = create_qwen3("medium", task_type="classification", num_labels=10)

        # Create a custom model by overriding parameters
        custom_model = create_qwen3("small", num_layers=6, hidden_size=512)
        ```
    """
    if isinstance(config_or_variant, str):
        variant = config_or_variant
        if variant not in Qwen3.MODEL_VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Available: {list(Qwen3.MODEL_VARIANTS.keys())}")
        config = Qwen3.MODEL_VARIANTS[variant].copy()
        config.pop("description", None)
    elif isinstance(config_or_variant, dict):
        config = config_or_variant.copy()
    else:
        raise TypeError("config_or_variant must be a string (variant name) or a dictionary.")

    task_kwargs = {}
    model_kwargs = {}
    task_specific_keys = ["num_labels", "pooling_strategy", "classifier_dropout", "use_weight_tying"]
    for key, value in kwargs.items():
        if key in task_specific_keys:
            task_kwargs[key] = value
        else:
            model_kwargs[key] = value

    config.update(model_kwargs)

    # Add task-specific args to the main config dict for the factories
    config.update(task_kwargs)

    if task_type == "generation":
        return create_qwen3_for_generation(config)
    elif task_type == "classification":
        num_labels = config.pop("num_labels", None)
        if num_labels is None:
            raise ValueError("`num_labels` must be provided for 'classification' task.")
        return create_qwen3_for_classification(config, num_labels, **config)
    else:
        raise ValueError(f"Unknown task_type '{task_type}'. Supported: 'generation', 'classification'.")

# ---------------------------------------------------------------------
