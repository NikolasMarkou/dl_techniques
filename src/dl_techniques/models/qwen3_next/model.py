"""
Qwen3 Next Model Implementation (Corrected)
============================================

A complete implementation of the Qwen3 Next architecture following the correct block structure:
- Each block contains 3x Gated DeltaNet layers + 1x Gated Attention layer
- Each layer has its own Zero-Centered RMSNorm and MoE
- Proper residual connections throughout

Based on the architectural diagram showing the precise layer arrangement and connections.
"""

import keras
import numpy as np
from typing import Optional, Union, Any, Dict

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.moe import MoEConfig, ExpertConfig, GatingConfig

from .components import Qwen3NextBlock

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Qwen3Next(keras.Model):
    """
    Qwen3 Next (Mixture of Experts) model with correct architecture.

    This implementation follows the exact pattern shown in the architecture diagram:
    - Token embeddings + RoPE position embeddings
    - N blocks, each containing 3x GatedDeltaNet + 1x GatedAttention
    - Each layer has its own Zero-Centered RMSNorm and MoE
    - Final normalization and language modeling head

    **Architecture Overview:**
    ```
    Input(input_ids)
           │
           ▼
    Token Embeddings (vocab_size=151k, dim=2048)
           │
           ▼
    RoPE Position Embeddings (theta=1M)
           │
           ▼
    Qwen3NextBlock₁:
        3x [Zero-Centered RMSNorm → GatedDeltaNet → MoE → Residual]
        1x [Zero-Centered RMSNorm → GatedAttention → MoE → Residual]
           │
           ▼
          ...
           │
           ▼
    Qwen3NextBlockₙ:
        3x [Zero-Centered RMSNorm → GatedDeltaNet → MoE → Residual]
        1x [Zero-Centered RMSNorm → GatedAttention → MoE → Residual]
           │
           ▼
    Final Zero-Centered RMSNorm
           │
           ▼
    Linear Projection → Logits (vocab_size=151k)
    ```

    Args:
        vocab_size: Integer, size of the vocabulary. Defaults to 151936.
        hidden_size: Integer, dimensionality of encoder layers. Defaults to 2048.
        num_layers: Integer, number of transformer blocks. Defaults to 12.
        num_attention_heads: Integer, number of attention heads. Defaults to 16.
        num_key_value_heads: Integer, number of key-value heads for GQA. Defaults to 4.
        max_position_embeddings: Integer, maximum sequence length. Defaults to 8192.
        num_experts: Integer, total number of experts in MoE layers. Defaults to 64.
        num_experts_per_tok: Integer, number of experts activated per token. Defaults to 8.
        moe_intermediate_size: Integer, individual expert intermediate size. Defaults to 1408.
        norm_eps: Float, epsilon for normalization layers. Defaults to 1e-6.
        dropout_rate: Float, dropout rate for regularization. Defaults to 0.0.
        initializer_range: Float, standard deviation for weight initialization. Defaults to 0.02.
        normalization_type: String, type of normalization layer. Defaults to "zero_centered_rms_norm".
        ffn_type: String, type of feed-forward network in experts. Defaults to "swiglu".
        use_stochastic_depth: Boolean, whether to enable stochastic depth. Defaults to False.
        stochastic_depth_rate: Float, drop path rate for stochastic depth. Defaults to 0.1.
        **kwargs: Additional keyword arguments for the `keras.Model` base class.
    """

    # Model variant configurations following Qwen3 Next specifications
    MODEL_VARIANTS = {
        "80b_a3b": {
            "vocab_size": 151936,
            "hidden_size": 2048,
            "num_layers": 12,  # 12 blocks, each with 3 delta + 1 attn = 48 layers total
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "max_position_embeddings": 8192,
            "num_experts": 64,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 1408,
            "description": "Qwen3 Next 80B-A3B: 12 blocks × (3 delta + 1 attn) = 48 effective layers"
        },
        "80b": {
            "vocab_size": 151936,
            "hidden_size": 2048,
            "num_layers": 12,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "max_position_embeddings": 8192,
            "num_experts": 1,  # Dense model
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 5632,
            "description": "Qwen3 Next 80B Dense: 12 blocks without MoE"
        },
        "small": {
            "vocab_size": 151936,
            "hidden_size": 1024,
            "num_layers": 6,  # 6 blocks
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "max_position_embeddings": 2048,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 704,
            "description": "Qwen3 Next Small: 6 blocks for experimentation"
        },
        "tiny": {
            "vocab_size": 151936,
            "hidden_size": 512,
            "num_layers": 3,  # 3 blocks
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "max_position_embeddings": 1024,
            "num_experts": 4,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 352,
            "description": "Qwen3 Next Tiny: 3 blocks for mobile/edge deployment"
        },
    }

    def __init__(
            self,
            vocab_size: int = 151936,
            hidden_size: int = 2048,
            num_layers: int = 12,
            num_attention_heads: int = 16,
            num_key_value_heads: int = 4,
            max_position_embeddings: int = 8192,
            num_experts: int = 64,
            num_experts_per_tok: int = 8,
            moe_intermediate_size: int = 1408,
            norm_eps: float = 1e-6,
            dropout_rate: float = 0.0,
            initializer_range: float = 0.02,
            normalization_type: str = "zero_centered_rms_norm",
            ffn_type: str = "swiglu",
            use_stochastic_depth: bool = False,
            stochastic_depth_rate: float = 0.1,
            **kwargs: Any
    ) -> None:
        # CRITICAL FIX: Call super() FIRST. This is mandatory for Keras models.
        super().__init__(**kwargs)

        # Validate configuration parameters
        self._validate_config(
            vocab_size, hidden_size, num_layers, num_attention_heads,
            num_key_value_heads, num_experts, num_experts_per_tok
        )

        # Store all configuration parameters
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
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
        total_effective_layers = num_layers * 4  # Each block has 3 delta + 1 attn
        active_params_pct = (self.num_experts_per_tok / self.num_experts) * 100 if self.num_experts > 1 else 100.0
        logger.info(
            f"Created Qwen3 Next model: {self.num_layers} blocks "
            f"({total_effective_layers} effective layers), "
            f"hidden_size={self.hidden_size}, experts={self.num_experts}, "
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

        # CRITICAL FIX: Removed unused `self.rope_embedding`.
        # RoPE is handled within each `GatedAttention` layer, not at the model level.

        # Create MoE configuration
        moe_config = None
        if self.num_experts > 1:
            moe_config = MoEConfig(
                num_experts=self.num_experts,
                expert_config=ExpertConfig(
                    ffn_config={
                        "type": self.ffn_type,
                        "output_dim": self.hidden_size,
                        "ffn_expansion_factor": max(1, self.moe_intermediate_size // self.hidden_size)
                    }
                ),
                gating_config=GatingConfig(
                    top_k=self.num_experts_per_tok,
                    gating_type="linear"
                )
            )

        # Create a linear schedule for the drop path rate
        if self.stochastic_depth_rate > 0:
            dpr = [x for x in np.linspace(0.0, self.stochastic_depth_rate, self.num_layers)]
        else:
            dpr = [0.0 for _ in range(self.num_layers)]

        # Create Qwen3Next blocks
        self.blocks = []
        for i in range(self.num_layers):
            block = Qwen3NextBlock(
                dim=self.hidden_size,
                num_heads=self.num_attention_heads,
                head_dim=self.head_dim,
                moe_config=moe_config,
                normalization_type=self.normalization_type,
                norm_eps=self.norm_eps,
                dropout_rate=self.dropout_rate,
                use_stochastic_depth=self.use_stochastic_depth,
                stochastic_depth_rate=dpr[i],
                name=f"qwen3_next_block_{i}"
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
        Forward pass of the Qwen3 Next model.

        Args:
            inputs: Input token IDs or dictionary containing inputs.
            attention_mask: Mask to avoid attention on padding tokens.
            training: Boolean, whether the model is in training mode.
            return_dict: Boolean, whether to return outputs as a dictionary.

        Returns:
            Model outputs. The format depends on `return_dict`:
            - `return_dict=False`: `logits` tensor of shape (batch, seq_len, vocab_size).
            - `return_dict=True`: Dictionary with keys `logits` and optionally others.
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

        # Pass through all Qwen3Next blocks
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
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
    ) -> "Qwen3Next":
        """
        Create a Qwen3 Next model from a predefined variant.

        Args:
            variant: String, one of "80b_a3b", "80b", "small", "tiny"
            **kwargs: Additional arguments passed to the constructor

        Returns:
            Qwen3Next model instance
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        config.pop("description", None)

        logger.info(f"Creating Qwen3Next-{variant.upper()} model")
        logger.info(f"Configuration: {cls.MODEL_VARIANTS[variant]['description']}")

        return cls(**config, **kwargs)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "moe_intermediate_size": self.moe_intermediate_size,
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
    def from_config(cls, config: Dict[str, Any]) -> "Qwen3Next":
        """Create model from configuration."""
        return cls(**config)

    def summary(self, **kwargs) -> None:
        """Print model summary with additional Qwen3 Next-specific information."""
        super().summary(**kwargs)

        # Calculate statistics
        total_blocks = self.num_layers
        total_delta_layers = total_blocks * 3
        total_attention_layers = total_blocks * 1
        total_effective_layers = total_delta_layers + total_attention_layers
        total_experts = self.num_experts * total_effective_layers if self.num_experts > 1 else 0
        active_experts_per_token = self.num_experts_per_tok * total_effective_layers if self.num_experts > 1 else 0
        sparsity_ratio = self.num_experts / self.num_experts_per_tok if self.num_experts > 1 else 1

        logger.info("Qwen3 Next Model Configuration:")
        logger.info(f"  - Architecture: {total_blocks} blocks → {total_effective_layers} effective layers")
        logger.info(f"    - {total_delta_layers} Gated DeltaNet layers")
        logger.info(f"    - {total_attention_layers} Gated Attention layers")
        logger.info(f"  - Hidden size: {self.hidden_size}")
        logger.info(f"  - Attention heads: {self.num_attention_heads} (KV heads: {self.num_key_value_heads})")
        logger.info(f"  - Vocabulary: {self.vocab_size:,} tokens")
        logger.info(f"  - Max sequence length: {self.max_position_embeddings:,}")
        if self.num_experts > 1:
            logger.info(f"  - MoE Configuration:")
            logger.info(f"    - Experts per layer: {self.num_experts}")
            logger.info(f"    - Active per token: {self.num_experts_per_tok}")
            logger.info(f"    - Sparsity ratio: {sparsity_ratio:.1f}:1")
            logger.info(f"    - Total experts: {total_experts:,}")
            logger.info(f"    - Active experts per token: {active_experts_per_token}")
        logger.info(f"  - Normalization: {self.normalization_type}")
        logger.info(f"  - Expert FFN: {self.ffn_type}")
        if self.use_stochastic_depth:
            logger.info(f"  - Stochastic depth: {self.stochastic_depth_rate}")


# ---------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------

def create_qwen3_next_generation(config: Dict[str, Any]) -> keras.Model:
    """Create a Qwen3 Next model optimized for text generation."""
    logger.info("Creating Qwen3 Next model for text generation")

    qwen3_next = Qwen3Next(**config, name="qwen3_next")
    input_ids = keras.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = keras.Input(shape=(None,), dtype="int32", name="attention_mask")

    logits = qwen3_next(
        inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    )

    model = keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=logits,
        name="qwen3_next_for_generation"
    )

    logger.info(f"Created Qwen3 Next generation model with {model.count_params():,} parameters")
    return model

# ---------------------------------------------------------------------

def create_qwen3_next_classification(
        config: Dict[str, Any],
        num_labels: int,
        classifier_dropout: Optional[float] = None
) -> keras.Model:
    """Create a Qwen3 Next model for sequence classification tasks."""
    if num_labels <= 0:
        raise ValueError(f"num_labels must be positive, got {num_labels}")

    logger.info(f"Creating Qwen3 Next classification model with {num_labels} labels")

    qwen3_next = Qwen3Next(**config, name="qwen3_next")
    input_ids = keras.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = keras.Input(shape=(None,), dtype="int32", name="attention_mask")

    sequence_output = qwen3_next(
        inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    )

    pooled_output = sequence_output[:, 0]  # Shape: (batch_size, hidden_size)

    if classifier_dropout is None:
        classifier_dropout = config.get("dropout_rate", 0.1)

    if classifier_dropout > 0.0:
        pooled_output = keras.layers.Dropout(
            classifier_dropout,
            name="classifier_dropout"
        )(pooled_output)

    logits = keras.layers.Dense(
        units=num_labels,
        kernel_initializer=keras.initializers.TruncatedNormal(
            stddev=config.get("initializer_range", 0.02)
        ),
        name="classifier"
    )(pooled_output)

    model = keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=logits,
        name="qwen3_next_for_classification"
    )

    logger.info(f"Created Qwen3 Next classification model with {model.count_params():,} parameters")
    return model

# ---------------------------------------------------------------------

def create_qwen3_next(
        variant: str = "small",
        task_type: str = "generation",
        num_labels: Optional[int] = None,
        **kwargs: Any
) -> keras.Model:
    """Convenience function to create Qwen3 Next models for common tasks."""
    if variant in Qwen3Next.MODEL_VARIANTS:
        config = Qwen3Next.MODEL_VARIANTS[variant].copy()
        config.pop("description", None)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    config.update(kwargs)

    if task_type == "generation":
        return create_qwen3_next_generation(config)
    elif task_type == "classification":
        if num_labels is None:
            raise ValueError("num_labels must be provided for classification task")
        return create_qwen3_next_classification(config, num_labels)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

# ---------------------------------------------------------------------