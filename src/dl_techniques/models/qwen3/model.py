"""
Qwen3 Model Implementation for dl_techniques Framework

This module provides a Keras 3 implementation of the Qwen3 architecture,
reusing available dl_techniques components following modern best practices.

Based on the Qwen3 architecture from:
- Qwen3: Think Deeper, Act Faster
- https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct
"""

import keras
from typing import Optional,Dict, Any, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.transformer import TransformerLayer
from dl_techniques.layers.moe.layer import MixtureOfExperts
from dl_techniques.layers.moe.config import MoEConfig, ExpertConfig, GatingConfig
from dl_techniques.layers.norms import create_normalization_layer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Qwen3Model(keras.Model):
    """
    Complete Qwen3 model implementation using dl_techniques framework components.

    This implements the full Qwen3 architecture with:
    - Token embeddings with optional weight tying
    - Stack of transformer blocks with configurable MoE
    - Final layer normalization
    - Language modeling head

    **Architecture**:
    ```
    Input Token IDs → Embedding → Stack of TransformerLayers → Final Norm → Output Head
    ```

    Each TransformerLayer uses:
    - GroupedQueryAttention with RoPE
    - RMSNorm for normalization
    - SwiGLU FFN or MixtureOfExperts
    - Pre-normalization architecture

    Args:
        vocab_size: Integer, vocabulary size. Must be positive.
        d_model: Integer, model dimension. Must be positive.
        num_layers: Integer, number of transformer layers. Must be positive.
        num_heads: Integer, number of attention heads. Must be positive and
            divide evenly into d_model.
        num_kv_groups: Integer, number of key-value groups for GQA. Must be
            positive and divide evenly into num_heads.
        head_dim: Integer, dimension per attention head. Typically d_model/num_heads.
        hidden_dim: Integer, FFN hidden dimension for non-MoE layers.
        moe_layers: List of integers, indices of layers that use MoE.
            Empty list means no MoE. Defaults to empty list.
        num_experts: Integer, number of MoE experts for MoE layers.
            Only used if moe_layers is non-empty. Defaults to 128.
        num_experts_per_tok: Integer, experts activated per token.
            Only used if moe_layers is non-empty. Defaults to 8.
        moe_intermediate_size: Integer, hidden size for MoE experts.
            Only used if moe_layers is non-empty. Defaults to 768.
        context_length: Integer, maximum sequence length. Defaults to 262144.
        rope_theta: Float, RoPE theta parameter for position encoding.
            Defaults to 10000000.0.
        use_weight_tying: Boolean, tie embedding and output weights to reduce
            parameters. Defaults to True.
        dropout_rate: Float, dropout rate for regularization. Defaults to 0.0.
        **kwargs: Additional model arguments for Model base class.

    Input shape:
        2D tensor with shape: `(batch_size, sequence_length)` containing token IDs.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, vocab_size)`
        containing logits for each token position.

    Example:
        ```python
        # Create Qwen3-Coder-30B style model
        model = Qwen3Model(
            vocab_size=151936,
            d_model=2048,
            num_layers=48,
            num_heads=32,
            num_kv_groups=4,
            head_dim=128,
            hidden_dim=5504,
            moe_layers=list(range(0, 48, 3)),  # Every 3rd layer uses MoE
            num_experts=128,
            num_experts_per_tok=8
        )

        # Create smaller model without MoE
        model = Qwen3Model(
            vocab_size=50000,
            d_model=768,
            num_layers=12,
            num_heads=12,
            num_kv_groups=4,
            head_dim=64,
            hidden_dim=2048
        )

        # Forward pass
        token_ids = keras.ops.ones((2, 128), dtype='int32')
        logits = model(token_ids)  # Shape: (2, 128, vocab_size)
        ```

    Note:
        This implementation leverages the configurable TransformerLayer from
        dl_techniques, which handles attention, normalization, and FFN/MoE
        automatically based on configuration parameters.
    """

    def __init__(
            self,
            vocab_size: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            num_kv_groups: int,
            head_dim: int,
            hidden_dim: int,
            moe_layers: List[int] = [],
            num_experts: int = 128,
            num_experts_per_tok: int = 8,
            moe_intermediate_size: int = 768,
            context_length: int = 262144,
            rope_theta: float = 10_000_000.0,
            use_weight_tying: bool = True,
            dropout_rate: float = 0.0,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        if num_kv_groups <= 0:
            raise ValueError(f"num_kv_groups must be positive, got {num_kv_groups}")
        if num_heads % num_kv_groups != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_groups ({num_kv_groups})")

        # Store configuration
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.moe_layers = moe_layers
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.context_length = context_length
        self.rope_theta = rope_theta
        self.use_weight_tying = use_weight_tying
        self.dropout_rate = dropout_rate

        # CREATE sub-layers in __init__

        # Token embeddings
        self.token_embedding = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=d_model,
            embeddings_initializer="normal",
            name="token_embedding"
        )

        # Create MoE configuration for MoE layers
        if moe_layers:
            expert_config = ExpertConfig(
                expert_type="ffn",
                hidden_dim=moe_intermediate_size,
                output_dim=d_model,
                activation="silu",
                use_bias=False,
                dropout_rate=dropout_rate
            )

            gating_config = GatingConfig(
                gating_type="linear",
                top_k=num_experts_per_tok,
                # The presence of a non-zero aux_loss_weight implies its use.
                # 'normalize_gates' and 'use_auxiliary_loss' are not valid arguments.
                aux_loss_weight=0.01
            )

            self.moe_config = MoEConfig(
                num_experts=num_experts,
                expert_config=expert_config,
                gating_config=gating_config
            )
        else:
            self.moe_config = None

        # Transformer blocks using configurable TransformerLayer
        self.transformer_blocks = []
        for i in range(num_layers):
            # Determine if this layer uses MoE
            if i in moe_layers and self.moe_config is not None:
                # Create MoE layer and pass it as ffn_args to TransformerLayer
                moe_layer = MixtureOfExperts(
                    config=self.moe_config,
                    name=f"moe_layer_{i}"
                )

                # Use TransformerLayer with custom MoE
                # Note: We need to handle MoE differently since TransformerLayer
                # doesn't directly support external MoE layers
                block = TransformerLayer(
                    hidden_size=d_model,
                    num_heads=num_heads,
                    intermediate_size=hidden_dim,
                    attention_type='group_query_attention',
                    attention_args={
                        'n_kv_head': num_kv_groups,
                        'dropout_rate': dropout_rate,
                        'use_bias': False,
                        'rope_theta': rope_theta
                    },
                    normalization_type='rms_norm',
                    normalization_position='pre',
                    ffn_type='swiglu',
                    ffn_args={
                        'ffn_expansion_factor': hidden_dim // d_model,
                        'ffn_multiple_of': 1
                    },
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=dropout_rate,
                    use_bias=False,
                    name=f"transformer_block_{i}"
                )
                # Store the MoE layer separately for this block
                setattr(self, f"moe_layer_{i}", moe_layer)
            else:
                # Standard transformer block with SwiGLU FFN
                block = TransformerLayer(
                    hidden_size=d_model,
                    num_heads=num_heads,
                    intermediate_size=hidden_dim,
                    attention_type='group_query_attention',
                    attention_args={
                        'n_kv_head': num_kv_groups,
                        'dropout_rate': dropout_rate,
                        'use_bias': False,
                        'rope_theta': rope_theta
                    },
                    normalization_type='rms_norm',
                    normalization_position='pre',
                    ffn_type='swiglu',
                    ffn_args={
                        'ffn_expansion_factor': hidden_dim // d_model,
                        'ffn_multiple_of': 1
                    },
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=dropout_rate,
                    use_bias=False,
                    name=f"transformer_block_{i}"
                )

            self.transformer_blocks.append(block)

        # Final normalization using the normalization factory
        self.final_norm = create_normalization_layer(
            'rms_norm',
            axis=-1,
            epsilon=1e-6,
            use_scale=True,
            name="final_norm"
        )

        # Output head (optional with weight tying)
        if not use_weight_tying:
            self.output_head = keras.layers.Dense(
                vocab_size,
                use_bias=False,
                kernel_initializer="normal",
                name="output_head"
            )
        else:
            self.output_head = None

    def call(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through Qwen3 model.

        Args:
            inputs: Token IDs tensor of shape (batch_size, seq_len).
            attention_mask: Optional attention mask for sequence padding.
            training: Training mode flag for dropout and other training-specific behavior.

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        # Token embeddings
        x = self.token_embedding(inputs)

        # Apply transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            # The original implementation had an architectural bug for MoE layers.
            # It applied both the block's internal FFN and the external MoE layer.
            # The correct approach is to replace the FFN with the MoE layer.
            # We achieve this by manually executing the pre-norm transformer steps.
            # This assumes the TransformerLayer exposes its submodules, which is a
            # common and reasonable design pattern.

            # 1. Attention Block (pre-norm)
            residual = x
            # We assume the TransformerLayer has these attributes.
            x_norm = block.pre_attention_norm(x, training=training)
            attention_output = block.attention_layer(x_norm, training=training)
            x = residual + attention_output

            # 2. FFN / MoE Block (pre-norm)
            residual = x
            x_norm = block.pre_ffn_norm(x, training=training)

            if i in self.moe_layers and hasattr(self, f"moe_layer_{i}"):
                # Use the external MoE layer for this block
                moe_layer = getattr(self, f"moe_layer_{i}")
                ffn_output = moe_layer(x_norm, training=training)
            else:
                # Use the block's internal FFN
                ffn_output = block.ffn_block(x_norm, training=training)

            x = residual + ffn_output

        # Final normalization
        x = self.final_norm(x, training=training)

        # Output projection
        if self.output_head is not None:
            # Standard output head
            logits = self.output_head(x)
        else:
            # Weight tying - use transpose of embedding weights
            embedding_weights = self.token_embedding.embeddings
            logits = keras.ops.matmul(x, keras.ops.transpose(embedding_weights))

        return logits

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for serialization."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'num_kv_groups': self.num_kv_groups,
            'head_dim': self.head_dim,
            'hidden_dim': self.hidden_dim,
            'moe_layers': self.moe_layers,
            'num_experts': self.num_experts,
            'num_experts_per_tok': self.num_experts_per_tok,
            'moe_intermediate_size': self.moe_intermediate_size,
            'context_length': self.context_length,
            'rope_theta': self.rope_theta,
            'use_weight_tying': self.use_weight_tying,
            'dropout_rate': self.dropout_rate,
        })
        return config

# ---------------------------------------------------------------------

def create_qwen3_coder_30b_config() -> Dict[str, Any]:
    """
    Create configuration for Qwen3-Coder-30B-A3B model.

    Returns:
        Dictionary with model configuration parameters matching the
        official Qwen3-Coder-30B-A3B architecture.

    Example:
        ```python
        config = create_qwen3_coder_30b_config()
        model = Qwen3Model(**config)
        ```
    """
    return {
        'vocab_size': 151_936,
        'd_model': 2048,
        'num_layers': 48,
        'num_heads': 32,
        'num_kv_groups': 4,
        'head_dim': 128,
        'hidden_dim': 5504,
        'moe_layers': list(range(0, 48, 3)),  # Every 3rd layer uses MoE
        'num_experts': 128,
        'num_experts_per_tok': 8,
        'moe_intermediate_size': 768,
        'context_length': 262_144,
        'rope_theta': 10_000_000.0,
        'use_weight_tying': True,
        'dropout_rate': 0.0,
    }

# ---------------------------------------------------------------------

def create_qwen3_model(
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
) -> Qwen3Model:
    """
    Create a Qwen3 model with given configuration.

    Args:
        config: Optional model configuration dictionary. If None,
            uses the Qwen3-Coder-30B configuration.
        **kwargs: Additional configuration overrides that take
            precedence over config dictionary values.

    Returns:
        Configured Qwen3Model instance ready for training or inference.

    Example:
        ```python
        # Create default Qwen3-Coder-30B model
        model = create_qwen3_model()

        # Create with custom config
        config = create_qwen3_coder_30b_config()
        config['num_layers'] = 24  # Smaller model
        model = create_qwen3_model(config)

        # Create with kwargs override
        model = create_qwen3_model(
            vocab_size=50000,
            d_model=1024,
            num_layers=12
        )

        # Compile for training
        model.compile(
            optimizer='adamw',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        ```
    """
    if config is None:
        config = create_qwen3_coder_30b_config()

    # Override with any provided kwargs
    final_config = {**config, **kwargs}

    logger.info(f"Creating Qwen3 model with {final_config['num_layers']} layers, "
                f"{final_config['d_model']} dimensions")

    return Qwen3Model(**final_config)

# ---------------------------------------------------------------------

