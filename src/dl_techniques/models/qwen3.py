"""
Qwen3 Model Implementation for dl_techniques Framework

This module provides a Keras 3 implementation of the Qwen3 architecture,
reusing available dl_techniques components where possible.

Based on the Qwen3 architecture from:
- Qwen3: Think Deeper, Act Faster
- https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct
"""

import keras
from typing import Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from ..layers.norms.rms_norm import RMSNorm
from ..layers.ffn.swiglu_ffn import SwiGLUFFN
from ..layers.moe.layer import MixtureOfExperts
from ..layers.moe.config import MoEConfig, ExpertConfig, GatingConfig
from ..layers.attention.group_query_attention import GroupedQueryAttention

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Qwen3FeedForward(keras.layers.Layer):
    """
    Simple SwiGLU feed-forward network for Qwen3.

    This layer implements the standard feed-forward network used in Qwen3
    non-MoE layers, using SwiGLU activation.

    Args:
        d_model: Integer, model dimension.
        hidden_dim: Integer, hidden layer dimension.
        dtype: Data type for computations.
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dtype: Union[str, keras.DType] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.dtype = dtype

        # Use the existing SwiGLU implementation
        self.swiglu_ffn = SwiGLUFFN(
            d_model=d_model,
            ffn_expansion_factor=1,  # We'll set hidden_dim directly
            ffn_multiple_of=1,  # No constraints
            name="swiglu_ffn"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer."""
        # Manually set the hidden dimension for the SwiGLU layer
        # by overriding its computation
        self.swiglu_ffn._hidden_dim = self.hidden_dim
        self.swiglu_ffn.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass."""
        return self.swiglu_ffn(inputs, training=training)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'hidden_dim': self.hidden_dim,
            'dtype': self.dtype,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Qwen3MoEFeedForward(keras.layers.Layer):
    """
    Mixture of Experts feed-forward network for Qwen3.

    This implements the MoE routing used in Qwen3, where each token
    is routed to the top-k experts based on gating scores.

    Args:
        d_model: Integer, model dimension.
        num_experts: Integer, total number of experts.
        num_experts_per_tok: Integer, number of experts to activate per token.
        moe_intermediate_size: Integer, hidden size for each expert.
        dtype: Data type for computations.
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_intermediate_size: int,
        dtype: Union[str, keras.DType] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        self.d_model = d_model
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.dtype = dtype

        # Create MoE configuration
        expert_config = ExpertConfig(
            expert_type="ffn",
            hidden_dim=moe_intermediate_size,
            output_dim=d_model,
            activation="silu",
            use_bias=False,
            dropout_rate=0.0
        )

        gating_config = GatingConfig(
            gating_type="linear",
            top_k=num_experts_per_tok,
            normalize_gates=True,
            use_auxiliary_loss=True,
            auxiliary_loss_weight=0.01
        )

        moe_config = MoEConfig(
            num_experts=num_experts,
            expert_config=expert_config,
            gating_config=gating_config
        )

        # Create MoE layer using dl_techniques implementation
        self.moe_layer = MixtureOfExperts(
            config=moe_config,
            name="moe_layer"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the MoE layer."""
        self.moe_layer.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through MoE."""
        return self.moe_layer(inputs, training=training)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_experts': self.num_experts,
            'num_experts_per_tok': self.num_experts_per_tok,
            'moe_intermediate_size': self.moe_intermediate_size,
            'dtype': self.dtype,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Qwen3TransformerBlock(keras.layers.Layer):
    """
    Qwen3 Transformer block with grouped query attention and MoE/FFN.

    This implements a single transformer layer with:
    - Pre-normalization architecture
    - Grouped query attention with RoPE
    - Either MoE or standard FFN
    - Residual connections

    Args:
        d_model: Integer, model dimension.
        num_heads: Integer, number of attention heads.
        num_kv_groups: Integer, number of key-value groups for GQA.
        head_dim: Integer, dimension per attention head.
        hidden_dim: Integer, FFN hidden dimension (for non-MoE).
        num_experts: Integer, number of experts (0 for standard FFN).
        num_experts_per_tok: Integer, experts per token for MoE.
        moe_intermediate_size: Integer, hidden size for MoE experts.
        rope_theta: Float, RoPE theta parameter.
        qk_norm: Boolean, whether to use QK normalization.
        dtype: Data type for computations.
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_groups: int,
        head_dim: int,
        hidden_dim: int,
        num_experts: int = 0,
        num_experts_per_tok: int = 8,
        moe_intermediate_size: int = 768,
        rope_theta: float = 10_000_000.0,
        qk_norm: bool = True,
        dtype: Union[str, keras.DType] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Store configuration
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.rope_theta = rope_theta
        self.qk_norm = qk_norm
        self.dtype = dtype

        # Create normalization layers
        self.norm1 = RMSNorm(
            axis=-1,
            epsilon=1e-6,
            use_scale=True,
            scale_initializer="ones",
            name="norm1"
        )
        self.norm2 = RMSNorm(
            axis=-1,
            epsilon=1e-6,
            use_scale=True,
            scale_initializer="ones",
            name="norm2"
        )

        # Create attention layer
        self.attention = GroupedQueryAttention(
            d_model=d_model,
            n_head=num_heads,
            n_kv_head=num_kv_groups,
            dropout_rate=0.0,
            use_bias=False,
            rope_theta=rope_theta,
            name="attention"
        )

        # Create feed-forward layer (MoE or standard)
        if num_experts > 0:
            self.ffn = Qwen3MoEFeedForward(
                d_model=d_model,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                moe_intermediate_size=moe_intermediate_size,
                dtype=dtype,
                name="moe_ffn"
            )
        else:
            self.ffn = Qwen3FeedForward(
                d_model=d_model,
                hidden_dim=hidden_dim,
                dtype=dtype,
                name="ffn"
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build all sub-layers."""
        self.norm1.build(input_shape)
        self.norm2.build(input_shape)
        self.attention.build(input_shape)
        self.ffn.build(input_shape)
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through transformer block.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, d_model).
            attention_mask: Optional attention mask.
            training: Training mode flag.

        Returns:
            Output tensor of same shape as inputs.
        """
        # Pre-normalization architecture

        # Self-attention block
        residual = inputs
        x = self.norm1(inputs, training=training)
        x = self.attention(x, training=training)
        x = x + residual

        # Feed-forward block
        residual = x
        x = self.norm2(x, training=training)
        x = self.ffn(x, training=training)
        x = x + residual

        return x

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration."""
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_kv_groups': self.num_kv_groups,
            'head_dim': self.head_dim,
            'hidden_dim': self.hidden_dim,
            'num_experts': self.num_experts,
            'num_experts_per_tok': self.num_experts_per_tok,
            'moe_intermediate_size': self.moe_intermediate_size,
            'rope_theta': self.rope_theta,
            'qk_norm': self.qk_norm,
            'dtype': self.dtype,
        })
        return config

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Qwen3Model(keras.Model):
    """
    Complete Qwen3 model implementation for dl_techniques framework.

    This implements the full Qwen3 architecture with:
    - Token embeddings with optional weight tying
    - Stack of transformer blocks with MoE
    - Final layer normalization
    - Language modeling head

    Args:
        vocab_size: Integer, vocabulary size.
        d_model: Integer, model dimension.
        num_layers: Integer, number of transformer layers.
        num_heads: Integer, number of attention heads.
        num_kv_groups: Integer, number of key-value groups.
        head_dim: Integer, dimension per attention head.
        hidden_dim: Integer, FFN hidden dimension.
        num_experts: Integer, number of MoE experts.
        num_experts_per_tok: Integer, experts per token.
        moe_intermediate_size: Integer, MoE expert hidden size.
        context_length: Integer, maximum sequence length.
        rope_theta: Float, RoPE theta parameter.
        qk_norm: Boolean, whether to use QK normalization.
        use_weight_tying: Boolean, tie embedding and output weights.
        dtype: Data type for computations.
        **kwargs: Additional model arguments.
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
        num_experts: int = 0,
        num_experts_per_tok: int = 8,
        moe_intermediate_size: int = 768,
        context_length: int = 262_144,
        rope_theta: float = 10_000_000.0,
        qk_norm: bool = True,
        use_weight_tying: bool = True,
        dtype: Union[str, keras.DType] = "bfloat16",
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Store configuration
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.context_length = context_length
        self.rope_theta = rope_theta
        self.qk_norm = qk_norm
        self.use_weight_tying = use_weight_tying
        self.dtype = dtype

        # Token embeddings
        self.token_embedding = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=d_model,
            embeddings_initializer="normal",
            name="token_embedding"
        )

        # Transformer blocks
        self.transformer_blocks = []
        for i in range(num_layers):
            block = Qwen3TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_groups=num_kv_groups,
                head_dim=head_dim,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                moe_intermediate_size=moe_intermediate_size,
                rope_theta=rope_theta,
                qk_norm=qk_norm,
                dtype=dtype,
                name=f"transformer_block_{i}"
            )
            self.transformer_blocks.append(block)

        # Final normalization
        self.final_norm = RMSNorm(
            axis=-1,
            epsilon=1e-6,
            use_scale=True,
            scale_initializer="ones",
            name="final_norm"
        )

        # Output head
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
            attention_mask: Optional attention mask.
            training: Training mode flag.

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        # Token embeddings
        x = self.token_embedding(inputs)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask=attention_mask, training=training)

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
        """Return model configuration."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'num_kv_groups': self.num_kv_groups,
            'head_dim': self.head_dim,
            'hidden_dim': self.hidden_dim,
            'num_experts': self.num_experts,
            'num_experts_per_tok': self.num_experts_per_tok,
            'moe_intermediate_size': self.moe_intermediate_size,
            'context_length': self.context_length,
            'rope_theta': self.rope_theta,
            'qk_norm': self.qk_norm,
            'use_weight_tying': self.use_weight_tying,
            'dtype': self.dtype,
        })
        return config


def create_qwen3_coder_30b_config() -> Dict[str, Any]:
    """
    Create configuration for Qwen3-Coder-30B-A3B model.

    Returns:
        Dictionary with model configuration parameters.
    """
    return {
        'vocab_size': 151_936,
        'd_model': 2048,
        'num_layers': 48,
        'num_heads': 32,
        'num_kv_groups': 4,
        'head_dim': 128,
        'hidden_dim': 5504,  # For non-MoE layers
        'num_experts': 128,
        'num_experts_per_tok': 8,
        'moe_intermediate_size': 768,
        'context_length': 262_144,
        'rope_theta': 10_000_000.0,
        'qk_norm': True,
        'use_weight_tying': True,
        'dtype': 'bfloat16',
    }


def create_qwen3_model(
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> Qwen3Model:
    """
    Create a Qwen3 model with given configuration.

    Args:
        config: Optional model configuration dictionary.
        **kwargs: Additional configuration overrides.

    Returns:
        Configured Qwen3Model instance.

    Example:
        ```python
        # Create default Qwen3-Coder-30B model
        model = create_qwen3_model()

        # Create with custom config
        config = create_qwen3_coder_30b_config()
        config['num_layers'] = 24  # Smaller model
        model = create_qwen3_model(config)

        # Create with kwargs
        model = create_qwen3_model(
            vocab_size=50000,
            d_model=1024,
            num_layers=12
        )
        ```
    """
    if config is None:
        config = create_qwen3_coder_30b_config()

    # Override with any provided kwargs
    config = {**config, **kwargs}

    logger.info(f"Creating Qwen3 model with config: {config}")

    return Qwen3Model(**config)