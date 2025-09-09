"""
Qwen3 Model Implementation for dl_techniques Framework

This module provides a clean, refined Keras 3 implementation of the Qwen3 architecture,
leveraging the enhanced dl_techniques TransformerLayer with built-in MoE support and
factory systems for maximum code reuse and maintainability.

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
class Qwen3Model(keras.Model):
    """
    Complete Qwen3 model implementation using enhanced dl_techniques TransformerLayer.

    This implements the full Qwen3 architecture with:
    - Token embeddings with optional weight tying
    - Stack of configurable TransformerLayers with integrated MoE support
    - Final layer normalization using factory
    - Language modeling head

    **Architecture**:
    ```
    Input Token IDs → Token Embedding → Stack of TransformerLayers → Final Norm → Output Head
    ```

    Each TransformerLayer automatically handles:
    - GroupedQueryAttention with RoPE positioning
    - RMSNorm for normalization (pre-norm architecture)
    - SwiGLU FFN or integrated MixtureOfExperts
    - Proper residual connections and dropout

    **Key Features**:
    - Uses TransformerLayer's built-in MoE support via `moe_config` parameter
    - Clean factory-based component creation
    - Full Keras 3 serialization support
    - Efficient parameter management with optional vocabulary padding
    - Configurable MoE layers with load balancing

    Args:
        vocab_size: Integer, vocabulary size. Must be positive.
        d_model: Integer, model dimension. Must be positive and divisible by num_heads.
        num_layers: Integer, number of transformer layers. Must be positive.
        num_heads: Integer, number of attention heads. Must be positive and
            divide evenly into d_model.
        num_kv_groups: Integer, number of key-value groups for GQA. Must be
            positive and divide evenly into num_heads. Default is num_heads (no grouping).
        hidden_dim: Integer, FFN hidden dimension for non-MoE layers.
            Defaults to 4 * d_model.
        moe_layers: List of integers, indices of layers that use MoE.
            Empty list means no MoE. Defaults to empty list.
        num_experts: Integer, number of MoE experts for MoE layers.
            Only used if moe_layers is non-empty. Defaults to 8.
        num_experts_per_tok: Integer, experts activated per token.
            Only used if moe_layers is non-empty. Defaults to 2.
        moe_intermediate_size: Integer, hidden size for MoE experts.
            Only used if moe_layers is non-empty. Defaults to hidden_dim.
        context_length: Integer, maximum sequence length for RoPE.
            Defaults to 32768.
        rope_theta: Float, RoPE theta parameter for position encoding.
            Defaults to 10000000.0 (10M for long context).
        use_weight_tying: Boolean, tie embedding and output weights to reduce
            parameters. Defaults to True.
        dropout_rate: Float, dropout rate for regularization. Defaults to 0.0.
        vocab_padding_size: Integer, pad vocabulary to this size for efficiency.
            If None, no padding is applied. Defaults to None.
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
            moe_layers=list(range(0, 48, 3)),  # Every 3rd layer uses MoE
            num_experts=128,
            num_experts_per_tok=8,
            moe_intermediate_size=768,
            context_length=262144,
            rope_theta=10000000.0
        )

        # Create smaller model without MoE
        model = Qwen3Model(
            vocab_size=50000,
            d_model=768,
            num_layers=12,
            num_heads=12,
            num_kv_groups=4,
            hidden_dim=2048
        )

        # Compile for training
        model.compile(
            optimizer='adamw',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Forward pass
        token_ids = keras.random.randint((2, 128), 0, vocab_size)
        logits = model(token_ids)  # Shape: (2, 128, vocab_size)
        ```

    Note:
        This implementation uses the enhanced TransformerLayer with built-in
        MoE support, eliminating the need for custom transformer blocks and
        ensuring proper integration of all dl_techniques components.
    """

    def __init__(
            self,
            vocab_size: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            num_kv_groups: Optional[int] = None,
            hidden_dim: Optional[int] = None,
            moe_layers: Optional[List[int]] = None,
            num_experts: int = 8,
            num_experts_per_tok: int = 2,
            moe_intermediate_size: Optional[int] = None,
            context_length: int = 32768,
            rope_theta: float = 10_000_000.0,
            use_weight_tying: bool = True,
            dropout_rate: float = 0.0,
            vocab_padding_size: Optional[int] = None,
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

        # Set defaults
        if num_kv_groups is None:
            num_kv_groups = num_heads  # No grouping by default
        if hidden_dim is None:
            hidden_dim = 4 * d_model  # Standard 4x expansion
        if moe_layers is None:
            moe_layers = []
        if moe_intermediate_size is None:
            moe_intermediate_size = hidden_dim

        # Additional validations
        if num_kv_groups <= 0:
            raise ValueError(f"num_kv_groups must be positive, got {num_kv_groups}")
        if num_heads % num_kv_groups != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_groups ({num_kv_groups})")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0.0 and 1.0, got {dropout_rate}")

        # Store configuration
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = d_model // num_heads
        self.hidden_dim = hidden_dim
        self.moe_layers = moe_layers
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.context_length = context_length
        self.rope_theta = rope_theta
        self.use_weight_tying = use_weight_tying
        self.dropout_rate = dropout_rate
        self.vocab_padding_size = vocab_padding_size

        # Determine final vocabulary size (with optional padding)
        self.final_vocab_size = vocab_padding_size if vocab_padding_size is not None else vocab_size

        # CREATE sub-layers in __init__

        # Token embeddings
        self.token_embedding = keras.layers.Embedding(
            input_dim=self.final_vocab_size,
            output_dim=d_model,
            embeddings_initializer="normal",
            name="token_embedding"
        )

        # Create MoE configuration for MoE layers
        self.moe_config = None
        if moe_layers:
            logger.info(f"Configuring MoE for layers: {moe_layers}")

            # Create expert configuration using FFN factory format
            expert_config = ExpertConfig(
                ffn_config={
                    "type": "swiglu",
                    "output_dim": d_model,
                    "ffn_expansion_factor": moe_intermediate_size // d_model,
                    "dropout_rate": dropout_rate,
                    "use_bias": False
                }
            )

            # Create gating configuration
            gating_config = GatingConfig(
                top_k=num_experts_per_tok,
                aux_loss_weight=0.01,  # Standard auxiliary loss weight
                capacity_factor=1.25,   # Allow some load balancing flexibility
                noise_std=0.01 if dropout_rate > 0 else 0.0  # Add jitter if using dropout
            )

            self.moe_config = MoEConfig(
                num_experts=num_experts,
                expert_config=expert_config,
                gating_config=gating_config
            )

        # Create transformer blocks using enhanced TransformerLayer
        self.transformer_blocks = []
        for i in range(num_layers):
            # Determine if this layer uses MoE
            use_moe = i in moe_layers if moe_layers else False

            # Create attention arguments for GroupedQueryAttention
            attention_args = {
                'dim': d_model,
                'num_heads': num_heads,
                'num_kv_heads': num_kv_groups,
                'max_seq_len': context_length,
                'dropout_rate': dropout_rate,
                'rope_theta': rope_theta
            }

            # Create FFN arguments for SwiGLU
            ffn_args = {
                'output_dim': d_model,
                'ffn_expansion_factor': hidden_dim // d_model,
                'dropout_rate': dropout_rate,
                'use_bias': False
            }

            # Create transformer layer
            if use_moe:
                # Use MoE configuration - this replaces the standard FFN
                block = TransformerLayer(
                    hidden_size=d_model,
                    num_heads=num_heads,
                    intermediate_size=hidden_dim,  # Ignored when moe_config is provided
                    attention_type='group_query_attention',
                    attention_args=attention_args,
                    normalization_type='rms_norm',
                    normalization_position='pre',
                    moe_config=self.moe_config,  # This replaces FFN with MoE
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=dropout_rate,
                    use_bias=False,
                    name=f"transformer_moe_block_{i}"
                )
                logger.info(f"Created MoE transformer block {i} with {num_experts} experts")
            else:
                # Standard transformer block with SwiGLU FFN
                block = TransformerLayer(
                    hidden_size=d_model,
                    num_heads=num_heads,
                    intermediate_size=hidden_dim,
                    attention_type='group_query_attention',
                    attention_args=attention_args,
                    normalization_type='rms_norm',
                    normalization_position='pre',
                    ffn_type='swiglu',
                    ffn_args=ffn_args,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=dropout_rate,
                    use_bias=False,
                    name=f"transformer_block_{i}"
                )

            self.transformer_blocks.append(block)

        # Final normalization using the factory
        self.final_norm = create_normalization_layer(
            'rms_norm',
            axis=-1,
            epsilon=1e-6,
            use_scale=True,
            name="final_norm"
        )

        # Output head (conditional on weight tying)
        if not use_weight_tying:
            self.output_head = keras.layers.Dense(
                self.final_vocab_size,
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
                Currently not used but kept for future compatibility.
            training: Training mode flag for dropout and other training-specific behavior.

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        # Token embeddings
        x = self.token_embedding(inputs)

        # Apply transformer blocks
        # TransformerLayer handles all the internal logic including MoE
        for block in self.transformer_blocks:
            x = block(x, training=training)

        # Final normalization
        x = self.final_norm(x, training=training)

        # Output projection
        if self.output_head is not None:
            # Standard output head
            logits = self.output_head(x)
        else:
            # Weight tying - use transpose of embedding weights
            # Only use the actual vocabulary size, not padded size
            embedding_weights = self.token_embedding.embeddings
            if self.vocab_padding_size is not None:
                # Slice to actual vocabulary size
                embedding_weights = embedding_weights[:self.vocab_size, :]

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
            'hidden_dim': self.hidden_dim,
            'moe_layers': self.moe_layers,
            'num_experts': self.num_experts,
            'num_experts_per_tok': self.num_experts_per_tok,
            'moe_intermediate_size': self.moe_intermediate_size,
            'context_length': self.context_length,
            'rope_theta': self.rope_theta,
            'use_weight_tying': self.use_weight_tying,
            'dropout_rate': self.dropout_rate,
            'vocab_padding_size': self.vocab_padding_size,
        })
        return config

    def get_auxiliary_loss(self) -> Optional[keras.KerasTensor]:
        """
        Get auxiliary loss from MoE layers.

        Returns:
            Auxiliary loss tensor if MoE layers are present, None otherwise.
            This should be added to the main loss during training.
        """
        if not self.moe_layers:
            return None

        aux_losses = []
        for i, block in enumerate(self.transformer_blocks):
            if i in self.moe_layers and hasattr(block, 'get_auxiliary_loss'):
                aux_loss = block.get_auxiliary_loss()
                if aux_loss is not None:
                    aux_losses.append(aux_loss)

        if aux_losses:
            return keras.ops.sum(keras.ops.stack(aux_losses))
        return None

    def get_moe_metrics(self) -> Dict[str, keras.KerasTensor]:
        """
        Compute additional metrics for MoE models. This does not override the
        Keras `compute_metrics` method and is intended for manual inspection.

        Returns:
            Dictionary of additional metrics including load balancing statistics.
        """
        metrics = {}

        if self.moe_layers:
            # Collect MoE-specific metrics
            total_router_z_loss = []
            total_expert_usage = []

            for i, block in enumerate(self.transformer_blocks):
                if i in self.moe_layers and hasattr(block, 'get_metrics'):
                    block_metrics = block.get_metrics()
                    if block_metrics:
                        if 'router_z_loss' in block_metrics:
                            total_router_z_loss.append(block_metrics['router_z_loss'])
                        if 'expert_usage' in block_metrics:
                            total_expert_usage.append(block_metrics['expert_usage'])

            if total_router_z_loss:
                metrics['avg_router_z_loss'] = keras.ops.mean(
                    keras.ops.stack(total_router_z_loss)
                )

            if total_expert_usage:
                metrics['avg_expert_usage'] = keras.ops.mean(
                    keras.ops.stack(total_expert_usage)
                )

        return metrics


# ---------------------------------------------------------------------
# Configuration presets
# ---------------------------------------------------------------------

def create_qwen3_coder_30b_config() -> Dict[str, Any]:
    """
    Create configuration for Qwen3-Coder-30B-A3B model.

    Returns:
        Dictionary with model configuration parameters matching the
        official Qwen3-Coder-30B-A3B architecture specifications.

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


def create_qwen3_small_config() -> Dict[str, Any]:
    """
    Create configuration for a smaller Qwen3 model suitable for experimentation.

    Returns:
        Dictionary with smaller model configuration for development and testing.

    Example:
        ```python
        config = create_qwen3_small_config()
        model = Qwen3Model(**config)
        ```
    """
    return {
        'vocab_size': 32_000,
        'd_model': 768,
        'num_layers': 12,
        'num_heads': 12,
        'num_kv_groups': 4,
        'hidden_dim': 2048,
        'moe_layers': [3, 6, 9],  # Some MoE layers for testing
        'num_experts': 8,
        'num_experts_per_tok': 2,
        'moe_intermediate_size': 1024,
        'context_length': 4096,
        'rope_theta': 10_000.0,
        'use_weight_tying': True,
        'dropout_rate': 0.1,
    }


def create_qwen3_medium_config() -> Dict[str, Any]:
    """
    Create configuration for a medium-sized Qwen3 model.

    Returns:
        Dictionary with medium model configuration balancing performance and efficiency.

    Example:
        ```python
        config = create_qwen3_medium_config()
        model = create_qwen3_model(config=config)
        ```
    """
    return {
        'vocab_size': 100_000,
        'd_model': 1024,
        'num_layers': 24,
        'num_heads': 16,
        'num_kv_groups': 4,
        'hidden_dim': 4096,
        'moe_layers': list(range(6, 24, 4)),  # Sparse MoE usage
        'num_experts': 16,
        'num_experts_per_tok': 4,
        'moe_intermediate_size': 2048,
        'context_length': 16_384,
        'rope_theta': 100_000.0,
        'use_weight_tying': True,
        'dropout_rate': 0.05,
    }


# ---------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------

def create_qwen3_model(
        config: Optional[Dict[str, Any]] = None,
        model_size: Optional[str] = None,
        **kwargs: Any
) -> Qwen3Model:
    """
    Create a Qwen3 model with specified configuration.

    Args:
        config: Optional model configuration dictionary. Takes precedence over model_size.
        model_size: String identifier for predefined configurations:
            - 'small': Small model for experimentation (12 layers, 768 dim)
            - 'medium': Medium model for balanced performance (24 layers, 1024 dim)
            - '30b' or 'large': Qwen3-Coder-30B configuration (48 layers, 2048 dim)
            - If None and config is None, defaults to 'small'
        **kwargs: Additional configuration overrides that take precedence over
            both config dictionary and model_size preset values.

    Returns:
        Configured Qwen3Model instance ready for training or inference.

    Raises:
        ValueError: If model_size is not recognized or configuration is invalid.

    Example:
        ```python
        # Create small model (default)
        model = create_qwen3_model()

        # Create with size preset
        model = create_qwen3_model(model_size='medium')

        # Create large model
        model = create_qwen3_model(model_size='30b')

        # Create with custom config
        config = create_qwen3_small_config()
        config['num_layers'] = 6  # Even smaller
        model = create_qwen3_model(config=config)

        # Create with kwargs override
        model = create_qwen3_model(
            model_size='small',
            vocab_size=50000,    # Override vocabulary size
            dropout_rate=0.2     # Override dropout rate
        )

        # Compile for training
        model.compile(
            optimizer='adamw',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        ```
    """
    # Determine base configuration
    if config is not None:
        base_config = config.copy()
    elif model_size is not None:
        if model_size in ['small']:
            base_config = create_qwen3_small_config()
        elif model_size in ['medium']:
            base_config = create_qwen3_medium_config()
        elif model_size in ['30b', 'large']:
            base_config = create_qwen3_coder_30b_config()
        else:
            raise ValueError(f"Unknown model_size '{model_size}'. "
                           f"Available options: 'small', 'medium', '30b', 'large'")
    else:
        # Default to small model
        base_config = create_qwen3_small_config()

    # Override with any provided kwargs
    final_config = {**base_config, **kwargs}

    logger.info(f"Creating Qwen3 model: {final_config['num_layers']} layers, "
                f"{final_config['d_model']} dimensions, "
                f"MoE layers: {len(final_config.get('moe_layers', []))}")

    return Qwen3Model(**final_config)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def get_qwen3_model_info(model: Qwen3Model) -> Dict[str, Any]:
    """
    Get detailed information about a Qwen3 model instance.

    Args:
        model: Qwen3Model instance to analyze.

    Returns:
        Dictionary containing model architecture and parameter information.

    Example:
        ```python
        model = create_qwen3_model(model_size='medium')
        model(keras.ops.zeros((1, 10), dtype='int32')) # Build model
        info = get_qwen3_model_info(model)
        print(f"Total parameters: {info['estimated_total_params']:,}")
        print(f"MoE parameters: {info['estimated_moe_params']:,}")
        ```
    """
    info = {
        'architecture': 'Qwen3',
        'vocab_size': model.vocab_size,
        'd_model': model.d_model,
        'num_layers': model.num_layers,
        'num_heads': model.num_heads,
        'num_kv_groups': model.num_kv_groups,
        'head_dim': model.head_dim,
        'hidden_dim': model.hidden_dim,
        'context_length': model.context_length,
        'rope_theta': model.rope_theta,
        'use_weight_tying': model.use_weight_tying,
        'has_moe': bool(model.moe_layers),
        'moe_layers': model.moe_layers,
        'num_experts': model.num_experts if model.moe_layers else 0,
        'experts_per_token': model.num_experts_per_tok if model.moe_layers else 0,
    }

    # Calculate approximate parameter counts
    embed_params = model.final_vocab_size * model.d_model

    # Attention parameters for GQA (Wq, Wk, Wv, Wo)
    attention_params = (
            (model.d_model * model.d_model) +  # Wq
            (model.d_model * model.num_kv_groups * model.head_dim) +  # Wk
            (model.d_model * model.num_kv_groups * model.head_dim) +  # Wv
            (model.d_model * model.d_model)  # Wo
    )
    norm_params = model.d_model * 2  # Two RMSNorm layers per transformer block

    # SwiGLU FFN uses 3 matrices (gate, up, down), resulting in ~3x parameters
    # compared to a standard FFN's 2 matrices.
    standard_ffn_params = 3 * model.d_model * model.hidden_dim
    moe_ffn_params = (
            model.num_experts * (3 * model.d_model * model.moe_intermediate_size) +
            model.d_model * model.num_experts
    )  # Experts (each a SwiGLU FFN) + gating layer

    total_params = embed_params
    moe_params = 0

    for i in range(model.num_layers):
        total_params += attention_params + norm_params
        if i in model.moe_layers:
            total_params += moe_ffn_params
            moe_params += moe_ffn_params
        else:
            total_params += standard_ffn_params

    # Final norm and output head
    total_params += model.d_model  # Final norm
    if not model.use_weight_tying:
        total_params += model.final_vocab_size * model.d_model

    info.update({
        'estimated_total_params': total_params,
        'estimated_moe_params': moe_params,
        'estimated_non_moe_params': total_params - moe_params,
    })

    return info


# ---------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------

def compile_qwen3_for_training(
        model: Qwen3Model,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        gradient_clip_norm: float = 1.0
) -> None:
    """
    Compile Qwen3 model for training with optimal settings.

    Args:
        model: Qwen3Model to compile.
        learning_rate: Peak learning rate for cosine schedule.
        weight_decay: Weight decay for AdamW optimizer.
        warmup_steps: Number of warmup steps.
        max_steps: Total training steps.
        gradient_clip_norm: Gradient clipping norm.

    Example:
        ```python
        model = create_qwen3_model(model_size='small')
        compile_qwen3_for_training(model, learning_rate=5e-4)
        ```
    """
    from dl_techniques.optimization import optimizer_builder, learning_rate_schedule_builder

    # Create learning rate schedule
    lr_schedule = learning_rate_schedule_builder({
        'type': 'cosine_decay',
        'learning_rate': learning_rate,
        'decay_steps': max_steps,
        'warmup_steps': warmup_steps,
        'warmup_start_lr': learning_rate / 100,
        'alpha': 0.1
    })

    # Create optimizer
    optimizer = optimizer_builder({
        'type': 'adamw',
        'beta_1': 0.9,
        'beta_2': 0.95,
        'epsilon': 1e-8,
        'gradient_clipping_by_norm': gradient_clip_norm
    }, lr_schedule)

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    logger.info(f"Compiled Qwen3 model with lr={learning_rate}, "
                f"warmup_steps={warmup_steps}, max_steps={max_steps}")