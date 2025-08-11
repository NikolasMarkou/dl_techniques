"""
Configuration classes for Mixture of Experts (MoE) models.

This module provides comprehensive configuration dataclasses for MoE components,
enabling flexible and reproducible model architectures.
"""

import keras
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any, Literal

@dataclass
class ExpertConfig:
    """
    Configuration for individual expert networks in MoE models.

    This dataclass defines the architecture and behavior of expert networks,
    supporting various expert types including FFN, attention, and convolutional experts.

    Args:
        expert_type: Type of expert network ('ffn', 'attention', 'conv2d').
        hidden_dim: Hidden dimension size for expert networks.
        output_dim: Output dimension size. If None, defaults to input dimension.
        activation: Activation function for expert networks.
        dropout_rate: Dropout probability for regularization.
        use_bias: Whether to include bias terms in linear layers.
        kernel_initializer: Weight initialization strategy.
        bias_initializer: Bias initialization strategy.
        kernel_regularizer: Regularization applied to weights.
        bias_regularizer: Regularization applied to biases.

        # FFN-specific parameters
        intermediate_size: Intermediate layer size for FFN experts.

        # Attention-specific parameters
        num_heads: Number of attention heads for attention experts.
        head_dim: Dimension per attention head.

        # Conv2D-specific parameters
        filters: Number of filters for convolutional experts.
        kernel_size: Kernel size for convolutional experts.
        strides: Stride configuration for convolutional experts.
        padding: Padding configuration for convolutional experts.

    Example:
        ```python
        # FFN expert configuration
        ffn_config = ExpertConfig(
            expert_type='ffn',
            hidden_dim=768,
            intermediate_size=3072,
            activation='gelu'
        )

        # Attention expert configuration
        attn_config = ExpertConfig(
            expert_type='attention',
            hidden_dim=768,
            num_heads=12,
            head_dim=64
        )
        ```
    """
    expert_type: Literal['ffn', 'attention', 'conv2d'] = 'ffn'
    hidden_dim: int = 768
    output_dim: Optional[int] = None
    activation: Union[str, callable] = 'gelu'
    dropout_rate: float = 0.1
    use_bias: bool = True
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform'
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros'
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None
    bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None

    # FFN-specific parameters
    intermediate_size: Optional[int] = None

    # Attention-specific parameters
    num_heads: int = 8
    head_dim: Optional[int] = None

    # Conv2D-specific parameters
    filters: Optional[int] = None
    kernel_size: Union[int, tuple] = 3
    strides: Union[int, tuple] = 1
    padding: str = 'same'

    def __post_init__(self):
        """Initialize derived parameters after dataclass creation."""
        if self.output_dim is None:
            self.output_dim = self.hidden_dim

        if self.expert_type == 'ffn' and self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_dim

        if self.expert_type == 'attention' and self.head_dim is None:
            self.head_dim = self.hidden_dim // self.num_heads

        if self.expert_type == 'conv2d' and self.filters is None:
            self.filters = self.hidden_dim


@dataclass
class GatingConfig:
    """
    Configuration for MoE gating networks (routers).

    This dataclass defines the routing mechanism for MoE models, supporting
    various gating strategies and load balancing techniques.

    Args:
        gating_type: Type of gating mechanism ('linear', 'cosine', 'softmoe').
        top_k: Number of experts to select per token.
        capacity_factor: Multiplier for expert capacity calculation.
        add_noise: Whether to add noise to gating logits for exploration.
        noise_std: Standard deviation of gating noise.
        temperature: Temperature parameter for gating softmax.

        # Linear gating parameters
        use_bias: Whether to use bias in linear gating.

        # Cosine gating parameters
        embedding_dim: Dimension of expert embeddings for cosine gating.
        learnable_temperature: Whether temperature is learnable in cosine gating.

        # SoftMoE parameters
        num_slots: Number of input slots per expert in SoftMoE.

        # Load balancing parameters
        aux_loss_weight: Weight for auxiliary load balancing loss.
        z_loss_weight: Weight for router z-loss (entropy regularization).

    Example:
        ```python
        # Linear gating with load balancing
        linear_config = GatingConfig(
            gating_type='linear',
            top_k=2,
            capacity_factor=1.25,
            aux_loss_weight=0.01
        )

        # Cosine similarity gating
        cosine_config = GatingConfig(
            gating_type='cosine',
            top_k=1,
            embedding_dim=256,
            temperature=0.1
        )
        ```
    """
    gating_type: Literal['linear', 'cosine', 'softmoe'] = 'linear'
    top_k: int = 1
    capacity_factor: float = 1.25
    add_noise: bool = True
    noise_std: float = 1.0
    temperature: float = 1.0

    # Linear gating parameters
    use_bias: bool = False

    # Cosine gating parameters
    embedding_dim: int = 256
    learnable_temperature: bool = True

    # SoftMoE parameters
    num_slots: int = 4

    # Load balancing parameters
    aux_loss_weight: float = 0.01
    z_loss_weight: float = 1e-3


@dataclass
class MoEConfig:
    """
    Complete configuration for Mixture of Experts models.

    This dataclass combines expert and gating configurations with additional
    MoE-specific parameters to define complete MoE architectures.

    Args:
        num_experts: Total number of expert networks.
        expert_config: Configuration for individual expert networks.
        gating_config: Configuration for the gating network.

        # System-level parameters
        expert_parallel: Whether to use expert parallelism.
        jitter_noise: Standard deviation for expert capacity jittering.
        drop_tokens: Whether to drop tokens when expert capacity is exceeded.
        use_residual_connection: Whether to add residual connection for dropped tokens.

        # Training parameters
        train_capacity_factor: Capacity factor during training.
        eval_capacity_factor: Capacity factor during evaluation.

        # Advanced features
        hierarchical_routing: Whether to use hierarchical routing.
        num_routing_levels: Number of levels in hierarchical routing.
        routing_dtype: Data type for routing computations.

    Example:
        ```python
        # Standard MoE configuration
        moe_config = MoEConfig(
            num_experts=8,
            expert_config=ExpertConfig(expert_type='ffn', hidden_dim=768),
            gating_config=GatingConfig(gating_type='linear', top_k=2)
        )

        # Large-scale MoE with expert parallelism
        large_moe_config = MoEConfig(
            num_experts=64,
            expert_config=ExpertConfig(
                expert_type='ffn',
                hidden_dim=2048,
                intermediate_size=8192
            ),
            gating_config=GatingConfig(
                gating_type='cosine',
                top_k=1,
                aux_loss_weight=0.1
            ),
            expert_parallel=True,
            drop_tokens=True
        )
        ```
    """
    num_experts: int = 8
    expert_config: ExpertConfig = field(default_factory=ExpertConfig)
    gating_config: GatingConfig = field(default_factory=GatingConfig)

    # System-level parameters
    expert_parallel: bool = False
    jitter_noise: float = 0.01
    drop_tokens: bool = True
    use_residual_connection: bool = True

    # Training parameters
    train_capacity_factor: Optional[float] = None
    eval_capacity_factor: Optional[float] = None

    # Advanced features
    hierarchical_routing: bool = False
    num_routing_levels: int = 2
    routing_dtype: str = 'float32'

    def __post_init__(self):
        """Initialize derived parameters after dataclass creation."""
        if self.train_capacity_factor is None:
            self.train_capacity_factor = self.gating_config.capacity_factor

        if self.eval_capacity_factor is None:
            self.eval_capacity_factor = max(1.0, self.gating_config.capacity_factor * 0.8)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'num_experts': self.num_experts,
            'expert_config': self.expert_config.__dict__,
            'gating_config': self.gating_config.__dict__,
            'expert_parallel': self.expert_parallel,
            'jitter_noise': self.jitter_noise,
            'drop_tokens': self.drop_tokens,
            'use_residual_connection': self.use_residual_connection,
            'train_capacity_factor': self.train_capacity_factor,
            'eval_capacity_factor': self.eval_capacity_factor,
            'hierarchical_routing': self.hierarchical_routing,
            'num_routing_levels': self.num_routing_levels,
            'routing_dtype': self.routing_dtype
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MoEConfig':
        """Create configuration from dictionary."""
        expert_config = ExpertConfig(**config_dict.pop('expert_config', {}))
        gating_config = GatingConfig(**config_dict.pop('gating_config', {}))

        return cls(
            expert_config=expert_config,
            gating_config=gating_config,
            **config_dict
        )


# Pre-defined configurations for common use cases
DEFAULT_MOE_CONFIG = MoEConfig()

LARGE_MOE_CONFIG = MoEConfig(
    num_experts=16,
    expert_config=ExpertConfig(
        expert_type='ffn',
        hidden_dim=1024,
        intermediate_size=4096,
        dropout_rate=0.1
    ),
    gating_config=GatingConfig(
        gating_type='linear',
        top_k=2,
        capacity_factor=1.5,
        aux_loss_weight=0.1
    )
)

ATTENTION_MOE_CONFIG = MoEConfig(
    num_experts=8,
    expert_config=ExpertConfig(
        expert_type='attention',
        hidden_dim=768,
        num_heads=12,
        head_dim=64
    ),
    gating_config=GatingConfig(
        gating_type='cosine',
        top_k=1,
        embedding_dim=256,
        temperature=0.1
    )
)

VISION_MOE_CONFIG = MoEConfig(
    num_experts=6,
    expert_config=ExpertConfig(
        expert_type='conv2d',
        hidden_dim=256,
        filters=256,
        kernel_size=3
    ),
    gating_config=GatingConfig(
        gating_type='linear',
        top_k=1,
        capacity_factor=1.0,
        add_noise=False
    )
)