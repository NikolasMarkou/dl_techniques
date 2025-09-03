"""
Configuration classes for Mixture of Experts (MoE) models.

This module provides simplified configuration dataclasses for MoE components,
focused exclusively on FFN experts and leveraging the dl_techniques FFN factory.
"""

import keras
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any, Literal


@dataclass
class ExpertConfig:
    """
    Simplified configuration for FFN expert networks in MoE models.

    This dataclass defines FFN expert configuration by leveraging the existing
    dl_techniques FFN factory system, eliminating parameter duplication and
    ensuring consistency with the broader framework.

    Args:
        ffn_config: Dictionary containing FFN configuration that will be passed
            directly to the FFN factory's create_ffn_from_config() function.
            This should include 'type' and any FFN-specific parameters.
            Example: {"type": "swiglu", "d_model": 768, "ffn_expansion_factor": 4}

        use_bias: Whether to include bias terms in any additional linear layers
            (not part of the FFN itself). Defaults to True.
        kernel_initializer: Weight initialization strategy for any additional layers.
        bias_initializer: Bias initialization strategy for any additional layers.
        kernel_regularizer: Regularization applied to weights in additional layers.
        bias_regularizer: Regularization applied to biases in additional layers.

    Example:
        ```python
        # SwiGLU FFN expert configuration
        config = ExpertConfig(
            ffn_config={
                "type": "swiglu",
                "d_model": 768,
                "ffn_expansion_factor": 4,
                "dropout_rate": 0.1
            }
        )

        # Standard MLP expert configuration
        config = ExpertConfig(
            ffn_config={
                "type": "mlp",
                "hidden_dim": 2048,
                "output_dim": 768,
                "activation": "gelu",
                "dropout_rate": 0.1
            }
        )

        # GeGLU expert with custom initialization
        config = ExpertConfig(
            ffn_config={
                "type": "geglu",
                "hidden_dim": 3072,
                "output_dim": 768
            },
            kernel_initializer="he_normal"
        )
        ```

    Note:
        All FFN-specific validation and parameter handling is delegated to the
        FFN factory system, ensuring consistency across the framework and
        eliminating code duplication.
    """
    ffn_config: Dict[str, Any] = field(default_factory=dict)

    # Additional layer parameters (not part of FFN itself)
    use_bias: bool = True
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform'
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros'
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None
    bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None

    def __post_init__(self):
        """Validate FFN configuration after dataclass creation."""
        if not self.ffn_config:
            # Provide sensible default FFN configuration
            self.ffn_config = {
                "type": "mlp",
                "hidden_dim": 2048,
                "output_dim": 512
            }
        elif 'type' not in self.ffn_config:
            raise ValueError("ffn_config must contain 'type' field specifying FFN type")


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
    Complete configuration for Mixture of Experts models focused on FFN experts.

    This dataclass combines expert and gating configurations with MoE-specific
    parameters to define complete MoE architectures using FFN experts exclusively.

    Args:
        num_experts: Total number of FFN expert networks.
        expert_config: Configuration for FFN expert networks.
        gating_config: Configuration for the gating network.

        # System-level parameters
        jitter_noise: Standard deviation for expert capacity jittering.
        drop_tokens: Whether to drop tokens when expert capacity is exceeded.
        use_residual_connection: Whether to add residual connection for dropped tokens.

        # Training parameters
        train_capacity_factor: Capacity factor during training.
        eval_capacity_factor: Capacity factor during evaluation.

        # Advanced features
        routing_dtype: Data type for routing computations.

    Example:
        ```python
        # Standard FFN MoE with SwiGLU experts
        moe_config = MoEConfig(
            num_experts=8,
            expert_config=ExpertConfig(
                ffn_config={
                    "type": "swiglu",
                    "d_model": 768,
                    "ffn_expansion_factor": 4
                }
            ),
            gating_config=GatingConfig(gating_type='linear', top_k=2)
        )

        # MoE with standard MLP experts
        moe_config = MoEConfig(
            num_experts=16,
            expert_config=ExpertConfig(
                ffn_config={
                    "type": "mlp",
                    "hidden_dim": 2048,
                    "output_dim": 768,
                    "activation": "gelu"
                }
            )
        )
        ```
    """
    num_experts: int = 8
    expert_config: ExpertConfig = field(default_factory=ExpertConfig)
    gating_config: GatingConfig = field(default_factory=GatingConfig)

    # System-level parameters
    jitter_noise: float = 0.01
    drop_tokens: bool = True
    use_residual_connection: bool = True

    # Training parameters
    train_capacity_factor: Optional[float] = None
    eval_capacity_factor: Optional[float] = None

    # Advanced features
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
            'jitter_noise': self.jitter_noise,
            'drop_tokens': self.drop_tokens,
            'use_residual_connection': self.use_residual_connection,
            'train_capacity_factor': self.train_capacity_factor,
            'eval_capacity_factor': self.eval_capacity_factor,
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