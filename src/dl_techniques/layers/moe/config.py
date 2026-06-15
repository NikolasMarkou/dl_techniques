"""
Configuration classes for Mixture of Experts (MoE) models.

This module provides simplified configuration dataclasses for MoE components,
focused exclusively on FFN experts and leveraging the dl_techniques FFN factory.
"""

import keras
from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any, Literal

# ---------------------------------------------------------------------

@dataclass
class ExpertConfig:
    """
    Simplified configuration for FFN expert networks in MoE models.

    This dataclass defines FFN expert configuration by leveraging the existing
    dl_techniques FFN factory system, eliminating parameter duplication and
    ensuring consistency with the broader framework.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────┐
        │    ExpertConfig     │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  ffn_config (dict)  │──▶ FFN Factory
        │  ├─ type            │    (create_ffn_from_config)
        │  ├─ output_dim      │
        │  └─ ...params       │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Additional Layers  │
        │  (use_bias, init,   │
        │   regularizers)     │
        └─────────────────────┘

    :param ffn_config: Dictionary containing FFN configuration that will be passed
        directly to the FFN factory's create_ffn_from_config() function.
        This should include 'type' and any FFN-specific parameters.
    :type ffn_config: Dict[str, Any]
    :param use_bias: Whether to include bias terms in any additional linear layers
        (not part of the FFN itself).
    :type use_bias: bool
    :param kernel_initializer: Weight initialization strategy for any additional layers.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Bias initialization strategy for any additional layers.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularization applied to weights in additional layers.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Regularization applied to biases in additional layers.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    """
    ffn_config: Dict[str, Any] = field(default_factory=dict)

    # Additional layer parameters (not part of FFN itself)
    use_bias: bool = True
    kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform'
    bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros'
    kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None
    bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None

    # Optional normalization for experts (instantiated via the norms factory).
    # When ``norm_type`` is set, ``pre_norm`` defaults to True and ``post_norm``
    # to False. Both can be toggled independently.
    norm_type: Optional[str] = None
    norm_config: Dict[str, Any] = field(default_factory=dict)
    pre_norm: bool = True
    post_norm: bool = False

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

# ---------------------------------------------------------------------

@dataclass
class GatingConfig:
    """
    Configuration for MoE gating networks (routers).

    This dataclass defines the routing mechanism for MoE models, supporting
    various gating strategies and load balancing techniques.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────┐
        │   Input Tokens      │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │   Gating Network    │
        │   (linear │ cosine  │
        │    │ softmoe)       │
        └──────────┬──────────┘
                   │
            ┌──────┴──────┐
            ▼             ▼
        ┌────────┐  ┌────────┐
        │ top-k  │  │ aux    │
        │ routes │  │ losses │
        └────────┘  └────────┘

    :param gating_type: Type of gating mechanism ('linear', 'cosine', 'softmoe').
    :type gating_type: Literal['linear', 'cosine', 'softmoe']
    :param top_k: Number of experts to select per token.
    :type top_k: int
    :param capacity_factor: Multiplier for expert capacity calculation.
    :type capacity_factor: float
    :param add_noise: Whether to add noise to gating logits for exploration.
    :type add_noise: bool
    :param noise_std: Standard deviation of gating noise.
    :type noise_std: float
    :param temperature: Temperature parameter for gating softmax.
    :type temperature: float
    :param use_bias: Whether to use bias in linear gating.
    :type use_bias: bool
    :param embedding_dim: Dimension of expert embeddings for cosine gating.
    :type embedding_dim: int
    :param learnable_temperature: Whether temperature is learnable in cosine gating.
    :type learnable_temperature: bool
    :param num_slots: Number of input slots per expert in SoftMoE.
    :type num_slots: int
    :param aux_loss_weight: Weight for auxiliary load balancing loss.
    :type aux_loss_weight: float
    :param z_loss_weight: Weight for router z-loss (entropy regularization).
    :type z_loss_weight: float
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

    # Optional pre-gating normalization via the norms factory.
    norm_type: Optional[str] = None
    norm_config: Dict[str, Any] = field(default_factory=dict)

# ---------------------------------------------------------------------

@dataclass
class MoEConfig:
    """
    Complete configuration for Mixture of Experts models focused on FFN experts.

    This dataclass combines expert and gating configurations with MoE-specific
    parameters to define complete MoE architectures using FFN experts exclusively.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────┐
        │           MoEConfig              │
        │                                  │
        │  ┌────────────┐ ┌────────────┐   │
        │  │ExpertConfig│ │GatingConfig│   │
        │  └─────┬──────┘ └─────┬──────┘   │
        │        │              │          │
        │        ▼              ▼          │
        │  ┌───────────────────────────┐   │
        │  │    MoE Layer Assembly     │   │
        │  │  (N experts + router)     │   │
        │  └───────────────────────────┘   │
        └──────────────────────────────────┘

    :param num_experts: Total number of FFN expert networks.
    :type num_experts: int
    :param expert_config: Configuration for FFN expert networks.
    :type expert_config: ExpertConfig
    :param gating_config: Configuration for the gating network.
    :type gating_config: GatingConfig
    :param jitter_noise: Standard deviation for uniform noise added to the
        gating input during training. Note: ``LinearGating`` also injects
        learned-scale Gaussian noise to the gating *logits* when
        ``add_noise=True``; the two sources stack. Set ``jitter_noise=0`` to
        rely solely on the gating-level noise.
    :type jitter_noise: float
    :param drop_tokens: Reserved for future capacity-based dispatch (the
        current hard-routing kernel is dense and does not drop tokens).
    :type drop_tokens: bool
    :param use_residual_connection: Reserved for future capacity-based
        dispatch. Has no effect in the current dense kernel.
    :type use_residual_connection: bool
    :param routing_dtype: Data type for routing computations.
    :type routing_dtype: str
    """
    num_experts: int = 8
    expert_config: ExpertConfig = field(default_factory=ExpertConfig)
    gating_config: GatingConfig = field(default_factory=GatingConfig)

    # System-level parameters
    jitter_noise: float = 0.01
    drop_tokens: bool = True
    use_residual_connection: bool = True

    # Advanced features
    routing_dtype: str = 'float32'

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        # Serialize ExpertConfig, handling Keras initializer/regularizer objects
        expert_dict = {}
        for k, v in self.expert_config.__dict__.items():
            if isinstance(v, keras.initializers.Initializer):
                expert_dict[k] = keras.initializers.serialize(v)
            elif isinstance(v, keras.regularizers.Regularizer):
                expert_dict[k] = keras.regularizers.serialize(v)
            else:
                expert_dict[k] = v

        return {
            'num_experts': self.num_experts,
            'expert_config': expert_dict,
            'gating_config': dict(self.gating_config.__dict__),
            'jitter_noise': self.jitter_noise,
            'drop_tokens': self.drop_tokens,
            'use_residual_connection': self.use_residual_connection,
            'routing_dtype': self.routing_dtype,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MoEConfig':
        """Create configuration from dictionary (does not mutate input)."""
        config_dict = dict(config_dict)  # shallow copy to avoid mutating caller's dict

        # Drop legacy keys that were removed (kept for backward-compat reads).
        for legacy_key in ('train_capacity_factor', 'eval_capacity_factor'):
            config_dict.pop(legacy_key, None)

        # Deserialize ExpertConfig, handling serialized Keras objects
        expert_raw = config_dict.pop('expert_config', {})
        for k in ('kernel_initializer', 'bias_initializer'):
            if k in expert_raw and isinstance(expert_raw[k], dict):
                expert_raw[k] = keras.initializers.deserialize(expert_raw[k])
        for k in ('kernel_regularizer', 'bias_regularizer'):
            if k in expert_raw and isinstance(expert_raw[k], dict):
                expert_raw[k] = keras.regularizers.deserialize(expert_raw[k])
        expert_config = ExpertConfig(**expert_raw)

        gating_config = GatingConfig(**config_dict.pop('gating_config', {}))

        return cls(
            expert_config=expert_config,
            gating_config=gating_config,
            **config_dict
        )

# ---------------------------------------------------------------------
