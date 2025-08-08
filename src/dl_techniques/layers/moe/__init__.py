"""
Mixture of Experts (MoE) module for dl_techniques.

This module provides a comprehensive implementation of modern Mixture of Experts
architectures, supporting various expert types, routing mechanisms, and training
strategies for building sparse, scalable neural networks.

Key Components:
    - MixtureOfExperts: Main MoE layer combining experts and gating
    - Expert networks: FFN, Attention, and Convolutional experts
    - Gating networks: Linear, Cosine, and SoftMoE routing mechanisms
    - Configuration system: Comprehensive configuration classes
    - Load balancing: Auxiliary losses for stable training

Usage:
    ```python
    from dl_techniques.layers.moe import MixtureOfExperts, MoEConfig, create_ffn_moe

    # Quick creation with factory function
    moe_layer = create_ffn_moe(
        num_experts=8,
        hidden_dim=768,
        top_k=2
    )

    # Advanced configuration
    config = MoEConfig(
        num_experts=16,
        expert_config=ExpertConfig(expert_type='ffn', hidden_dim=1024),
        gating_config=GatingConfig(gating_type='cosine', top_k=1)
    )
    moe_layer = MixtureOfExperts(config)
    ```

Example Applications:
    - Switch Transformer-style language models
    - Vision Transformer with convolutional MoE
    - Mixture-of-Attention (MoA) for specialized attention patterns
    - Large-scale multimodal models with expert specialization
"""

# Configuration classes
from .config import (
    ExpertConfig,
    GatingConfig,
    MoEConfig,
    DEFAULT_MOE_CONFIG,
    LARGE_MOE_CONFIG,
    ATTENTION_MOE_CONFIG,
    VISION_MOE_CONFIG
)

# Expert implementations
from .experts import (
    BaseExpert,
    FFNExpert,
    AttentionExpert,
    Conv2DExpert,
    create_expert
)

# Gating implementations
from .gating import (
    BaseGating,
    LinearGating,
    CosineGating,
    SoftMoEGating,
    create_gating,
    compute_auxiliary_loss,
    compute_z_loss
)

# Main MoE layer and convenience functions
from .layer import (
    MixtureOfExperts,
    create_ffn_moe,
    create_attention_moe,
    create_conv_moe
)


def get_moe_info() -> str:
    """
    Get information about the MoE module.

    Returns:
        String containing module information and usage examples.
    """
    info = f"""
    dl_techniques MoE Module v{__version__}

    A comprehensive implementation of modern Mixture of Experts architectures.

    Supported Expert Types:
    - FFN (Feed-Forward Network): Standard transformer FFN blocks
    - Attention: Multi-head attention experts for MoA architectures
    - Conv2D: Convolutional experts for vision models

    Supported Gating Types:
    - Linear: Standard linear gating with optional noise
    - Cosine: Cosine similarity-based routing in hypersphere
    - SoftMoE: Soft token routing without dropping

    Key Features:
    - Load balancing with auxiliary losses
    - Expert capacity management
    - Token dropping and residual connections
    - Hierarchical routing support
    - Multi-modal and multi-task routing
    - Expert parallelism ready

    Quick Start:
    ```python
    from dl_techniques.layers.moe import create_ffn_moe

    moe = create_ffn_moe(num_experts=8, hidden_dim=768, top_k=2)
    ```
    """
    return info


# Convenience function for getting pre-configured MoE layers
def get_preset_moe(preset_name: str, **overrides) -> MixtureOfExperts:
    """
    Get a pre-configured MoE layer based on common use cases.

    Args:
        preset_name: Name of the preset configuration.
            Options: 'default', 'large', 'attention', 'vision'
        **overrides: Configuration overrides for customization.

    Returns:
        Configured MixtureOfExperts layer.

    Raises:
        ValueError: If preset_name is not recognized.

    Example:
        ```python
        # Get default configuration with custom expert count
        moe = get_preset_moe('default', num_experts=16)

        # Get large configuration with different routing
        moe = get_preset_moe('large', gating_type='cosine')
        ```
    """
    preset_configs = {
        'default': DEFAULT_MOE_CONFIG,
        'large': LARGE_MOE_CONFIG,
        'attention': ATTENTION_MOE_CONFIG,
        'vision': VISION_MOE_CONFIG
    }

    if preset_name not in preset_configs:
        available = list(preset_configs.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")

    # Get base configuration
    base_config = preset_configs[preset_name]

    # Apply overrides if provided
    if overrides:
        config_dict = base_config.to_dict()

        # Handle nested overrides
        for key, value in overrides.items():
            if key == 'expert_config' and isinstance(value, dict):
                config_dict['expert_config'].update(value)
            elif key == 'gating_config' and isinstance(value, dict):
                config_dict['gating_config'].update(value)
            else:
                config_dict[key] = value

        config = MoEConfig.from_dict(config_dict)
    else:
        config = base_config

    return MixtureOfExperts(config=config)


# Module validation function
def validate_moe_installation():
    """
    Validate that the MoE module is correctly installed and functional.

    This function performs basic validation of the MoE components to ensure
    they can be imported and instantiated correctly.

    Raises:
        ImportError: If required dependencies are missing.
        RuntimeError: If components cannot be instantiated.
    """
    try:
        # Test configuration creation
        config = MoEConfig()
        assert config.num_experts == 8, "Default configuration validation failed"

        # Test expert creation
        expert = create_expert('ffn', hidden_dim=64)
        assert isinstance(expert, FFNExpert), "Expert creation validation failed"

        # Test gating creation
        gating = create_gating('linear', num_experts=4)
        assert isinstance(gating, LinearGating), "Gating creation validation failed"

        # Test MoE layer creation
        moe = create_ffn_moe(num_experts=4, hidden_dim=64)
        assert isinstance(moe, MixtureOfExperts), "MoE layer creation validation failed"

        print("✓ MoE module validation passed successfully")

    except Exception as e:
        raise RuntimeError(f"MoE module validation failed: {str(e)}") from e


# Version and metadata
__version__ = '1.0.0'
__author__ = 'dl_techniques'

# All public exports
__all__ = [
    # Configuration
    'ExpertConfig',
    'GatingConfig',
    'MoEConfig',
    'DEFAULT_MOE_CONFIG',
    'LARGE_MOE_CONFIG',
    'ATTENTION_MOE_CONFIG',
    'VISION_MOE_CONFIG',

    # Expert networks
    'BaseExpert',
    'FFNExpert',
    'AttentionExpert',
    'Conv2DExpert',
    'create_expert',

    # Gating networks
    'BaseGating',
    'LinearGating',
    'CosineGating',
    'SoftMoEGating',
    'create_gating',
    'compute_auxiliary_loss',
    'compute_z_loss',

    # Main MoE layer
    'MixtureOfExperts',
    'create_ffn_moe',
    'create_attention_moe',
    'create_conv_moe',

    'get_preset_moe'
]

if __name__ == "__main__":
    # Run validation when module is executed directly
    validate_moe_installation()
    print(get_moe_info())