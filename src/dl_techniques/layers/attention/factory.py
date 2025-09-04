"""
Attention Layer Factory for the dl_techniques Framework

Provides a centralized factory function for creating various attention layers
with a unified interface, type safety, and comprehensive parameter validation.
This factory allows for easy swapping of attention mechanisms within models
by simply changing the `attention_type` string.
"""

import keras
from typing import Dict, Any, Literal, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

from .adaptive_softmax_mha import AdaptiveMultiHeadAttention
from .anchor_attention import AnchorAttention
from .capsule_routing_attention import CapsuleRoutingSelfAttention
from .channel_attention import ChannelAttention
from .convolutional_block_attention import CBAM
from .cross_attention_perceiver_attention import PerceiverAttention
from .differential_attention import DifferentialMultiHeadAttention
from .fnet_fourier_transform import FNetFourierTransform
from .group_query_attention import GroupedQueryAttention
from .hopfield_attention import HopfieldAttention
from .mobile_mqa import MobileMQA
from .multi_head_attention import MultiHeadAttention
from .non_local_attention import NonLocalAttention
from .shared_weights_cross_attention import SharedWeightsCrossAttention
from .spatial_attention import SpatialAttention
from .window_attention import WindowAttention

# ---------------------------------------------------------------------
# Type definition for Attention types
# ---------------------------------------------------------------------

AttentionType = Literal[
    'adaptive_multi_head',
    'anchor',
    'capsule_routing',
    'channel',
    'cbam',
    'perceiver',
    'differential',
    'fnet',
    'group_query',
    'hopfield',
    'mobile_mqa',
    'multi_head',
    'non_local',
    'shared_weights_cross',
    'spatial',
    'window'
]

# ---------------------------------------------------------------------
# Attention layer registry mapping types to classes and parameter info
# ---------------------------------------------------------------------

ATTENTION_REGISTRY: Dict[str, Dict[str, Any]] = {
    'adaptive_multi_head': {
        'class': AdaptiveMultiHeadAttention,
        'description': 'Multi-Head Attention with adaptive temperature softmax.',
        'required_params': ['num_heads', 'key_dim'],
        'optional_params': {
            'value_dim': None, 'dropout': 0.0, 'use_bias': True, 'output_shape': None,
            'attention_axes': None, 'flash_attention': None, 'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros', 'kernel_regularizer': None, 'bias_regularizer': None,
            'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None,
            'seed': None, 'min_temp': 0.1, 'max_temp': 1.0, 'entropy_threshold': 0.5,
            'polynomial_coeffs': None
        },
        'use_case': 'Transformer models where attention sharpness needs to adapt to sequence length.'
    },
    'anchor': {
        'class': AnchorAttention,
        'description': 'Hierarchical attention with anchor tokens for memory efficiency.',
        'required_params': ['dim'],
        'optional_params': {
            'num_heads': 8, 'dropout': 0.0, 'use_bias': True, 'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros', 'kernel_regularizer': None, 'bias_regularizer': None
        },
        'use_case': 'Long-sequence models where full self-attention is too costly.'
    },
    'capsule_routing': {
        'class': CapsuleRoutingSelfAttention,
        'description': 'Self-attention with capsule network-style dynamic routing.',
        'required_params': ['num_heads'],
        'optional_params': {
            'key_dim': None, 'value_dim': None, 'dropout': 0.0, 'use_bias': True,
            'kernel_initializer': 'glorot_uniform', 'bias_initializer': 'zeros',
            'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None,
            'routing_iterations': 3, 'use_vertical_routing': True, 'use_horizontal_routing': True,
            'use_positional_routing': True, 'epsilon': 1e-8
        },
        'use_case': 'Experimental models aiming for better contextualization through routing.'
    },
    'channel': {
        'class': ChannelAttention,
        'description': 'Channel attention module from CBAM.',
        'required_params': ['channels'],
        'optional_params': {
            'ratio': 8, 'kernel_initializer': 'glorot_uniform', 'kernel_regularizer': None,
            'use_bias': False
        },
        'use_case': 'CNN architectures to recalibrate channel-wise feature responses.'
    },
    'cbam': {
        'class': CBAM,
        'description': 'Convolutional Block Attention Module (Channel + Spatial).',
        'required_params': ['channels'],
        'optional_params': {
            'ratio': 8, 'kernel_size': 7, 'channel_kernel_initializer': 'glorot_uniform',
            'spatial_kernel_initializer': 'glorot_uniform', 'channel_kernel_regularizer': None,
            'spatial_kernel_regularizer': None, 'channel_use_bias': False, 'spatial_use_bias': True
        },
        'use_case': 'Plug-and-play attention module for any CNN to refine feature maps.'
    },
    'perceiver': {
        'class': PerceiverAttention,
        'description': 'Cross-attention mechanism from the Perceiver architecture.',
        'required_params': ['dim'],
        'optional_params': {
            'num_heads': 8, 'dropout': 0.0, 'use_bias': True, 'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros', 'kernel_regularizer': None, 'bias_regularizer': None
        },
        'use_case': 'Cross-modal attention (e.g., text to image) and latent bottleneck models.'
    },
    'differential': {
        'class': DifferentialMultiHeadAttention,
        'description': 'Dual multi-head attention to amplify signal and cancel noise.',
        'required_params': ['dim', 'num_heads', 'head_dim'],
        'optional_params': {
            'dropout': 0.0, 'attention_dropout': 0.0, 'lambda_init': 0.8,
            'kernel_initializer': 'glorot_uniform', 'kernel_regularizer': None,
            'bias_initializer': 'zeros', 'bias_regularizer': None, 'activity_regularizer': None
        },
        'use_case': 'Transformers requiring improved focus and reduced hallucination.'
    },
    'fnet': {
        'class': FNetFourierTransform,
        'description': 'Parameter-free token mixing using Fourier Transforms instead of attention.',
        'required_params': [],
        'optional_params': {
            'implementation': 'matrix', 'normalize_dft': True, 'epsilon': 1e-12
        },
        'use_case': 'Efficient replacement for self-attention in sequence models (e.g., FNet, F-GPT).'
    },
    'group_query': {
        'class': GroupedQueryAttention,
        'description': 'Grouped Query Attention with shared K/V heads for efficiency.',
        'required_params': ['d_model', 'n_head', 'n_kv_head'],
        'optional_params': {
            'max_seq_len': 2048, 'dropout_rate': 0.0, 'rope_percentage': 1.0,
            'rope_theta': 10000.0, 'use_bias': False, 'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros', 'kernel_regularizer': None, 'bias_regularizer': None
        },
        'use_case': 'Large language models where K/V cache size is a bottleneck.'
    },
    'hopfield': {
        'class': HopfieldAttention,
        'description': 'Modern Hopfield Network with iterative updates for pattern retrieval.',
        'required_params': ['num_heads', 'key_dim'],
        'optional_params': {
            'value_dim': None, 'dropout_rate': 0.0, 'use_bias': True,
            'kernel_initializer': 'glorot_uniform', 'bias_initializer': 'zeros',
            'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None,
            'normalize_patterns': True, 'update_steps_max': 0, 'update_steps_eps': 1e-4
        },
        'use_case': 'Associative memory tasks; setting update_steps_max=0 mimics standard attention.'
    },
    'mobile_mqa': {
        'class': MobileMQA,
        'description': 'Mobile-optimized Multi-Query Attention for 4D vision tensors.',
        'required_params': ['dim'],
        'optional_params': {
            'num_heads': 8, 'use_downsampling': False, 'kernel_initializer': 'he_normal',
            'kernel_regularizer': None
        },
        'use_case': 'Efficient attention in vision models for mobile and edge devices.'
    },
    'multi_head': {
        'class': MultiHeadAttention,
        'description': 'Standard Multi-Head Self-Attention.',
        'required_params': ['embed_dim'],
        'optional_params': {
            'num_heads': 8, 'dropout_rate': 0.0, 'kernel_initializer': 'he_normal',
            'kernel_regularizer': None, 'use_bias': False
        },
        'use_case': 'General-purpose self-attention in vision and sequence models.'
    },
    'non_local': {
        'class': NonLocalAttention,
        'description': 'Non-local attention block for capturing long-range dependencies in CNNs.',
        'required_params': ['attention_channels'],
        'optional_params': {
            'kernel_size': (7, 7), 'use_bias': False, 'normalization': 'batch',
            'intermediate_activation': 'relu', 'output_activation': 'linear', 'output_channels': -1,
            'dropout_rate': 0.0, 'attention_mode': 'gaussian', 'kernel_initializer': 'glorot_normal',
            'bias_initializer': 'zeros', 'kernel_regularizer': None, 'bias_regularizer': None,
            'activity_regularizer': None
        },
        'use_case': 'Augmenting CNNs with global context reasoning for vision tasks.'
    },
    'shared_weights_cross': {
        'class': SharedWeightsCrossAttention,
        'description': 'Cross-attention between modalities with shared weights for efficiency.',
        'required_params': ['dim'],
        'optional_params': {
            'num_heads': 8, 'dropout_rate': 0.0, 'use_bias': True,
            'kernel_initializer': 'glorot_uniform', 'bias_initializer': 'zeros',
            'kernel_regularizer': None, 'bias_regularizer': None
        },
        'use_case': 'Multi-modal learning where different data types exchange information.'
    },
    'spatial': {
        'class': SpatialAttention,
        'description': 'Spatial attention module from CBAM.',
        'required_params': [],
        'optional_params': {
            'kernel_size': 7, 'kernel_initializer': 'glorot_uniform', 'kernel_regularizer': None,
            'use_bias': True
        },
        'use_case': 'CNN architectures to highlight spatially significant feature regions.'
    },
    'window': {
        'class': WindowAttention,
        'description': 'Windowed Multi-Head Attention from Swin Transformer.',
        'required_params': ['dim', 'window_size', 'num_heads'],
        'optional_params': {
            'qkv_bias': True, 'qk_scale': None, 'attn_dropout_rate': 0.0,
            'proj_dropout_rate': 0.0, 'proj_bias': True, 'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros', 'kernel_regularizer': None, 'bias_regularizer': None
        },
        'use_case': 'Vision transformers like Swin, providing efficient local attention.'
    }
}


# ---------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------

def get_attention_info() -> Dict[str, Dict[str, Any]]:
    """
    Get comprehensive information about all available attention layer types.

    Returns:
        Dict containing details for each attention type, including description,
        required/optional parameters, and primary use case.

    Example:
        ```python
        info = get_attention_info()
        for attn_type, details in info.items():
            print(f"{attn_type}: {details['description']}")
            print(f"  Required: {details['required_params']}")
        ```
    """
    return {attn_type: info.copy() for attn_type, info in ATTENTION_REGISTRY.items()}


def validate_attention_config(attention_type: str, **kwargs: Any) -> None:
    """
    Validate attention layer configuration parameters.

    Args:
        attention_type: The type of attention layer to validate.
        **kwargs: Parameters to validate against the layer's requirements.

    Raises:
        ValueError: If `attention_type` is invalid, required parameters are
                    missing, or common parameters have invalid values.
    """
    if attention_type not in ATTENTION_REGISTRY:
        available_types = list(ATTENTION_REGISTRY.keys())
        raise ValueError(
            f"Unknown attention type '{attention_type}'. "
            f"Available types: {available_types}"
        )

    info = ATTENTION_REGISTRY[attention_type]
    required = info['required_params']
    missing = [p for p in required if p not in kwargs]
    if missing:
        raise ValueError(
            f"Required parameters for '{attention_type}' are missing: {missing}. "
            f"Required: {required}"
        )

    # Common parameter validation
    dim_params = ['dim', 'embed_dim', 'd_model', 'channels']
    for p in dim_params:
        if p in kwargs and kwargs[p] <= 0:
            raise ValueError(f"Parameter '{p}' must be positive, got {kwargs[p]}")

    head_params = ['num_heads', 'n_head']
    for p in head_params:
        if p in kwargs and kwargs[p] <= 0:
            raise ValueError(f"Parameter '{p}' must be positive, got {kwargs[p]}")

    dropout_params = ['dropout', 'dropout_rate', 'attention_dropout', 'attn_dropout_rate', 'proj_dropout_rate']
    for p in dropout_params:
        if p in kwargs and not (0.0 <= kwargs[p] <= 1.0):
            raise ValueError(f"Parameter '{p}' must be between 0.0 and 1.0, got {kwargs[p]}")


def create_attention_layer(
        attention_type: AttentionType,
        name: Optional[str] = None,
        **kwargs: Any
) -> keras.layers.Layer:
    """
    Factory function for creating attention layers with a unified interface.

    This function provides a centralized way to instantiate any attention layer
    supported by the framework, with comprehensive parameter validation.

    Args:
        attention_type: The type of attention layer to create.
        name: Optional name for the created layer.
        **kwargs: Parameters specific to the chosen attention type. Use
                  `get_attention_info()` to see requirements for each type.

    Returns:
        An instantiated and configured Keras attention layer.

    Raises:
        ValueError: If `attention_type` is invalid, required parameters are
                    missing, or parameter values are incorrect.

    Example:
        ```python
        # Create a standard multi-head attention layer
        mha = create_attention_layer('multi_head', embed_dim=256, num_heads=8)

        # Create a CBAM block for a CNN
        cbam = create_attention_layer('cbam', channels=128, ratio=16)

        # Create a Grouped Query Attention layer for an LLM
        gqa = create_attention_layer(
            'group_query',
            d_model=1024,
            n_head=16,
            n_kv_head=4,
            name='gqa_block_1'
        )
        ```
    """
    try:
        validate_attention_config(attention_type, **kwargs)
        info = ATTENTION_REGISTRY[attention_type]
        attn_class = info['class']

        # Prepare parameters by merging defaults with user-provided kwargs
        params = info['optional_params'].copy()
        params.update(kwargs)

        # Filter out any kwargs not expected by the class constructor
        valid_param_names = set(info['required_params']) | set(info['optional_params'].keys())
        final_params = {k: v for k, v in params.items() if k in valid_param_names}

        if name:
            final_params['name'] = name

        logger.info(f"Creating '{attention_type}' layer with parameters: {final_params}")
        return attn_class(**final_params)

    except (TypeError, ValueError) as e:
        info = ATTENTION_REGISTRY.get(attention_type)
        if info:
            class_name = info['class'].__name__
            error_msg = (
                f"Failed to create '{attention_type}' layer ({class_name}). "
                f"Required: {info['required_params']}. "
                f"Provided: {list(kwargs.keys())}. "
                f"Please check parameter compatibility. Original error: {e}"
            )
        else:
            error_msg = f"Failed to create layer. Unknown type '{attention_type}'. Error: {e}"

        logger.error(error_msg)
        raise ValueError(error_msg) from e


def create_attention_from_config(config: Dict[str, Any]) -> keras.layers.Layer:
    """
    Create an attention layer from a configuration dictionary.

    Args:
        config: A dictionary containing a 'type' key specifying the attention
                layer and other keys for its parameters.

    Returns:
        An instantiated and configured Keras attention layer.

    Raises:
        ValueError: If the config is not a dict or is missing the 'type' key.

    Example:
        ```python
        attn_config = {
            'type': 'window',
            'dim': 96,
            'window_size': 7,
            'num_heads': 4,
            'name': 'swin_attn_block_2'
        }
        window_attn = create_attention_from_config(attn_config)
        ```
    """
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dictionary, got {type(config)}")
    if 'type' not in config:
        raise ValueError("Configuration dictionary must include a 'type' key.")

    config_copy = config.copy()
    attention_type = config_copy.pop('type')
    return create_attention_layer(attention_type, **config_copy)