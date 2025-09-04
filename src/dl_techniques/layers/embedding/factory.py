"""
Embedding Layer Factory for dl_techniques Framework

Provides a centralized factory function for creating various embedding layers
with a unified interface, type safety, and comprehensive parameter validation.
This factory supports patch embeddings, learned positional embeddings, and various
forms of rotary position embeddings (RoPE).
"""

import keras
from typing import Dict, Any, Literal, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

from .continuous_rope_embedding import ContinuousRoPE
from .dual_rotary_position_embedding import DualRotaryPositionEmbedding
from .continuous_sin_cos_embedding import ContinuousSinCosEmbed
from .patch_embedding import PatchEmbedding1D, PatchEmbedding2D
from .positional_embedding import PositionalEmbedding
from .rotary_position_embedding import RotaryPositionEmbedding

# ---------------------------------------------------------------------
# Type definition for Embedding types
# ---------------------------------------------------------------------

EmbeddingType = Literal[
    'patch_1d',
    'patch_2d',
    'positional_learned',
    'rope',
    'dual_rope',
    'continuous_rope',
    'continuous_sincos'
]

# ---------------------------------------------------------------------
# Embedding layer registry mapping types to classes and parameter info
# ---------------------------------------------------------------------

EMBEDDING_REGISTRY: Dict[str, Dict[str, Any]] = {
    'patch_1d': {
        'class': PatchEmbedding1D,
        'description': '1D patch embedding for time series data with optional overlap.',
        'required_params': ['patch_size', 'embed_dim'],
        'optional_params': {
            'stride': None,
            'padding': 'causal',
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros'
        },
        'use_case': 'Tokenizing time series or other 1D sequential data for transformers.'
    },
    'patch_2d': {
        'class': PatchEmbedding2D,
        'description': '2D image to patch embedding layer for Vision Transformers (ViT).',
        'required_params': ['patch_size', 'embed_dim'],
        'optional_params': {
            'kernel_initializer': 'glorot_normal',
            'kernel_regularizer': None,
            'bias_initializer': 'zeros',
            'bias_regularizer': None,
            'activation': 'linear',
            'use_bias': True
        },
        'use_case': 'The first layer in ViT-style models to convert images into a sequence of patch embeddings.'
    },
    'positional_learned': {
        'class': PositionalEmbedding,
        'description': 'Adds learned, trainable positional embeddings to a sequence.',
        'required_params': ['max_seq_len', 'dim'],
        'optional_params': {
            'dropout': 0.0,
            'pos_initializer': 'truncated_normal',
            'scale': 0.02
        },
        'use_case': 'Standard positional encoding for transformer models where positions are learned from data.'
    },
    'rope': {
        'class': RotaryPositionEmbedding,
        'description': 'Standard Rotary Position Embedding (RoPE) for relative position encoding.',
        'required_params': ['head_dim', 'max_seq_len'],
        'optional_params': {
            'rope_theta': 10000.0,
            'rope_percentage': 0.5
        },
        'use_case': 'Injecting relative positional information into query/key vectors in attention mechanisms.'
    },
    'dual_rope': {
        'class': DualRotaryPositionEmbedding,
        'description': 'Dual RoPE for Gemma3-style models with separate global and local configurations.',
        'required_params': ['head_dim', 'max_seq_len'],
        'optional_params': {
            'global_theta_base': 1_000_000.0,
            'local_theta_base': 10_000.0
        },
        'use_case': 'Models using both global (full) and local (sliding window) attention patterns.'
    },
    'continuous_rope': {
        'class': ContinuousRoPE,
        'description': 'RoPE extended to handle continuous multi-dimensional coordinates.',
        'required_params': ['dim', 'ndim'],
        'optional_params': {
            'max_wavelength': 10000.0,
            'assert_positive': True
        },
        'use_case': 'Applying rotational position encoding to data with continuous spatial coordinates (e.g., 3D point clouds).'
    },
    'continuous_sincos': {
        'class': ContinuousSinCosEmbed,
        'description': 'Embeds continuous coordinates using fixed sine and cosine functions.',
        'required_params': ['dim', 'ndim'],
        'optional_params': {
            'max_wavelength': 10000.0,
            'assert_positive': True
        },
        'use_case': 'Creating fixed, smooth positional representations for continuous coordinate data.'
    }
}

# ---------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------

def get_embedding_info() -> Dict[str, Dict[str, Any]]:
    """
    Get comprehensive information about all available embedding layer types.

    Returns:
        Dict containing information about each embedding type, including:
        - description: Human-readable description
        - required_params: List of required parameter names
        - optional_params: Dict of optional parameters with defaults
        - use_case: Recommended use case for the embedding type

    Example:
        ```python
        info = get_embedding_info()
        for embed_type, details in info.items():
            print(f"{embed_type}: {details['description']}")
            print(f"  Required: {details['required_params']}")
            print(f"  Use case: {details['use_case']}")
        ```
    """
    return {embed_type: info.copy() for embed_type, info in EMBEDDING_REGISTRY.items()}

def validate_embedding_config(embedding_type: str, **kwargs: Any) -> None:
    """
    Validate embedding configuration parameters.

    Args:
        embedding_type: Type of embedding to validate.
        **kwargs: Parameters to validate against the embedding type's requirements.

    Raises:
        ValueError: If embedding_type is invalid, required parameters are missing,
                    or parameter values are out of their valid range.

    Example:
        ```python
        try:
            validate_embedding_config('rope', head_dim=64, max_seq_len=512)
            print("Configuration is valid.")
        except ValueError as e:
            print(f"Invalid configuration: {e}")
        ```
    """
    if embedding_type not in EMBEDDING_REGISTRY:
        available_types = list(EMBEDDING_REGISTRY.keys())
        raise ValueError(
            f"Unknown embedding type '{embedding_type}'. "
            f"Available types: {available_types}"
        )

    embed_info = EMBEDDING_REGISTRY[embedding_type]
    required_params = embed_info['required_params']

    # Check for missing required parameters
    missing_params = [param for param in required_params if param not in kwargs]
    if missing_params:
        raise ValueError(
            f"Required parameters missing for {embedding_type}: {missing_params}. "
            f"Required: {required_params}"
        )

    # --- Parameter Value Validations ---

    # Common positive integer checks
    positive_params = ['dim', 'embed_dim', 'head_dim', 'max_seq_len', 'patch_size', 'ndim']
    for param in positive_params:
        if param in kwargs and kwargs[param] is not None:
            value = kwargs[param]
            # Special case for patch_size in patch_2d
            if param == 'patch_size' and embedding_type == 'patch_2d':
                if isinstance(value, int):
                    if value <= 0:
                        raise ValueError(f"{param} must be a positive integer, got {value}")
                elif isinstance(value, (list, tuple)):
                     if len(value) != 2 or not all(isinstance(p, int) and p > 0 for p in value):
                         raise ValueError(f"{param} must be a tuple of 2 positive integers, got {value}")
                else:
                    raise TypeError(f"{param} must be an int or a tuple of 2 ints, got {type(value)}")
            elif isinstance(value, int) and value <= 0:
                raise ValueError(f"{param} must be positive, got {value}")


    # Type-specific validations
    if embedding_type == 'patch_1d':
        if 'stride' in kwargs and kwargs['stride'] is not None and kwargs['stride'] <= 0:
            raise ValueError(f"stride must be positive, got {kwargs['stride']}")
        if 'padding' in kwargs and kwargs['padding'] not in ['same', 'valid', 'causal']:
            raise ValueError(f"padding must be 'same', 'valid', or 'causal', got {kwargs['padding']}")

    if embedding_type == 'positional_learned':
        if 'dropout' in kwargs and not (0.0 <= kwargs['dropout'] <= 1.0):
            raise ValueError(f"dropout must be in [0, 1], got {kwargs['dropout']}")
        if 'scale' in kwargs and kwargs['scale'] <= 0:
            raise ValueError(f"scale must be positive, got {kwargs['scale']}")

    if embedding_type == 'rope':
        if 'rope_theta' in kwargs and kwargs['rope_theta'] <= 0:
            raise ValueError(f"rope_theta must be positive, got {kwargs['rope_theta']}")
        if 'rope_percentage' in kwargs and not (0.0 < kwargs['rope_percentage'] <= 1.0):
            raise ValueError(f"rope_percentage must be in (0, 1], got {kwargs['rope_percentage']}")

    if embedding_type == 'dual_rope':
        if 'head_dim' in kwargs and kwargs['head_dim'] % 2 != 0:
            raise ValueError(f"head_dim must be even for dual_rope, got {kwargs['head_dim']}")
        if 'global_theta_base' in kwargs and kwargs['global_theta_base'] <= 0:
            raise ValueError(f"global_theta_base must be positive, got {kwargs['global_theta_base']}")
        if 'local_theta_base' in kwargs and kwargs['local_theta_base'] <= 0:
            raise ValueError(f"local_theta_base must be positive, got {kwargs['local_theta_base']}")

    if embedding_type in ['continuous_rope', 'continuous_sincos']:
        if 'max_wavelength' in kwargs and kwargs['max_wavelength'] <= 0:
            raise ValueError(f"max_wavelength must be positive, got {kwargs['max_wavelength']}")


def create_embedding_layer(
    embedding_type: EmbeddingType,
    name: Optional[str] = None,
    **kwargs: Any
) -> keras.layers.Layer:
    """
    Factory function for creating embedding layers with a unified interface.

    This function provides a centralized way to create any embedding layer
    supported by dl_techniques, with comprehensive parameter validation.

    Args:
        embedding_type: Type of embedding layer to create.
        name: Optional name for the layer.
        **kwargs: Parameters specific to the embedding type. See individual layer
            documentation or use `get_embedding_info()` for details.

    Returns:
        A configured Keras embedding layer instance.

    Raises:
        ValueError: If embedding_type is invalid, parameters are missing or invalid.
        TypeError: If parameter types are incorrect for the layer.

    Example:
        ```python
        # Create a 2D patch embedding layer for a ViT
        patch_embed = create_embedding_layer(
            'patch_2d',
            patch_size=16,
            embed_dim=768,
            name='vit_patch_embed'
        )

        # Create a standard Rotary Position Embedding layer
        rope_embed = create_embedding_layer(
            'rope',
            head_dim=64,
            max_seq_len=2048,
            rope_percentage=0.25
        )
        ```
    """
    try:
        # Validate the provided configuration
        validate_embedding_config(embedding_type, **kwargs)

        # Get layer info and class from the registry
        embed_info = EMBEDDING_REGISTRY[embedding_type]
        embed_class = embed_info['class']

        # Prepare parameters, starting with defaults and overriding with user kwargs
        params = {}
        params.update(embed_info['optional_params'])
        params.update(kwargs)

        # Filter out any unknown parameters to avoid "Unrecognized keyword arguments"
        valid_param_names = set(embed_info['required_params']) | set(embed_info['optional_params'].keys())
        final_params = {key: val for key, val in params.items() if key in valid_param_names}

        # Add layer name if provided
        if name is not None:
            final_params['name'] = name

        # Log final parameters before creating the layer
        logger.info(f"Creating '{embedding_type}' embedding layer with parameters:")
        log_params = {**final_params} # create a copy for logging
        if name:
            log_params['name'] = name
        for param_name, param_value in sorted(log_params.items()):
            logger.info(f"  {param_name}: {param_value!r}")


        # Create the layer instance
        embedding_layer = embed_class(**final_params)

        logger.debug(f"Successfully created '{embedding_type}' layer: {embedding_layer.name}")
        return embedding_layer

    except (TypeError, ValueError) as e:
        # Provide enhanced error reporting with context
        embed_info = EMBEDDING_REGISTRY.get(embedding_type)
        if embed_info:
            required = embed_info.get('required_params', [])
            provided = list(kwargs.keys())
            class_name = embed_info.get('class', type(None)).__name__
            error_msg = (
                f"Failed to create '{embedding_type}' embedding layer ({class_name}).\n"
                f"  Required params: {required}\n"
                f"  Provided params: {provided}\n"
                f"  Check parameter compatibility and types. "
                f"Use get_embedding_info() for details.\n"
                f"  Original error: {e}"
            )
        else:
            error_msg = f"Failed to create embedding layer. Unknown type '{embedding_type}'. Original error: {e}"

        logger.error(error_msg)
        raise ValueError(error_msg) from e


def create_embedding_from_config(config: Dict[str, Any]) -> keras.layers.Layer:
    """
    Create an embedding layer from a configuration dictionary.

    Args:
        config: A dictionary containing a 'type' key specifying the embedding
                layer and other keys as its parameters.

    Returns:
        A configured Keras embedding layer instance.

    Raises:
        ValueError: If 'type' key is missing from config or config is not a dict.

    Example:
        ```python
        config = {
            'type': 'positional_learned',
            'max_seq_len': 1024,
            'dim': 512,
            'dropout': 0.1,
            'name': 'learned_pos_embed'
        }
        pos_embed = create_embedding_from_config(config)
        ```
    """
    if not isinstance(config, dict):
        raise ValueError(f"config must be a dictionary, got {type(config)}")

    if 'type' not in config:
        raise ValueError("Configuration dictionary must include a 'type' key.")

    config_copy = config.copy()
    embedding_type = config_copy.pop('type')

    logger.debug(f"Creating embedding from config - type: {embedding_type}, params: {list(config_copy.keys())}")

    return create_embedding_layer(embedding_type, **config_copy)