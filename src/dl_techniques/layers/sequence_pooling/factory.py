"""
Sequence Pooling Layer Factory.

A registry-driven factory for creating sequence-pooling layers with a unified
interface, type safety, parameter validation, and detailed metadata. The factory
mirrors the canonical ``attention/`` sub-package template and exposes the three
pooling layers in this package through a single, consistent construction surface.

The factory supports three pooling mechanisms:
    - ``attention``: learnable, content-aware attention-weighted pooling.
    - ``weighted``: learnable, content-agnostic per-position weighted pooling.
    - ``sequence``: a unified facade dispatching 18 strategies and 4 aggregation
      methods (internally composing the attention and weighted poolers).

Key Features:
    - Type-safe pooling-layer creation with parameter validation.
    - Unified interface across all pooling mechanisms.
    - Detailed parameter documentation and error handling.
    - Support for both dictionary-based and direct configuration.
    - Integration with the dl_techniques logging system.
    - Complete compatibility with Keras 3 serialization.
"""

import keras
from typing import Any, Dict, List, Literal, Optional

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

from .attention_pooling import AttentionPooling
from .weighted_pooling import WeightedPooling
from .sequence_pooling import SequencePooling

# ---------------------------------------------------------------------
# Type Definitions
# ---------------------------------------------------------------------

SequencePoolingType = Literal['attention', 'weighted', 'sequence']
"""
Type alias for supported sequence-pooling mechanisms.

This literal type provides IDE autocompletion and type checking for valid
pooling-layer types supported by the factory.
"""

# ---------------------------------------------------------------------
# Sequence Pooling Layer Registry
# ---------------------------------------------------------------------

SEQUENCE_POOLING_REGISTRY: Dict[str, Dict[str, Any]] = {
    'attention': {
        'class': AttentionPooling,
        'description': (
            'Learnable, content-aware attention pooling. Each token is passed '
            'through a tanh projection and scored against a learnable context '
            'vector to produce per-token importance weights via softmax; the '
            'output is the weighted sum of the input tokens, optionally across '
            'multiple averaged attention heads.'
        ),
        'required_params': [],
        'optional_params': {
            'hidden_dim': 256,
            'num_heads': 1,
            'dropout_rate': 0.0,
            'use_bias': True,
            'temperature': 1.0,
            'kernel_initializer': 'glorot_uniform',
            'kernel_regularizer': None,
        },
        'use_case': (
            'Collapsing a sequence into a single summary vector when relevance '
            'is input-dependent: sentence/document embeddings, sequence '
            'classification heads, and any task where the model should '
            'dynamically focus on the most informative tokens.'
        ),
    },

    'weighted': {
        'class': WeightedPooling,
        'description': (
            'Learnable, content-agnostic per-position weighted pooling. A scalar '
            'learnable weight is assigned to each position up to max_seq_len; the '
            'weights for the current sequence length are softmax-normalised and '
            'used to compute a weighted sum, capturing positional importance '
            'patterns rather than input-dependent relevance.'
        ),
        'required_params': [],
        'optional_params': {
            'max_seq_len': 512,
            'dropout_rate': 0.0,
            'temperature': 1.0,
            'initializer': 'ones',
            'regularizer': None,
        },
        'use_case': (
            'A middle ground between simple mean pooling and full attention '
            'pooling for fixed-length or positionally-structured sequences, '
            'where token position carries consistent importance signal.'
        ),
    },

    'sequence': {
        'class': SequencePooling,
        'description': (
            'Unified, configurable pooling facade exposing 18 strategies '
            '(positional, statistical, learnable, top-k, and special) and 4 '
            'aggregation methods (concat, add, multiply, weighted_sum). One or '
            'more strategies are applied and their outputs combined, internally '
            'composing the attention and weighted poolers for learnable modes.'
        ),
        'required_params': [],
        'optional_params': {
            'strategy': 'mean',
            'exclude_positions': None,
            'aggregation_method': 'concat',
            'attention_hidden_dim': 256,
            'attention_num_heads': 1,
            'attention_dropout': 0.0,
            'weighted_max_seq_len': 512,
            'top_k': 10,
            'temperature': 1.0,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
        },
        'use_case': (
            'The default choice when bridging a sequence encoder (Transformer, '
            'LSTM) to a fixed-size downstream head, and when experimenting with '
            'or combining multiple pooling strategies behind a single interface.'
        ),
    },
}
"""
Registry of sequence-pooling layer implementations with metadata.

Each entry contains:
    - class: The actual layer class implementation.
    - description: Technical description of the pooling mechanism.
    - required_params: List of mandatory parameters for instantiation.
    - optional_params: Dict of optional parameters with default values.
    - use_case: Scenarios and applications where this pooling excels.
"""


# ---------------------------------------------------------------------
# Public API Functions
# ---------------------------------------------------------------------

def get_sequence_pooling_info() -> Dict[str, Dict[str, Any]]:
    """
    Retrieve metadata for all available sequence-pooling layer types.

    Provides per-type metadata including the technical description, parameter
    specifications, and use cases for every supported pooling mechanism.

    Returns:
        A dictionary mapping each pooling type to its metadata (description,
        required_params, optional_params, use_case). Each entry is a shallow
        copy so callers cannot mutate the registry.
    """
    return {
        pool_type: info.copy()
        for pool_type, info in SEQUENCE_POOLING_REGISTRY.items()
    }


def validate_sequence_pooling_config(
        pooling_type: str,
        **kwargs: Any
) -> None:
    """
    Validate sequence-pooling configuration parameters.

    Performs type-existence checking, required-parameter completeness, and light
    value-range validation on any numeric parameters that are present.

    Args:
        pooling_type: The pooling-layer type to validate against.
        **kwargs: Parameter dictionary to validate for the specified type.

    Raises:
        ValueError: If pooling_type is not supported, required parameters are
            missing, or a provided parameter value violates its constraint.
    """
    if pooling_type not in SEQUENCE_POOLING_REGISTRY:
        available_types = list(SEQUENCE_POOLING_REGISTRY.keys())
        raise ValueError(
            f"Unknown sequence pooling type '{pooling_type}'. "
            f"Available types: {available_types}"
        )

    info = SEQUENCE_POOLING_REGISTRY[pooling_type]
    required = info['required_params']
    missing = [p for p in required if p not in kwargs]
    if missing:
        raise ValueError(
            f"Required parameters for '{pooling_type}' are missing: {missing}. "
            f"Required: {required}, Provided: {list(kwargs.keys())}"
        )

    # Validate positive-integer parameters
    positive_int_params = [
        'hidden_dim', 'num_heads', 'max_seq_len', 'attention_hidden_dim',
        'attention_num_heads', 'weighted_max_seq_len', 'top_k'
    ]
    for param in positive_int_params:
        if param in kwargs and kwargs[param] <= 0:
            raise ValueError(
                f"Parameter '{param}' must be positive, got {kwargs[param]}"
            )

    # Validate positive-float parameters
    if 'temperature' in kwargs and kwargs['temperature'] <= 0:
        raise ValueError(
            f"Parameter 'temperature' must be positive, "
            f"got {kwargs['temperature']}"
        )

    # Validate probability/rate parameters (0.0 to 1.0)
    rate_params = ['dropout_rate', 'attention_dropout']
    for param in rate_params:
        if param in kwargs and not (0.0 <= kwargs[param] <= 1.0):
            raise ValueError(
                f"Parameter '{param}' must be between 0.0 and 1.0, "
                f"got {kwargs[param]}"
            )

    logger.debug(
        f"Validation successful for '{pooling_type}' with parameters: {kwargs}"
    )


def create_sequence_pooling_layer(
        pooling_type: SequencePoolingType,
        name: Optional[str] = None,
        **kwargs: Any
) -> keras.layers.Layer:
    """
    Factory function for creating sequence-pooling layers.

    Provides a centralized, type-safe way to instantiate any sequence-pooling
    layer in this package, with parameter validation, default-value handling,
    and detailed error reporting. Dispatch is pure registry lookup (no if/elif).

    Args:
        pooling_type: The type of pooling layer to create
            (``'attention'``, ``'weighted'``, or ``'sequence'``).
        name: Optional name for the layer instance.
        **kwargs: Type-specific parameters for the pooling layer. See
            ``get_sequence_pooling_info()`` for per-type parameter specs.

    Returns:
        A fully configured and instantiated sequence-pooling layer.

    Raises:
        ValueError: If pooling_type is invalid, required parameters are missing,
            parameter values are out of range, or layer construction fails.
        TypeError: If parameter types are incompatible with the target class.
    """
    try:
        # Validate configuration before proceeding
        validate_sequence_pooling_config(pooling_type, **kwargs)

        # Get layer information and class
        info = SEQUENCE_POOLING_REGISTRY[pooling_type]
        pool_class = info['class']

        # Merge user parameters with defaults (user wins)
        params = info['optional_params'].copy()
        params.update(kwargs)

        # Filter parameters to match the constructor signature
        valid_param_names = set(info['required_params']) | set(
            info['optional_params'].keys()
        )
        final_params = {
            k: v for k, v in params.items() if k in valid_param_names
        }

        # Add name if provided
        if name:
            final_params['name'] = name

        logger.info(
            f"Creating '{pooling_type}' sequence pooling layer "
            f"({pool_class.__name__}) with parameters: {final_params}"
        )

        # Instantiate the pooling layer
        return pool_class(**final_params)

    except (TypeError, ValueError) as e:
        # Provide detailed error context
        info = SEQUENCE_POOLING_REGISTRY.get(pooling_type)
        if info:
            class_name = info['class'].__name__
            error_msg = (
                f"Failed to create '{pooling_type}' sequence pooling layer "
                f"({class_name}). "
                f"Required parameters: {info['required_params']}. "
                f"Provided parameters: {list(kwargs.keys())}. "
                f"Please verify parameter compatibility. Original error: {e}"
            )
        else:
            error_msg = (
                f"Failed to create sequence pooling layer. "
                f"Unknown type '{pooling_type}'. Error: {e}"
            )

        logger.error(error_msg)
        raise ValueError(error_msg) from e


def create_sequence_pooling_from_config(
        config: Dict[str, Any]
) -> keras.layers.Layer:
    """
    Create a sequence-pooling layer from a configuration dictionary.

    Convenience function for instantiating pooling layers from dictionary-based
    configurations, useful for loading architectures from JSON/YAML files,
    hyperparameter optimization, and configuration-driven model building.

    Args:
        config: Configuration dictionary containing a ``'type'`` key specifying
            the pooling-layer type and additional keys for layer-specific
            parameters.

    Returns:
        Instantiated and configured sequence-pooling layer.

    Raises:
        ValueError: If config is missing the required ``'type'`` key.
        TypeError: If config parameter types are invalid.
    """
    config_copy = config.copy()
    try:
        pooling_type = config_copy.pop('type')
    except KeyError as e:
        available_keys = list(config.keys()) if config else []
        raise ValueError(
            f"Configuration dictionary must include a 'type' key specifying the "
            f"sequence pooling layer type. Available keys in config: "
            f"{available_keys}. "
            f"Valid types: {list(SEQUENCE_POOLING_REGISTRY.keys())}"
        ) from e

    logger.debug(f"Creating sequence pooling layer from config: {config}")
    return create_sequence_pooling_layer(pooling_type, **config_copy)


def list_sequence_pooling_types() -> List[str]:
    """
    Get a list of all supported sequence-pooling layer types.

    Returns:
        Alphabetically sorted list of supported pooling-layer types.
    """
    return sorted(list(SEQUENCE_POOLING_REGISTRY.keys()))
