"""
Holonomic Field Layers for Deep Learning.

This package provides a comprehensive implementation of field-based neural network
layers inspired by gauge theory and differential geometry. These layers implement
the key concepts from Holonomic AI:

1. **Field Embeddings**: Represent tokens as fields with curvature, not point vectors
2. **Connection Layers**: Compute gauge connections for parallel transport
3. **Parallel Transport**: Move information along paths respecting geometry
4. **Holonomy Layers**: Compute gauge-invariant path integrals
5. **Gauge-Invariant Attention**: Attention that respects manifold structure
6. **Manifold Stress**: Detect anomalies through geometric inconsistency
7. **Holonomic Transformer**: Complete transformer with all field components

Key Benefits:
- Natural robustness to adversarial perturbations
- Built-in anomaly and poison detection
- Richer geometric representations
- Gauge-invariant processing

Mathematical Foundation:
These layers are based on concepts from differential geometry and gauge theory:
- Fiber bundles and connections
- Parallel transport and holonomy
- Curvature and torsion
- Gauge invariance

Example Usage:
    >>> from dl_techniques.layers.geometric.fields import (
    ...     create_field_layer,
    ...     HolonomicTransformerLayer,
    ...     FieldEmbedding
    ... )
    >>>
    >>> # Create a complete holonomic transformer layer
    >>> layer = create_field_layer(
    ...     'holonomic_transformer',
    ...     hidden_dim=256,
    ...     num_heads=8
    ... )
    >>>
    >>> # Or create individual components
    >>> embedding = create_field_layer(
    ...     'field_embedding',
    ...     vocab_size=10000,
    ...     embed_dim=256,
    ...     curvature_type='ricci'
    ... )
"""

from typing import Optional, Dict, Any, Literal, Union

import keras

# Import all layer classes
from .field_embedding import FieldEmbedding
from .connection_layer import ConnectionLayer
from .parallel_transport import ParallelTransportLayer
from .holonomy_layer import HolonomyLayer
from .gauge_invariant_attention import GaugeInvariantAttention
from .manifold_stress import ManifoldStressLayer
from .holonomic_transformer import HolonomicTransformerLayer, FieldNormalization

# Type for field layer types
FieldLayerType = Literal[
    'field_embedding',
    'connection',
    'parallel_transport',
    'holonomy',
    'gauge_attention',
    'manifold_stress',
    'holonomic_transformer',
    'field_norm'
]

# Layer registry mapping type names to classes
_FIELD_LAYER_REGISTRY: Dict[str, type] = {
    'field_embedding': FieldEmbedding,
    'connection': ConnectionLayer,
    'parallel_transport': ParallelTransportLayer,
    'holonomy': HolonomyLayer,
    'gauge_attention': GaugeInvariantAttention,
    'manifold_stress': ManifoldStressLayer,
    'holonomic_transformer': HolonomicTransformerLayer,
    'field_norm': FieldNormalization,
}

# Required parameters for each layer type
_REQUIRED_PARAMS: Dict[str, list] = {
    'field_embedding': ['vocab_size', 'embed_dim'],
    'connection': ['hidden_dim'],
    'parallel_transport': ['transport_dim'],
    'holonomy': ['hidden_dim'],
    'gauge_attention': ['hidden_dim'],
    'manifold_stress': ['hidden_dim'],
    'holonomic_transformer': ['hidden_dim'],
    'field_norm': [],
}

# Default parameters for each layer type
_DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    'field_embedding': {
        'curvature_type': 'ricci',
        'curvature_scale': 0.1,
        'curvature_regularization': 0.01,
    },
    'connection': {
        'connection_type': 'yang_mills',
        'num_generators': 8,
        'use_metric': True,
        'antisymmetric': True,
    },
    'parallel_transport': {
        'num_steps': 10,
        'transport_method': 'iterative',
        'step_size': 0.1,
    },
    'holonomy': {
        'loop_sizes': [2, 4, 8],
        'loop_type': 'rectangular',
        'num_loops': 4,
        'use_trace': True,
    },
    'gauge_attention': {
        'num_heads': 8,
        'attention_metric': 'hybrid',
        'use_curvature_gating': True,
        'use_parallel_transport': True,
    },
    'manifold_stress': {
        'stress_types': ['curvature', 'connection', 'combined'],
        'stress_threshold': 0.5,
        'use_learnable_baseline': True,
    },
    'holonomic_transformer': {
        'num_heads': 8,
        'curvature_type': 'ricci',
        'connection_type': 'yang_mills',
        'attention_metric': 'hybrid',
        'use_holonomy_features': True,
        'use_anomaly_detection': True,
        'dropout_rate': 0.1,
        'normalization_type': 'field_norm',
        'activation': 'gelu',
    },
    'field_norm': {
        'epsilon': 1e-6,
        'use_curvature_scaling': True,
        'center': True,
        'scale': True,
    },
}


def validate_field_config(
        layer_type: FieldLayerType,
        **kwargs: Any
) -> None:
    """
    Validate configuration for a field layer type.

    Args:
        layer_type: Type of field layer.
        **kwargs: Configuration parameters.

    Raises:
        ValueError: If layer_type is unknown or required parameters are missing.
    """
    if layer_type not in _FIELD_LAYER_REGISTRY:
        raise ValueError(
            f"Unknown field layer type: '{layer_type}'. "
            f"Available types: {list(_FIELD_LAYER_REGISTRY.keys())}"
        )

    required = _REQUIRED_PARAMS.get(layer_type, [])
    missing = [p for p in required if p not in kwargs]

    if missing:
        raise ValueError(
            f"Missing required parameters for '{layer_type}': {missing}"
        )


def create_field_layer(
        layer_type: FieldLayerType,
        name: Optional[str] = None,
        **kwargs: Any
) -> keras.layers.Layer:
    """
    Factory function to create field-based layers.

    This is the recommended way to create field layers, as it provides:
    - Automatic parameter validation
    - Default parameter filling
    - Consistent interface across all layer types

    Args:
        layer_type: Type of field layer to create. Options:
            - 'field_embedding': Field embedding with curvature
            - 'connection': Gauge connection computation
            - 'parallel_transport': Parallel transport of vectors
            - 'holonomy': Holonomy (path integral) computation
            - 'gauge_attention': Gauge-invariant attention
            - 'manifold_stress': Anomaly detection via stress
            - 'holonomic_transformer': Complete transformer layer
            - 'field_norm': Field-aware normalization
        name: Optional name for the layer.
        **kwargs: Layer-specific parameters.

    Returns:
        Configured field layer instance.

    Raises:
        ValueError: If layer_type is unknown or required parameters missing.

    Example:
        >>> # Create a holonomic transformer layer
        >>> layer = create_field_layer(
        ...     'holonomic_transformer',
        ...     hidden_dim=256,
        ...     num_heads=8,
        ...     use_holonomy_features=True
        ... )

        >>> # Create a field embedding
        >>> embedding = create_field_layer(
        ...     'field_embedding',
        ...     vocab_size=10000,
        ...     embed_dim=256,
        ...     curvature_type='ricci'
        ... )

        >>> # Create gauge-invariant attention
        >>> attention = create_field_layer(
        ...     'gauge_attention',
        ...     hidden_dim=256,
        ...     num_heads=8,
        ...     attention_metric='hybrid'
        ... )
    """
    # Validate configuration
    validate_field_config(layer_type, **kwargs)

    # Get layer class
    layer_class = _FIELD_LAYER_REGISTRY[layer_type]

    # Merge defaults with provided kwargs
    defaults = _DEFAULT_PARAMS.get(layer_type, {}).copy()
    defaults.update(kwargs)

    # Add name if provided
    if name is not None:
        defaults['name'] = name

    # Create and return layer
    return layer_class(**defaults)


def create_field_layer_from_config(config: Dict[str, Any]) -> keras.layers.Layer:
    """
    Create a field layer from a configuration dictionary.

    Args:
        config: Configuration dictionary containing 'type' and layer parameters.

    Returns:
        Configured field layer instance.

    Example:
        >>> config = {
        ...     'type': 'holonomic_transformer',
        ...     'hidden_dim': 256,
        ...     'num_heads': 8,
        ...     'use_holonomy_features': True
        ... }
        >>> layer = create_field_layer_from_config(config)
    """
    config = config.copy()
    layer_type = config.pop('type')
    return create_field_layer(layer_type, **config)


def get_field_layer_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all available field layer types.

    Returns:
        Dictionary mapping layer types to their information including:
        - 'class': The layer class
        - 'required_params': List of required parameters
        - 'default_params': Dictionary of default parameter values
        - 'description': Brief description of the layer

    Example:
        >>> info = get_field_layer_info()
        >>> print(info['holonomic_transformer']['description'])
    """
    descriptions = {
        'field_embedding': 'Embeds tokens as fields with curvature information',
        'connection': 'Computes gauge connection for parallel transport',
        'parallel_transport': 'Transports vectors along paths using connection',
        'holonomy': 'Computes holonomy (path integral) for gauge invariance',
        'gauge_attention': 'Attention mechanism respecting gauge structure',
        'manifold_stress': 'Detects anomalies via geometric stress',
        'holonomic_transformer': 'Complete transformer with field-based processing',
        'field_norm': 'Normalization that respects field curvature',
    }

    return {
        layer_type: {
            'class': layer_class,
            'required_params': _REQUIRED_PARAMS.get(layer_type, []),
            'default_params': _DEFAULT_PARAMS.get(layer_type, {}),
            'description': descriptions.get(layer_type, ''),
        }
        for layer_type, layer_class in _FIELD_LAYER_REGISTRY.items()
    }


# Public API
__all__ = [
    # Layer classes
    'FieldEmbedding',
    'ConnectionLayer',
    'ParallelTransportLayer',
    'HolonomyLayer',
    'GaugeInvariantAttention',
    'ManifoldStressLayer',
    'HolonomicTransformerLayer',
    'FieldNormalization',

    # Factory functions
    'create_field_layer',
    'create_field_layer_from_config',
    'validate_field_config',
    'get_field_layer_info',

    # Type definitions
    'FieldLayerType',
]