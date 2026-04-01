"""
Normalization Layer Factory Utility for dl_techniques Framework.

This module provides a centralized factory function for creating various normalization
layers supported by the dl_techniques framework. It offers a unified interface for
instantiating different normalization techniques with customizable parameters.

The factory supports both standard Keras normalization layers and specialized
normalization layers from the dl_techniques framework, enabling easy experimentation
and architectural flexibility.
"""

import keras
from typing import Optional, Dict, Any, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .rms_norm import RMSNorm
from .band_rms import BandRMS
from .adaptive_band_rms import AdaptiveBandRMS
from .band_logit_norm import BandLogitNorm
from .global_response_norm import GlobalResponseNormalization
from .logit_norm import LogitNorm
from .max_logit_norm import MaxLogitNorm, DecoupledMaxLogit, DMLPlus
from .dynamic_tanh import DynamicTanh
from .zero_centered_rms_norm import ZeroCenteredRMSNorm
from .zero_centered_band_rms_norm import ZeroCenteredBandRMSNorm

# ---------------------------------------------------------------------

NormalizationType = Literal[
    'layer_norm', 'batch_norm', 'rms_norm', 'zero_centered_rms_norm',
    'zero_centered_band_rms_norm', 'band_rms', 'adaptive_band_rms',
    'band_logit_norm', 'global_response_norm', 'logit_norm', 'max_logit_norm',
    'decoupled_max_logit', 'dml_plus_focal', 'dml_plus_center', 'dynamic_tanh'
]


# ---------------------------------------------------------------------

def create_normalization_layer(
        normalization_type: NormalizationType,
        name: Optional[str] = None,
        epsilon: float = 1e-6,
        **kwargs: Any
) -> keras.layers.Layer:
    """
    Create a normalization layer based on the specified type with customizable parameters.

    This factory function provides a unified interface for creating various normalization
    layers, supporting both standard Keras layers and specialized layers from the
    dl_techniques framework.

    :param normalization_type: Type of normalization layer to create. Supported types
        include 'layer_norm', 'batch_norm', 'rms_norm', 'zero_centered_rms_norm',
        'zero_centered_band_rms_norm', 'band_rms', 'adaptive_band_rms',
        'band_logit_norm', 'global_response_norm', 'logit_norm', 'max_logit_norm',
        'decoupled_max_logit', 'dml_plus_focal', 'dml_plus_center', and 'dynamic_tanh'.
    :type normalization_type: NormalizationType
    :param name: Optional name for the layer. If None, layer will use default naming.
    :type name: Optional[str]
    :param epsilon: Small constant for numerical stability. Defaults to 1e-6.
        Used by normalization layers that support epsilon parameter.
    :type epsilon: float
    :param kwargs: Additional keyword arguments specific to each normalization type.
        Common kwargs include axis, center, scale, use_scale, max_band_width,
        temperature, constant, alpha_init_value, and eps (for GRN).
    :return: Configured normalization layer instance ready for use in neural networks.
    :rtype: keras.layers.Layer
    :raises ValueError: If normalization_type is not supported or if invalid parameters
        are provided for the specific normalization type.
    :raises TypeError: If kwargs contain invalid parameter types for the chosen layer.
    """
    # Prepare base parameters
    layer_kwargs = kwargs.copy()
    if name is not None:
        layer_kwargs['name'] = name

    # Create the appropriate normalization layer
    if normalization_type == 'layer_norm':
        # Standard Keras LayerNormalization
        layer_kwargs.setdefault('epsilon', epsilon)
        return keras.layers.LayerNormalization(**layer_kwargs)

    elif normalization_type == 'batch_norm':
        # Standard Keras BatchNormalization
        layer_kwargs.setdefault('epsilon', epsilon)
        return keras.layers.BatchNormalization(**layer_kwargs)

    elif normalization_type == 'rms_norm':
        # Root Mean Square normalization
        layer_kwargs.setdefault('epsilon', epsilon)
        return RMSNorm(**layer_kwargs)

    elif normalization_type == 'zero_centered_rms_norm':
        # Zero-centered RMS normalization with enhanced stability
        layer_kwargs.setdefault('epsilon', epsilon)
        return ZeroCenteredRMSNorm(**layer_kwargs)

    elif normalization_type == 'zero_centered_band_rms_norm':
        # Zero-centered RMS with band constraints
        layer_kwargs.setdefault('epsilon', epsilon)
        return ZeroCenteredBandRMSNorm(**layer_kwargs)

    elif normalization_type == 'band_rms':
        # RMS normalization with bounded constraints
        layer_kwargs.setdefault('epsilon', epsilon)
        return BandRMS(**layer_kwargs)

    elif normalization_type == 'adaptive_band_rms':
        # Adaptive RMS with log-transformed scaling
        layer_kwargs.setdefault('epsilon', epsilon)
        return AdaptiveBandRMS(**layer_kwargs)

    elif normalization_type == 'band_logit_norm':
        # Band-constrained logit normalization
        layer_kwargs.setdefault('epsilon', epsilon)
        return BandLogitNorm(**layer_kwargs)

    elif normalization_type == 'global_response_norm':
        # Global Response Normalization (GRN)
        # GRN uses 'eps' instead of 'epsilon'
        if 'eps' not in layer_kwargs:
            layer_kwargs['eps'] = epsilon
        return GlobalResponseNormalization(**layer_kwargs)

    elif normalization_type == 'logit_norm':
        # LogitNorm for classification tasks
        layer_kwargs.setdefault('epsilon', epsilon)
        return LogitNorm(**layer_kwargs)

    elif normalization_type == 'max_logit_norm':
        # MaxLogit normalization for OOD detection
        layer_kwargs.setdefault('epsilon', epsilon)
        return MaxLogitNorm(**layer_kwargs)

    elif normalization_type == 'decoupled_max_logit':
        # Decoupled MaxLogit (DML) normalization
        layer_kwargs.setdefault('epsilon', epsilon)
        return DecoupledMaxLogit(**layer_kwargs)

    elif normalization_type == 'dml_plus_focal':
        # DML+ focal model variant
        layer_kwargs.setdefault('epsilon', epsilon)
        layer_kwargs['model_type'] = 'focal'
        return DMLPlus(**layer_kwargs)

    elif normalization_type == 'dml_plus_center':
        # DML+ center model variant
        layer_kwargs.setdefault('epsilon', epsilon)
        layer_kwargs['model_type'] = 'center'
        return DMLPlus(**layer_kwargs)

    elif normalization_type == 'dynamic_tanh':
        # Dynamic Tanh normalization (normalization-free transformers)
        # DynamicTanh doesn't use epsilon, remove it if present
        layer_kwargs.pop('epsilon', None)
        return DynamicTanh(**layer_kwargs)

    else:
        supported_types = [
            'layer_norm', 'batch_norm', 'rms_norm', 'zero_centered_rms_norm',
            'zero_centered_band_rms_norm', 'band_rms', 'adaptive_band_rms',
            'band_logit_norm', 'global_response_norm', 'logit_norm',
            'max_logit_norm', 'decoupled_max_logit', 'dml_plus_focal',
            'dml_plus_center', 'dynamic_tanh'
        ]
        raise ValueError(
            f"Unknown normalization type: '{normalization_type}'. "
            f"Supported types: {', '.join(supported_types)}"
        )


# ---------------------------------------------------------------------


def get_normalization_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all supported normalization types and their parameters.

    :return: Dictionary mapping normalization type names to their parameter information,
        including description, supported parameters, and usage notes.
    :rtype: Dict[str, Dict[str, Any]]
    """
    return {
        'layer_norm': {
            'description': 'Standard Keras LayerNormalization with learnable scale and bias',
            'parameters': ['axis', 'epsilon', 'center', 'scale'],
            'use_case': 'General purpose normalization for transformers and deep networks'
        },
        'batch_norm': {
            'description': 'Standard Keras BatchNormalization with moving statistics',
            'parameters': ['axis', 'epsilon', 'center', 'scale', 'momentum'],
            'use_case': 'Convolutional networks and batch-based training'
        },
        'rms_norm': {
            'description': 'Root Mean Square normalization without centering',
            'parameters': ['axis', 'epsilon', 'use_scale', 'scale_initializer'],
            'use_case': 'Transformers, especially for faster training and inference'
        },
        'zero_centered_rms_norm': {
            'description': 'Zero-centered RMS normalization combining RMSNorm efficiency with LayerNorm stability',
            'parameters': ['axis', 'epsilon', 'use_scale', 'scale_initializer'],
            'use_case': 'Large language models and transformers requiring enhanced training stability'
        },
        'zero_centered_band_rms_norm': {
            'description': 'Combines zero-centering, RMS, and band constraints for maximum stability',
            'parameters': ['max_band_width', 'axis', 'epsilon', 'band_initializer', 'band_regularizer'],
            'use_case': 'Advanced transformer and LLM architectures for ultimate stability and flexibility'
        },
        'band_rms': {
            'description': 'RMS normalization with bounded magnitude constraints',
            'parameters': ['max_band_width', 'axis', 'epsilon'],
            'use_case': 'Training stability in deep networks with gradient control'
        },
        'adaptive_band_rms': {
            'description': 'Adaptive RMS with log-transformed RMS-based scaling',
            'parameters': ['max_band_width', 'axis', 'epsilon'],
            'use_case': 'Advanced training stability with adaptive scaling'
        },
        'band_logit_norm': {
            'description': 'Band-constrained logit normalization for classification',
            'parameters': ['max_band_width', 'axis', 'epsilon'],
            'use_case': 'Classification tasks with logit magnitude control'
        },
        'global_response_norm': {
            'description': 'Global Response Normalization from ConvNeXt',
            'parameters': ['eps', 'gamma_initializer', 'beta_initializer'],
            'use_case': 'ConvNeXt-style architectures and vision_heads models'
        },
        'logit_norm': {
            'description': 'Temperature-scaled normalization for classification',
            'parameters': ['temperature', 'axis', 'epsilon'],
            'use_case': 'Classification with calibrated confidence estimates'
        },
        'max_logit_norm': {
            'description': 'MaxLogit normalization for out-of-distribution detection',
            'parameters': ['axis', 'epsilon'],
            'use_case': 'OOD detection and uncertainty estimation'
        },
        'decoupled_max_logit': {
            'description': 'Decoupled MaxLogit (DML) with constant decoupling',
            'parameters': ['constant', 'axis', 'epsilon'],
            'use_case': 'Advanced OOD detection with decoupled learning'
        },
        'dml_plus_focal': {
            'description': 'DML+ focal model for separate model training',
            'parameters': ['axis', 'epsilon'],
            'use_case': 'DML+ framework focal model component'
        },
        'dml_plus_center': {
            'description': 'DML+ center model for separate model training',
            'parameters': ['axis', 'epsilon'],
            'use_case': 'DML+ framework center model component'
        },
        'dynamic_tanh': {
            'description': 'Dynamic Tanh normalization for normalization-free transformers',
            'parameters': ['axis', 'alpha_init_value', 'kernel_initializer'],
            'use_case': 'Normalization-free transformer architectures'
        }
    }


# ---------------------------------------------------------------------

def validate_normalization_config(
        normalization_type: NormalizationType,
        **kwargs: Any
) -> bool:
    """
    Validate normalization configuration parameters.

    :param normalization_type: Type of normalization to validate.
    :type normalization_type: NormalizationType
    :param kwargs: Configuration parameters to validate.
    :return: True if configuration is valid.
    :rtype: bool
    :raises ValueError: If configuration is invalid.
    """
    info = get_normalization_info()

    if normalization_type not in info:
        raise ValueError(f"Unknown normalization type: {normalization_type}")

    valid_params = set(info[normalization_type]['parameters'] + ['name'])
    provided_params = set(kwargs.keys())

    # Check for invalid parameters
    invalid_params = provided_params - valid_params
    if invalid_params:
        raise ValueError(
            f"Invalid parameters for {normalization_type}: {invalid_params}. "
            f"Valid parameters: {valid_params}"
        )

    # Type-specific validations
    if normalization_type in ['band_rms', 'adaptive_band_rms', 'band_logit_norm', 'zero_centered_band_rms_norm']:
        if 'max_band_width' in kwargs:
            max_band_width = kwargs['max_band_width']
            if not isinstance(max_band_width, (int, float)) or max_band_width <= 0:
                raise ValueError("max_band_width must be a positive number")

    if normalization_type == 'logit_norm':
        if 'temperature' in kwargs:
            temperature = kwargs['temperature']
            if not isinstance(temperature, (int, float)) or temperature <= 0:
                raise ValueError("temperature must be a positive number")

    if normalization_type == 'decoupled_max_logit':
        if 'constant' in kwargs:
            constant = kwargs['constant']
            if not isinstance(constant, (int, float)):
                raise ValueError("constant must be a number")

    if normalization_type in ['rms_norm', 'zero_centered_rms_norm']:
        if 'epsilon' in kwargs:
            epsilon = kwargs['epsilon']
            if not isinstance(epsilon, (int, float)) or epsilon <= 0:
                raise ValueError("epsilon must be a positive number")

    if normalization_type == 'dynamic_tanh':
        if 'alpha_init_value' in kwargs:
            alpha_init_value = kwargs['alpha_init_value']
            if not isinstance(alpha_init_value, (int, float)) or alpha_init_value <= 0:
                raise ValueError("alpha_init_value must be a positive number")

    return True


# ---------------------------------------------------------------------


def create_normalization_from_config(config: Dict[str, Any]) -> keras.layers.Layer:
    """
    Create a normalization layer from a configuration dictionary.

    This function provides an alternative interface for creating normalization layers
    when the configuration is stored as a dictionary, commonly used in configuration
    files or hyperparameter specifications.

    :param config: Configuration dictionary containing a 'type' key (required),
        and optionally 'name', 'epsilon', and additional parameters specific to the
        normalization type.
    :type config: Dict[str, Any]
    :return: Configured normalization layer instance.
    :rtype: keras.layers.Layer
    :raises KeyError: If 'type' key is missing from config.
    :raises ValueError: If normalization type or parameters are invalid.
    """
    if 'type' not in config:
        raise KeyError("Configuration dictionary must contain 'type' key")

    config_copy = config.copy()
    normalization_type = config_copy.pop('type')

    return create_normalization_layer(normalization_type, **config_copy)

# ---------------------------------------------------------------------