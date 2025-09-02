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

# ---------------------------------------------------------------------

NormalizationType = Literal[
    'layer_norm', 'batch_norm', 'rms_norm', 'band_rms', 'adaptive_band_rms',
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
    dl_techniques framework. It allows for easy experimentation with different
    normalization techniques through a single API.

    Args:
        normalization_type: Type of normalization layer to create. Supported types:
            - 'layer_norm': Standard Keras LayerNormalization
            - 'batch_norm': Standard Keras BatchNormalization
            - 'rms_norm': Root Mean Square normalization
            - 'band_rms': RMS normalization with bounded constraints
            - 'adaptive_band_rms': Adaptive RMS with log-transformed scaling
            - 'band_logit_norm': Band-constrained logit normalization
            - 'global_response_norm': Global response normalization (GRN)
            - 'logit_norm': LogitNorm for classification tasks
            - 'max_logit_norm': MaxLogit normalization for OOD detection
            - 'decoupled_max_logit': Decoupled MaxLogit (DML) normalization
            - 'dml_plus_focal': DML+ focal model variant
            - 'dml_plus_center': DML+ center model variant
            - 'dynamic_tanh': Dynamic Tanh normalization (normalization-free)
        name: Optional name for the layer. If None, layer will use default naming.
        epsilon: Small constant for numerical stability. Defaults to 1e-6.
            Used by normalization layers that support epsilon parameter.
        **kwargs: Additional keyword arguments specific to each normalization type.
            These will override default parameters and allow fine-tuning of layer behavior.

            Common kwargs by normalization type:

            layer_norm/batch_norm:
                - axis: Normalization axis
                - center: Whether to add learnable bias
                - scale: Whether to add learnable scale

            rms_norm:
                - axis: Normalization axis (int or tuple)
                - use_scale: Whether to use learnable scale parameter

            band_rms/adaptive_band_rms:
                - max_band_width: Maximum band width constraint
                - axis: Normalization axis

            band_logit_norm:
                - max_band_width: Maximum band width for logit constraint
                - axis: Normalization axis

            global_response_norm:
                - eps: Alternative epsilon parameter name for GRN

            logit_norm:
                - temperature: Temperature parameter for logit scaling
                - axis: Normalization axis

            max_logit_norm:
                - axis: Normalization axis

            decoupled_max_logit:
                - constant: Constant value for decoupling
                - axis: Normalization axis

            dml_plus_focal/dml_plus_center:
                - model_type: Automatically set based on normalization_type
                - axis: Normalization axis

            dynamic_tanh:
                - axis: Normalization axis
                - alpha_init_value: Initial alpha value for dynamic scaling

    Returns:
        Configured normalization layer instance ready for use in neural networks.

    Raises:
        ValueError: If normalization_type is not supported or if invalid parameters
            are provided for the specific normalization type.
        TypeError: If kwargs contain invalid parameter types for the chosen layer.

    Example:
        ```python
        # Standard layer normalization
        layer_norm = create_normalization_layer(
            'layer_norm',
            name='encoder_norm',
            epsilon=1e-5
        )

        # RMS normalization with custom axis
        rms_norm = create_normalization_layer(
            'rms_norm',
            name='rms_norm_layer',
            axis=-1,
            use_scale=True
        )

        # Band RMS with custom constraints
        band_rms = create_normalization_layer(
            'band_rms',
            name='constrained_norm',
            max_band_width=0.05,
            epsilon=1e-7
        )

        # Global Response Normalization
        grn = create_normalization_layer(
            'global_response_norm',
            name='grn_layer',
            eps=1e-6
        )

        # Dynamic Tanh for normalization-free transformers
        dynamic_norm = create_normalization_layer(
            'dynamic_tanh',
            name='dynamic_norm',
            alpha_init_value=0.5,
            axis=[-1]
        )
        ```

    Note:
        - The epsilon parameter is automatically mapped to the appropriate parameter
          name for each layer type (e.g., 'eps' for GRN, 'epsilon' for others)
        - Some normalization types may ignore certain parameters if they're not applicable
        - For specialized normalization layers, refer to their individual documentation
          for detailed parameter descriptions and usage guidelines
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
            'layer_norm', 'batch_norm', 'rms_norm', 'band_rms', 'adaptive_band_rms',
            'band_logit_norm', 'global_response_norm', 'logit_norm', 'max_logit_norm',
            'decoupled_max_logit', 'dml_plus_focal', 'dml_plus_center', 'dynamic_tanh'
        ]
        raise ValueError(
            f"Unknown normalization type: '{normalization_type}'. "
            f"Supported types: {', '.join(supported_types)}"
        )

# ---------------------------------------------------------------------


def get_normalization_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all supported normalization types and their parameters.

    Returns:
        Dictionary mapping normalization type names to their parameter information,
        including description, supported parameters, and usage notes.

    Example:
        ```python
        info = get_normalization_info()
        print(info['rms_norm']['description'])
        print(info['band_rms']['parameters'])
        ```
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
            'parameters': ['axis', 'epsilon', 'use_scale'],
            'use_case': 'Transformers, especially for faster training and inference'
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
            'use_case': 'ConvNeXt-style architectures and vision models'
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
            'parameters': ['axis', 'alpha_init_value'],
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

    Args:
        normalization_type: Type of normalization to validate.
        **kwargs: Configuration parameters to validate.

    Returns:
        True if configuration is valid.

    Raises:
        ValueError: If configuration is invalid.

    Example:
        ```python
        # This will pass
        validate_normalization_config('rms_norm', axis=-1, use_scale=True)

        # This will raise ValueError
        validate_normalization_config('dynamic_tanh', epsilon=1e-6)  # DynamicTanh doesn't use epsilon
        ```
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
    if normalization_type in ['band_rms', 'adaptive_band_rms', 'band_logit_norm']:
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

    return True

# ---------------------------------------------------------------------
