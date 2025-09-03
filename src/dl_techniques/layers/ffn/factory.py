"""
FFN Factory Utility for dl_techniques Framework

Provides a centralized factory function for creating feed-forward network layers
with unified interface, type safety, and comprehensive parameter validation.
"""

import keras
from typing import Dict, Any, Literal, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

from .mlp import MLPBlock
from .swiglu_ffn import SwiGLUFFN
from .diff_ffn import DifferentialFFN
from .glu_ffn import GLUFFN
from .geglu_ffn import GeGLUFFN
from .residual_block import ResidualBlock
from .swin_mlp import SwinMLP

# ---------------------------------------------------------------------
# Type definition for FFN types
# ---------------------------------------------------------------------

FFNType = Literal[
    'mlp',
    'swiglu',
    'differential',
    'glu',
    'geglu',
    'residual',
    'swin_mlp'
]

# ---------------------------------------------------------------------
# FFN layer registry mapping types to classes and parameter info
# ---------------------------------------------------------------------

FFN_REGISTRY: Dict[str, Dict[str, Any]] = {
    'mlp': {
        'class': MLPBlock,
        'description': 'Standard MLP with intermediate expansion',
        'required_params': ['hidden_dim', 'output_dim'],
        'optional_params': {
            'activation': 'gelu',
            'dropout_rate': 0.0,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': 'General purpose feed-forward processing in transformers'
    },
    'swiglu': {
        'class': SwiGLUFFN,
        'description': 'SwiGLU Feed-Forward Network with gating mechanism',
        'required_params': ['d_model'],
        'optional_params': {
            'ffn_expansion_factor': 4,
            'ffn_multiple_of': 256,
            'dropout_rate': 0.0,
            'use_bias': False,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': 'Modern transformer architectures (LLaMa, Qwen, etc.)'
    },
    'differential': {
        'class': DifferentialFFN,
        'description': 'Differential Feed-Forward Network with dual-pathway processing',
        'required_params': ['hidden_dim', 'output_dim'],
        'optional_params': {
            'branch_activation': 'gelu',
            'gate_activation': 'sigmoid',
            'dropout_rate': 0.0,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': 'Enhanced feature processing with differential pathways'
    },
    'glu': {
        'class': GLUFFN,
        'description': 'Gated Linear Unit Feed Forward Network',
        'required_params': ['hidden_dim', 'output_dim'],
        'optional_params': {
            'activation': 'swish',
            'dropout_rate': 0.0,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': 'Gated processing for improved gradient flow'
    },
    'geglu': {
        'class': GeGLUFFN,
        'description': 'GELU Gated Linear Unit Feed-Forward Network',
        'required_params': ['hidden_dim', 'output_dim'],
        'optional_params': {
            'activation': 'gelu',
            'dropout_rate': 0.0,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': 'GELU-based gated processing for transformers'
    },
    'residual': {
        'class': ResidualBlock,
        'description': 'Residual block with skip connections',
        'required_params': ['hidden_dim', 'output_dim'],
        'optional_params': {
            'dropout_rate': 0.0,
            'activation': 'relu',
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': 'Deep networks requiring skip connections for gradient flow'
    },
    'swin_mlp': {
        'class': SwinMLP,
        'description': 'Swin Transformer MLP with configurable activation and regularization',
        'required_params': ['hidden_dim'],
        'optional_params': {
            'use_bias': True,
            'out_dim': None,
            'activation': 'gelu',
            'drop_rate': 0.0,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'activity_regularizer': None
        },
        'use_case': 'Swin Transformer architectures and vision models'
    }
}


# ---------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------

def get_ffn_info() -> Dict[str, Dict[str, Any]]:
    """
    Get comprehensive information about all available FFN types.

    Returns:
        Dict containing information about each FFN type, including:
        - description: Human-readable description
        - required_params: List of required parameter names
        - optional_params: Dict of optional parameters with defaults
        - use_case: Recommended use case for the FFN type

    Example:
        ```python
        info = get_ffn_info()

        # Print available FFN types
        for ffn_type, details in info.items():
            print(f"{ffn_type}: {details['description']}")
            print(f"  Required: {details['required_params']}")
            print(f"  Use case: {details['use_case']}")
        ```
    """
    return {ffn_type: info.copy() for ffn_type, info in FFN_REGISTRY.items()}


def validate_ffn_config(ffn_type: str, **kwargs: Any) -> None:
    """
    Validate FFN configuration parameters.

    Args:
        ffn_type: Type of FFN to validate
        **kwargs: Parameters to validate

    Raises:
        ValueError: If ffn_type is invalid or required parameters are missing

    Example:
        ```python
        # Validate before creating
        try:
            validate_ffn_config('mlp', hidden_dim=512, output_dim=256)
            print("Configuration is valid")
        except ValueError as e:
            print(f"Invalid configuration: {e}")
        ```
    """
    if ffn_type not in FFN_REGISTRY:
        available_types = list(FFN_REGISTRY.keys())
        raise ValueError(
            f"Unknown FFN type '{ffn_type}'. "
            f"Available types: {available_types}"
        )

    ffn_info = FFN_REGISTRY[ffn_type]
    required_params = ffn_info['required_params']

    # Check for required parameters
    missing_params = [param for param in required_params if param not in kwargs]
    if missing_params:
        raise ValueError(
            f"Required parameters missing for {ffn_type}: {missing_params}. "
            f"Required: {required_params}"
        )

    # Validate common parameter constraints
    if 'dropout_rate' in kwargs:
        dropout_rate = kwargs['dropout_rate']
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0.0 and 1.0, got {dropout_rate}")

    if 'drop_rate' in kwargs:  # Used by swin_mlp
        drop_rate = kwargs['drop_rate']
        if not (0.0 <= drop_rate <= 1.0):
            raise ValueError(f"drop_rate must be between 0.0 and 1.0, got {drop_rate}")

    if 'hidden_dim' in kwargs:
        hidden_dim = kwargs['hidden_dim']
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

    if 'output_dim' in kwargs:
        output_dim = kwargs['output_dim']
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")

    if 'd_model' in kwargs:
        d_model = kwargs['d_model']
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

    # Validate SwiGLU specific parameters
    if ffn_type == 'swiglu':
        if 'ffn_expansion_factor' in kwargs:
            factor = kwargs['ffn_expansion_factor']
            if factor <= 0:
                raise ValueError(f"ffn_expansion_factor must be positive, got {factor}")

        if 'ffn_multiple_of' in kwargs:
            multiple = kwargs['ffn_multiple_of']
            if multiple <= 0:
                raise ValueError(f"ffn_multiple_of must be positive, got {multiple}")

    # Validate activation functions are valid strings
    activation_params = ['activation', 'branch_activation', 'gate_activation']
    for param in activation_params:
        if param in kwargs:
            activation = kwargs[param]
            if isinstance(activation, str) and activation != 'linear':
                try:
                    # Test that the activation string is valid
                    keras.activations.get(activation)
                except (ValueError, KeyError):
                    raise ValueError(f"Unknown {param} function: '{activation}'")

    # Validate initializer strings
    initializer_params = ['kernel_initializer', 'bias_initializer']
    for param in initializer_params:
        if param in kwargs:
            initializer = kwargs[param]
            if isinstance(initializer, str):
                try:
                    keras.initializers.get(initializer)
                except (ValueError, KeyError):
                    raise ValueError(f"Unknown {param}: '{initializer}'")


def create_ffn_layer(
        ffn_type: FFNType,
        name: Optional[str] = None,
        **kwargs: Any
) -> keras.layers.Layer:
    """
    Factory function for creating FFN layers with unified interface.

    This function provides a centralized way to create any FFN layer supported by
    dl_techniques, with comprehensive parameter validation and consistent error handling.

    Args:
        ffn_type: Type of FFN layer to create. Supported types:
            - 'mlp': Standard MLP with intermediate expansion
            - 'swiglu': SwiGLU activation with gating mechanism
            - 'differential': Dual-pathway processing with differential branches
            - 'glu': Gated Linear Unit with sigmoid gating
            - 'geglu': GELU-based Gated Linear Unit
            - 'residual': Residual block with skip connections
            - 'swin_mlp': Swin Transformer MLP variant
        name: Optional name for the layer
        **kwargs: Parameters specific to the FFN type. See individual layer
            documentation for parameter details.

    Returns:
        Configured FFN layer instance

    Raises:
        ValueError: If ffn_type is invalid or required parameters are missing
        TypeError: If parameter types are incorrect

    Example:
        ```python
        # Create standard MLP
        mlp = create_ffn_layer('mlp', hidden_dim=512, output_dim=256)

        # Create SwiGLU with custom parameters
        swiglu = create_ffn_layer(
            'swiglu',
            d_model=768,
            ffn_expansion_factor=4,
            dropout_rate=0.1,
            name='swiglu_ffn'
        )

        # Create differential FFN with custom activations
        diff_ffn = create_ffn_layer(
            'differential',
            hidden_dim=1024,
            output_dim=512,
            branch_activation='relu',
            gate_activation='sigmoid'
        )

        # Create residual block
        res_block = create_ffn_layer(
            'residual',
            hidden_dim=256,
            output_dim=256,
            dropout_rate=0.2
        )
        ```

    Note:
        Each FFN type has specific required and optional parameters.
        Use get_ffn_info() to see parameter requirements for each type.
    """
    try:
        # Validate configuration
        validate_ffn_config(ffn_type, **kwargs)

        # Get FFN info and class
        ffn_info = FFN_REGISTRY[ffn_type]
        ffn_class = ffn_info['class']

        # Prepare parameters with defaults
        params = {}

        # Get all valid parameter names for this ffn_type
        valid_param_names = set(ffn_info['required_params']) | set(ffn_info['optional_params'].keys())

        # Start with defaults for all optional parameters
        params.update(ffn_info['optional_params'])
        # Update with any user-provided kwargs
        params.update(kwargs)

        # Filter out any unknown parameters to avoid "Unrecognized keyword arguments" error
        final_params = {key: val for key, val in params.items() if key in valid_param_names}

        # Add name if provided
        if name is not None:
            final_params['name'] = name

        # Log final parameters before creation
        logger.info(f"Creating {ffn_type} FFN layer with parameters:")
        log_params = final_params.copy()
        if name:
            log_params['name'] = name
        for param_name, param_value in sorted(log_params.items()):
            if param_name == 'name':
                logger.info(f"  {param_name}: '{param_value}'")
            elif isinstance(param_value, str):
                logger.info(f"  {param_name}: '{param_value}'")
            elif param_value is None:
                logger.info(f"  {param_name}: None")
            else:
                logger.info(f"  {param_name}: {param_value}")

        # Create FFN layer using registry class directly (no if/elif chain)
        ffn_layer = ffn_class(**final_params)

        logger.debug(f"Successfully created {ffn_type} FFN layer: {ffn_layer.name}")
        return ffn_layer

    except (TypeError, ValueError) as e:
        # Enhanced error reporting with context
        ffn_info = FFN_REGISTRY.get(ffn_type)
        if ffn_info:
            required_params = ffn_info.get('required_params', [])
            provided_params = list(kwargs.keys())
            class_name = ffn_info.get('class', type(None)).__name__
            error_msg = (
                f"Failed to create {ffn_type} FFN layer ({class_name}). "
                f"Required parameters: {required_params}. "
                f"Provided parameters: {provided_params}. "
                f"Check parameter compatibility and types. "
                f"Use get_ffn_info() for detailed parameter information. "
                f"Original error: {e}"
            )
        else:
            error_msg = f"Failed to create FFN layer. Unknown FFN type '{ffn_type}'. Original error: {e}"

        logger.error(error_msg)
        raise ValueError(error_msg) from e


def create_ffn_from_config(config: Dict[str, Any]) -> keras.layers.Layer:
    """
    Create FFN layer from configuration dictionary.

    Args:
        config: Configuration dictionary containing 'type' key and parameters

    Returns:
        Configured FFN layer instance

    Raises:
        ValueError: If 'type' key is missing from config

    Example:
        ```python
        config = {
            'type': 'mlp',
            'hidden_dim': 1024,
            'output_dim': 512,
            'dropout_rate': 0.1,
            'name': 'ffn_block'
        }
        ffn = create_ffn_from_config(config)
        ```
    """
    if not isinstance(config, dict):
        raise ValueError(f"config must be a dictionary, got {type(config)}")

    if 'type' not in config:
        raise ValueError("Configuration must include 'type' key")

    config_copy = config.copy()
    ffn_type = config_copy.pop('type')

    logger.debug(f"Creating FFN from config - type: {ffn_type}, params: {list(config_copy.keys())}")

    return create_ffn_layer(ffn_type, **config_copy)

# ---------------------------------------------------------------------