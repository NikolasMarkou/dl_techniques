"""
Activation Layer Factory for dl_techniques Framework

Provides a centralized factory for creating activation layers with a unified
interface, type safety, and comprehensive parameter validation. This utility
simplifies the instantiation of both standard and custom activation functions.
"""

import keras
from typing import Dict, Any, Literal, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

from .adaptive_softmax import AdaptiveTemperatureSoftmax
from .basis_function import BasisFunction
from .expanded_activations import (
    GELU, SiLU, xATLU, xGELU, xSiLU, EluPlusOne
)
from .hard_sigmoid import HardSigmoid
from .hard_swish import HardSwish
from .mish import Mish, SaturatedMish
from .relu_k import ReLUK
from .squash import SquashLayer
from .thresh_max import ThreshMax

# ---------------------------------------------------------------------
# Type definition for Activation types
# ---------------------------------------------------------------------

ActivationType = Literal[
    'adaptive_softmax',
    'basis_function',
    'gelu',
    'silu',
    'xatlu',
    'xgelu',
    'xsilu',
    'elu_plus_one',
    'hard_sigmoid',
    'hard_swish',
    'mish',
    'saturated_mish',
    'relu_k',
    'squash',
    'thresh_max'
]

# ---------------------------------------------------------------------
# Activation layer registry mapping types to classes and parameter info
# ---------------------------------------------------------------------

ACTIVATION_REGISTRY: Dict[str, Dict[str, Any]] = {
    'adaptive_softmax': {
        'class': AdaptiveTemperatureSoftmax,
        'description': 'Softmax with dynamic temperature based on input entropy.',
        'required_params': [],
        'optional_params': {
            'min_temp': 0.1,
            'max_temp': 1.0,
            'entropy_threshold': 0.5,
            'eps': 1e-7,
            'polynomial_coeffs': None
        },
        'use_case': 'Maintains sharpness in softmax for large output spaces, improving retrieval tasks.'
    },
    'basis_function': {
        'class': BasisFunction,
        'description': 'Implements b(x) = x * sigmoid(x), equivalent to Swish/SiLU.',
        'required_params': [],
        'optional_params': {},
        'use_case': 'Used in PowerMLP architectures for smooth, non-linear transformations.'
    },
    'gelu': {
        'class': GELU,
        'description': 'Gaussian Error Linear Unit, a smooth, non-monotonic activation.',
        'required_params': [],
        'optional_params': {},
        'use_case': 'State-of-the-art activation for Transformer-based models.'
    },
    'silu': {
        'class': SiLU,
        'description': 'Sigmoid Linear Unit (SiLU/Swish), defined as x * sigmoid(x).',
        'required_params': [],
        'optional_params': {},
        'use_case': 'Self-gated activation that often outperforms ReLU in deep networks.'
    },
    'xatlu': {
        'class': xATLU,
        'description': 'Expanded ArcTan Linear Unit with a trainable alpha parameter.',
        'required_params': [],
        'optional_params': {
            'alpha_initializer': 'zeros',
            'alpha_regularizer': None,
            'alpha_constraint': None
        },
        'use_case': 'Expanded activation with an arctan gate; provides adaptable gating for specialized tasks.'
    },
    'xgelu': {
        'class': xGELU,
        'description': 'Expanded Gaussian Error Linear Unit with a trainable alpha parameter.',
        'required_params': [],
        'optional_params': {
            'alpha_initializer': 'zeros',
            'alpha_regularizer': None,
            'alpha_constraint': None
        },
        'use_case': 'Extends GELU with a trainable parameter to adapt the gating range, enhancing flexibility.'
    },
    'xsilu': {
        'class': xSiLU,
        'description': 'Expanded Sigmoid Linear Unit with a trainable alpha parameter.',
        'required_params': [],
        'optional_params': {
            'alpha_initializer': 'zeros',
            'alpha_regularizer': None,
            'alpha_constraint': None
        },
        'use_case': 'Extends SiLU/Swish with a trainable parameter to adapt the gating range.'
    },
    'elu_plus_one': {
        'class': EluPlusOne,
        'description': 'Enhanced ELU activation: ELU(x) + 1 + epsilon.',
        'required_params': [],
        'optional_params': {},
        'use_case': 'Ensures outputs are strictly positive, useful for rate parameters in distributions.'
    },
    'hard_sigmoid': {
        'class': HardSigmoid,
        'description': 'Hard-sigmoid activation, a computationally efficient approximation of sigmoid.',
        'required_params': [],
        'optional_params': {},
        'use_case': 'Efficient gating in mobile networks and squeeze-and-excitation modules.'
    },
    'hard_swish': {
        'class': HardSwish,
        'description': 'Hard-swish activation, a computationally efficient variant of Swish/SiLU.',
        'required_params': [],
        'optional_params': {},
        'use_case': 'High-performance activation for mobile-optimized models like MobileNetV3.'
    },
    'mish': {
        'class': Mish,
        'description': 'A self-regularized, non-monotonic activation: x * tanh(softplus(x)).',
        'required_params': [],
        'optional_params': {},
        'use_case': 'Smooth activation that can outperform ReLU and Swish in deep vision and NLP models.'
    },
    'saturated_mish': {
        'class': SaturatedMish,
        'description': 'Mish variant that smoothly saturates for large positive inputs.',
        'required_params': [],
        'optional_params': {'alpha': 3.0, 'beta': 0.5},
        'use_case': 'Prevents activation explosion in very deep networks by saturating the Mish function.'
    },
    'relu_k': {
        'class': ReLUK,
        'description': 'Powered ReLU activation: max(0, x)^k.',
        'required_params': [],
        'optional_params': {'k': 3},
        'use_case': 'Creates more aggressive non-linearities than standard ReLU.'
    },
    'squash': {
        'class': SquashLayer,
        'description': 'Squashing non-linearity for Capsule Networks.',
        'required_params': [],
        'optional_params': {'axis': -1, 'epsilon': None},
        'use_case': 'Core non-linearity for Capsule Networks, normalizing vector outputs to represent probabilities.'
    },
    'thresh_max': {
        'class': ThreshMax,
        'description': 'Sparse softmax variant using a differentiable step function.',
        'required_params': [],
        'optional_params': {'axis': -1, 'slope': 10.0, 'epsilon': 1e-12},
        'use_case': 'Creates sparse, confident probability distributions as an alternative to softmax.'
    }
}

# ---------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------

def get_activation_info() -> Dict[str, Dict[str, Any]]:
    """
    Get comprehensive information about all available activation layer types.

    Returns:
        Dict containing information about each activation type, including:
        - description: Human-readable description
        - required_params: List of required parameter names
        - optional_params: Dict of optional parameters with defaults
        - use_case: Recommended use case for the activation type

    Example:
        ```python
        info = get_activation_info()
        for act_type, details in info.items():
            print(f"{act_type}: {details['description']}")
            print(f"  Use case: {details['use_case']}")
        ```
    """
    return {act_type: info.copy() for act_type, info in ACTIVATION_REGISTRY.items()}

def validate_activation_config(activation_type: str, **kwargs: Any) -> None:
    """
    Validate activation layer configuration parameters.

    Args:
        activation_type: Type of activation to validate.
        **kwargs: Parameters to validate.

    Raises:
        ValueError: If activation_type is invalid or parameters have invalid values.

    Example:
        ```python
        try:
            validate_activation_config('relu_k', k=3)
            print("Configuration is valid")
        except ValueError as e:
            print(f"Invalid configuration: {e}")
        ```
    """
    if activation_type not in ACTIVATION_REGISTRY:
        available_types = list(ACTIVATION_REGISTRY.keys())
        raise ValueError(
            f"Unknown activation type '{activation_type}'. "
            f"Available types: {available_types}"
        )

    # Specific parameter validations
    if activation_type == 'adaptive_softmax':
        min_temp = kwargs.get('min_temp', 0.1)
        max_temp = kwargs.get('max_temp', 1.0)
        entropy_threshold = kwargs.get('entropy_threshold', 0.5)
        if min_temp <= 0.0: raise ValueError(f"min_temp must be positive, got {min_temp}")
        if max_temp <= 0.0: raise ValueError(f"max_temp must be positive, got {max_temp}")
        if min_temp > max_temp: raise ValueError(f"min_temp ({min_temp}) must be <= max_temp ({max_temp})")
        if entropy_threshold < 0.0: raise ValueError(f"entropy_threshold must be non-negative, got {entropy_threshold}")

    if activation_type == 'saturated_mish':
        alpha = kwargs.get('alpha', 3.0)
        beta = kwargs.get('beta', 0.5)
        if alpha <= 0.0: raise ValueError(f"alpha must be positive, got {alpha}")
        if beta <= 0.0: raise ValueError(f"beta must be positive, got {beta}")

    if activation_type == 'relu_k':
        k = kwargs.get('k', 3)
        if not isinstance(k, int): raise TypeError(f"k must be an integer, got type {type(k).__name__}")
        if k <= 0: raise ValueError(f"k must be a positive integer, got {k}")

    if activation_type == 'thresh_max':
        slope = kwargs.get('slope', 10.0)
        epsilon = kwargs.get('epsilon', 1e-12)
        if slope <= 0: raise ValueError(f"slope must be positive, got {slope}")
        if epsilon <= 0: raise ValueError(f"epsilon must be positive, got {epsilon}")

    # Validate initializer/regularizer/constraint strings for expanded activations
    if activation_type in ['xatlu', 'xgelu', 'xsilu']:
        for param_name, getter in [
            ('alpha_initializer', keras.initializers.get),
            ('alpha_regularizer', keras.regularizers.get),
            ('alpha_constraint', keras.constraints.get)
        ]:
            if param_name in kwargs and isinstance(kwargs[param_name], str):
                try:
                    getter(kwargs[param_name])
                except (ValueError, KeyError):
                    raise ValueError(f"Unknown {param_name}: '{kwargs[param_name]}'")


# ---------------------------------------------------------------------


def create_activation_layer(
    activation_type: ActivationType,
    name: Optional[str] = None,
    **kwargs: Any
) -> keras.layers.Layer:
    """
    Factory function for creating activation layers with a unified interface.

    This function provides a centralized way to create any activation layer
    supported by dl_techniques, with comprehensive parameter validation.

    Args:
        activation_type: Type of activation layer to create.
        name: Optional name for the layer.
        **kwargs: Parameters specific to the activation type. See individual
            layer documentation for details.

    Returns:
        A configured activation layer instance.

    Raises:
        ValueError: If activation_type is invalid or parameters are incorrect.
        TypeError: If parameter types are incorrect.

    Example:
        ```python
        # Create a standard GELU
        gelu_layer = create_activation_layer('gelu')

        # Create ReLUK with a custom power
        relu_k_layer = create_activation_layer('relu_k', k=2, name='relu_squared')

        # Create a SaturatedMish with custom thresholds
        sat_mish = create_activation_layer(
            'saturated_mish',
            alpha=4.0,
            beta=0.2
        )
        ```

    Note:
        Use `get_activation_info()` to see parameter details for each type.
    """
    try:
        # Validate configuration first
        validate_activation_config(activation_type, **kwargs)

        # Get activation info and class from registry
        act_info = ACTIVATION_REGISTRY[activation_type]
        act_class = act_info['class']

        # Prepare parameters: start with defaults, override with user kwargs
        params = {}
        params.update(act_info['optional_params'])
        params.update(kwargs)

        # Filter out any unknown parameters to avoid constructor errors
        valid_param_names = set(act_info['required_params']) | set(act_info['optional_params'].keys())
        final_params = {key: val for key, val in params.items() if key in valid_param_names}

        # Add name if provided
        if name is not None:
            final_params['name'] = name

        # Log parameters before creating the layer
        logger.info(f"Creating {activation_type} activation layer with parameters:")
        log_params = {**final_params, 'name': name} if name else final_params
        for param_name, param_value in sorted(log_params.items()):
            logger.info(f"  {param_name}: {repr(param_value)}")

        # Create the layer instance
        activation_layer = act_class(**final_params)

        logger.debug(f"Successfully created {activation_type} layer: {activation_layer.name}")
        return activation_layer

    except (TypeError, ValueError) as e:
        # Provide enhanced error reporting with context
        act_info = ACTIVATION_REGISTRY.get(activation_type)
        if act_info:
            class_name = act_info.get('class', type(None)).__name__
            error_msg = (
                f"Failed to create {activation_type} layer ({class_name}). "
                f"Provided parameters: {list(kwargs.keys())}. "
                f"Check parameter compatibility and types. "
                f"Use get_activation_info() for detailed parameter information. "
                f"Original error: {e}"
            )
        else:
            error_msg = f"Failed to create activation layer. Unknown type '{activation_type}'. Original error: {e}"

        logger.error(error_msg)
        raise ValueError(error_msg) from e

# ---------------------------------------------------------------------

def create_activation_from_config(config: Dict[str, Any]) -> keras.layers.Layer:
    """
    Create an activation layer from a configuration dictionary.

    Args:
        config: Configuration dictionary containing a 'type' key and parameters.

    Returns:
        A configured activation layer instance.

    Raises:
        ValueError: If 'type' key is missing from config or config is not a dict.

    Example:
        ```python
        config = {
            'type': 'thresh_max',
            'slope': 15.0,
            'name': 'sparse_output_activation'
        }
        layer = create_activation_from_config(config)
        ```
    """
    if not isinstance(config, dict):
        raise ValueError(f"config must be a dictionary, got {type(config)}")

    if 'type' not in config:
        raise ValueError("Configuration must include a 'type' key")

    config_copy = config.copy()
    activation_type = config_copy.pop('type')

    logger.debug(f"Creating activation from config - type: {activation_type}, params: {list(config_copy.keys())}")

    return create_activation_layer(activation_type, **config_copy)

# ---------------------------------------------------------------------
