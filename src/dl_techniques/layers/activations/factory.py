"""
Activation Layer Factory for dl_techniques Framework

Provides a centralized factory for creating activation layers with a unified
interface, type safety, and comprehensive parameter validation. This utility
simplifies the instantiation of both standard and custom activation functions,
ensuring consistent configuration across the framework.
"""

import keras
from typing import Dict, Any, Literal, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

from .adaptive_softmax import AdaptiveTemperatureSoftmax
from .basis_function import BasisFunction
from .differentiable_step import DifferentiableStep
from .expanded_activations import (
    GELU, SiLU, xATLU, xGELU, xSiLU, EluPlusOne
)
from .golu import GoLU
from .hard_sigmoid import HardSigmoid
from .hard_swish import HardSwish
from .mish import Mish, SaturatedMish
from .monotonicity_layer import MonotonicityLayer
from .relu_k import ReLUK
from .routing_probabilities import RoutingProbabilitiesLayer
from .routing_probabilities_hierarchical import HierarchicalRoutingLayer
from .sparsemax import Sparsemax
from .squash import SquashLayer
from .thresh_max import ThreshMax

# ---------------------------------------------------------------------

# Type definition for Activation types
ActivationType = Literal[
    'adaptive_softmax',
    'basis_function',
    'differentiable_step',
    'elu_plus_one',
    'gelu',
    'golu',
    'hard_sigmoid',
    'hard_swish',
    'hierarchical_routing',
    'mish',
    'monotonicity',
    'relu',
    'relu_k',
    'routing_probabilities',
    'saturated_mish',
    'silu',
    'sparsemax',
    'squash',
    'thresh_max',
    'xatlu',
    'xgelu',
    'xsilu'
]


# Activation layer registry mapping types to classes and parameter info
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
        'use_case': (
            'Maintains sharpness in softmax for large output spaces, '
            'improving retrieval tasks.'
        )
    },
    'basis_function': {
        'class': BasisFunction,
        'description': 'Implements b(x) = x * sigmoid(x), equivalent to Swish/SiLU.',
        'required_params': [],
        'optional_params': {},
        'use_case': 'Used in PowerMLP architectures for smooth, non-linear transformations.'
    },
    'differentiable_step': {
        'class': DifferentiableStep,
        'description': 'Learnable, differentiable approximation of a step function (tanh-based).',
        'required_params': [],
        'optional_params': {
            'axis': -1,
            'slope_initializer': 'ones',
            'shift_initializer': 'zeros',
            'shift_regularizer': keras.regularizers.L2(1e-3),
            'shift_constraint': None  # Defaults to ValueRangeConstraint inside class
        },
        'use_case': 'Learnable binary gates, soft thresholding, or feature selection.'
    },
    'elu_plus_one': {
        'class': EluPlusOne,
        'description': 'Enhanced ELU activation: ELU(x) + 1 + epsilon.',
        'required_params': [],
        'optional_params': {},
        'use_case': 'Ensures outputs are strictly positive, useful for rate parameters in distributions.'
    },
    'gelu': {
        'class': GELU,
        'description': 'Gaussian Error Linear Unit, a smooth, non-monotonic activation.',
        'required_params': [],
        'optional_params': {},
        'use_case': 'State-of-the-art activation for Transformer-based models.'
    },
    'golu': {
        'class': GoLU,
        'description': 'Gompertz Linear Unit, a self-gated activation using an asymmetrical Gompertz curve.',
        'required_params': [],
        'optional_params': {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0},
        'use_case': (
            'Asymmetrical self-gated activation intended to create smoother '
            'loss landscapes and improve model generalization.'
        )
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
    'hierarchical_routing': {
        'class': HierarchicalRoutingLayer,
        'description': 'Trainable hierarchical probability tree for O(log N) classification.',
        'required_params': ['output_dim'],
        'optional_params': {
            'axis': -1,
            'epsilon': 1e-7,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None
        },
        'use_case': 'Efficient classification for very large output spaces (e.g., language modeling).'
    },
    'mish': {
        'class': Mish,
        'description': 'A self-regularized, non-monotonic activation: x * tanh(softplus(x)).',
        'required_params': [],
        'optional_params': {},
        'use_case': (
            'Smooth activation that can outperform ReLU and Swish in deep '
            'vision and NLP models.'
        )
    },
    'monotonicity': {
        'class': MonotonicityLayer,
        'description': 'Enforces monotonic (non-decreasing) constraints on predictions.',
        'required_params': [],
        'optional_params': {
            'method': 'cumulative_softplus',
            'axis': -1,
            'min_spacing': None,
            'max_spacing': None,
            'value_range': None,
            'clip_inputs': None,
            'input_clip_range': (-20.0, 20.0),
            'epsilon': 1e-7
        },
        'use_case': 'Quantile regression, survival analysis, dose-response modeling.'
    },
    'relu': {
        'class': keras.layers.ReLU,
        'description': 'Rectified Linear Unit, the most common activation function.',
        'required_params': [],
        'optional_params': {
            'max_value': None,
            'negative_slope': 0.0,
            'threshold': 0.0
        },
        'use_case': (
            'Default activation for hidden layers in many types of neural '
            'networks due to its simplicity and effectiveness.'
        )
    },
    'relu_k': {
        'class': ReLUK,
        'description': 'Powered ReLU activation: max(0, x)^k.',
        'required_params': [],
        'optional_params': {'k': 3},
        'use_case': 'Creates more aggressive non-linearities than standard ReLU.'
    },
    'routing_probabilities': {
        'class': RoutingProbabilitiesLayer,
        'description': 'A non-trainable hierarchical routing layer using cosine basis patterns.',
        'required_params': [],
        'optional_params': {
            'output_dim': None,
            'axis': -1,
            'epsilon': 1e-7
        },
        'use_case': (
            'Parameter-free alternative to softmax for multi-class '
            'classification, introducing a structured, hierarchical bias.'
        )
    },
    'saturated_mish': {
        'class': SaturatedMish,
        'description': 'Mish variant that smoothly saturates for large positive inputs.',
        'required_params': [],
        'optional_params': {'alpha': 3.0, 'beta': 0.5},
        'use_case': (
            'Prevents activation explosion in very deep networks by '
            'saturating the Mish function.'
        )
    },
    'silu': {
        'class': SiLU,
        'description': 'Sigmoid Linear Unit (SiLU/Swish), defined as x * sigmoid(x).',
        'required_params': [],
        'optional_params': {},
        'use_case': 'Self-gated activation that often outperforms ReLU in deep networks.'
    },
    'sparsemax': {
        'class': Sparsemax,
        'description': 'Projects logits onto the probability simplex using Euclidean projection (L2).',
        'required_params': [],
        'optional_params': {'axis': -1},
        'use_case': (
            'Produces sparse probability distributions (with exact zeros), '
            'ideal for interpretable attention mechanisms.'
        )
    },
    'squash': {
        'class': SquashLayer,
        'description': 'Squashing non-linearity for Capsule Networks.',
        'required_params': [],
        'optional_params': {'axis': -1, 'epsilon': None},
        'use_case': (
            'Core non-linearity for Capsule Networks, normalizing vector '
            'outputs to represent probabilities.'
        )
    },
    'thresh_max': {
        'class': ThreshMax,
        'description': 'Sparse softmax variant using a differentiable step function.',
        'required_params': [],
        'optional_params': {
            'axis': -1,
            'slope': 10.0,
            'epsilon': 1e-12,
            'trainable_slope': False,
            'slope_initializer': 'ones',
            'slope_regularizer': keras.regularizers.L2(-1e-3),
            'slope_constraint': None
        },
        'use_case': 'Creates sparse, confident probability distributions as an alternative to softmax.'
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
        'use_case': (
            'Expanded activation with an arctan gate; provides adaptable '
            'gating for specialized tasks.'
        )
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
        'use_case': (
            'Extends GELU with a trainable parameter to adapt the gating '
            'range, enhancing flexibility.'
        )
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
    }
}


# Public API functions
def get_activation_info() -> Dict[str, Dict[str, Any]]:
    """
    Get comprehensive information about all available activation layer types.

    Returns:
        Dict containing information about each activation type, including:
        - description: Human-readable description
        - required_params: List of required parameter names
        - optional_params: Dict of optional parameters with defaults
        - use_case: Recommended use case for the activation type
    """
    return {
        act_type: info.copy() for act_type, info in ACTIVATION_REGISTRY.items()
    }


def validate_activation_config(activation_type: str, **kwargs: Any) -> None:
    """
    Validate activation layer configuration parameters.

    Args:
        activation_type: Type of activation to validate.
        **kwargs: Parameters to validate.

    Raises:
        ValueError: If activation_type is invalid or parameters have invalid values.
        TypeError: If parameter types are incorrect.
    """
    if activation_type not in ACTIVATION_REGISTRY:
        available_types = sorted(list(ACTIVATION_REGISTRY.keys()))
        raise ValueError(
            f"Unknown activation type '{activation_type}'. "
            f"Available types: {available_types}"
        )

    # Validate required parameters exist
    required = ACTIVATION_REGISTRY[activation_type]['required_params']
    for param in required:
        if param not in kwargs:
            raise ValueError(
                f"Missing required parameter '{param}' for activation "
                f"type '{activation_type}'."
            )

    # Specific parameter logic validation
    if activation_type == 'adaptive_softmax':
        min_temp = kwargs.get('min_temp', 0.1)
        max_temp = kwargs.get('max_temp', 1.0)
        entropy_threshold = kwargs.get('entropy_threshold', 0.5)
        if min_temp <= 0.0:
            raise ValueError(f"min_temp must be positive, got {min_temp}")
        if max_temp <= 0.0:
            raise ValueError(f"max_temp must be positive, got {max_temp}")
        if min_temp > max_temp:
            raise ValueError(
                f"min_temp ({min_temp}) must be <= max_temp ({max_temp})"
            )
        if entropy_threshold < 0.0:
            raise ValueError(
                f"entropy_threshold must be non-negative, got {entropy_threshold}"
            )

    elif activation_type == 'differentiable_step':
        axis = kwargs.get('axis', -1)
        if axis is not None and not isinstance(axis, int):
            raise TypeError(f"axis must be int or None, got {type(axis)}")

    elif activation_type == 'golu':
        for param in ['alpha', 'beta', 'gamma']:
            val = kwargs.get(param, 1.0)
            if val <= 0.0:
                raise ValueError(f"{param} must be positive, got {val}")

    elif activation_type == 'hierarchical_routing':
        output_dim = kwargs.get('output_dim')
        if not isinstance(output_dim, int) or output_dim <= 1:
            raise ValueError(
                f"output_dim must be an integer > 1, got {output_dim}"
            )

    elif activation_type == 'monotonicity':
        method = kwargs.get('method', 'cumulative_softplus')
        valid_methods = [
            "cumulative_softplus", "exponential", "sigmoid",
            "normalized_softmax", "squared", "cumulative_exp"
        ]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid monotonicity method '{method}'. "
                f"Must be one of {valid_methods}"
            )
        if method in ["sigmoid", "normalized_softmax"]:
            if "value_range" not in kwargs or kwargs["value_range"] is None:
                raise ValueError(
                    f"value_range (min, max) is required for method '{method}'"
                )
            if len(kwargs["value_range"]) != 2:
                raise ValueError("value_range must be a tuple of (min, max)")

    elif activation_type == 'relu':
        max_val = kwargs.get('max_value')
        if max_val is not None and not isinstance(max_val, (int, float)):
            raise TypeError("max_value must be a number or None")

    elif activation_type == 'relu_k':
        k = kwargs.get('k', 3)
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")

    elif activation_type == 'routing_probabilities':
        output_dim = kwargs.get('output_dim')
        if output_dim is not None:
            if not isinstance(output_dim, int) or output_dim <= 1:
                raise ValueError(
                    f"output_dim must be integer > 1, got {output_dim}"
                )

    elif activation_type == 'saturated_mish':
        alpha = kwargs.get('alpha', 3.0)
        beta = kwargs.get('beta', 0.5)
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError("alpha and beta must be positive")

    elif activation_type == 'thresh_max':
        slope = kwargs.get('slope', 10.0)
        if slope <= 0:
            raise ValueError(f"slope must be positive, got {slope}")

    # Validate generic object params for expanded activations
    if activation_type in ['xatlu', 'xgelu', 'xsilu']:
        for param in ['alpha_initializer', 'alpha_regularizer', 'alpha_constraint']:
            if param in kwargs and isinstance(kwargs[param], str):
                try:
                    if 'initializer' in param:
                        keras.initializers.get(kwargs[param])
                    elif 'regularizer' in param:
                        keras.regularizers.get(kwargs[param])
                    elif 'constraint' in param:
                        keras.constraints.get(kwargs[param])
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid {param}: {e}")


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
        **kwargs: Parameters specific to the activation type.

    Returns:
        A configured activation layer instance.

    Raises:
        ValueError: If activation_type is invalid or parameters are incorrect.
        TypeError: If parameter types are incorrect.

    Example:
        ```python
        # Create a standard GELU
        gelu = create_activation_layer('gelu')

        # Create Monotonicity Layer
        mono = create_activation_layer(
            'monotonicity',
            method='exponential',
            clip_inputs=True
        )

        # Create Hierarchical Routing
        routing = create_activation_layer(
            'hierarchical_routing',
            output_dim=10000
        )
        ```
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

        # Add required parameters only if they are in kwargs
        for req in act_info['required_params']:
            if req in kwargs:
                params[req] = kwargs[req]

        # Update with remaining kwargs
        params.update(kwargs)

        # Filter out parameters that are not accepted by the class constructor
        # This is a safety measure, though typically classes accept **kwargs
        valid_param_names = (
            set(act_info['required_params']) |
            set(act_info['optional_params'].keys())
        )

        # Only strict filtering if we want to prevent passing extra kwargs to
        # layers that might not handle them well, but most Keras layers use **kwargs.
        # We'll trust the validation logic above and pass what we have,
        # but prioritizing explicit registry params + name.

        final_params = {
            k: v for k, v in params.items()
            if k in valid_param_names or k in kwargs
        }

        if name is not None:
            final_params['name'] = name

        # Log creation
        logger.info(f"Creating {activation_type} layer.")
        logger.debug(f"Params: {final_params}")

        # Instantiate
        activation_layer = act_class(**final_params)

        return activation_layer

    except (TypeError, ValueError) as e:
        # Provide enhanced error reporting
        act_info = ACTIVATION_REGISTRY.get(activation_type)
        class_name = act_info.get('class', type(None)).__name__ if act_info else "Unknown"

        error_msg = (
            f"Failed to create {activation_type} layer ({class_name}). "
            f"Provided keys: {list(kwargs.keys())}. "
            f"Error: {e}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def create_activation_from_config(config: Dict[str, Any]) -> keras.layers.Layer:
    """
    Create an activation layer from a configuration dictionary.

    Args:
        config: Configuration dictionary containing a 'type' key and parameters.

    Returns:
        A configured activation layer instance.

    Raises:
        ValueError: If 'type' key is missing or config is invalid.
    """
    if not isinstance(config, dict):
        raise ValueError(f"config must be a dictionary, got {type(config)}")

    if 'type' not in config:
        raise ValueError("Configuration must include a 'type' key")

    config_copy = config.copy()
    activation_type = config_copy.pop('type')

    return create_activation_layer(activation_type, **config_copy)