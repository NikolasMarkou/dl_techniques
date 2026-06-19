"""
A Factory Method design pattern to provide a single,
centralized entry point for creating various Feed-Forward Network (FFN)
architectures. By abstracting the instantiation logic, it decouples client
code from the concrete implementation of specific FFN layers. This is a core
component for building flexible, modular, and configuration-driven models.

Architectural Overview:
The factory operates on a registry-based design (`FFN_REGISTRY`). This
registry maps a simple string identifier (the `ffn_type`) to the corresponding
Keras Layer class and its associated metadata, such as required parameters
and default values.

When called, the factory performs the following steps:
1.  **Validation**: It first consults the registry to validate the requested
    `ffn_type` and ensures that all required hyperparameters are provided in
    `**kwargs`. This centralized validation guarantees that any layer created
    is correctly configured.
2.  **Class Retrieval**: It retrieves the appropriate Keras Layer class
    associated with the `ffn_type`.
3.  **Instantiation**: It instantiates the retrieved class, passing the
    validated and filtered keyword arguments to its constructor.

This design provides several key advantages for machine learning engineering:
-   **Modularity and Extensibility**: New FFN architectures can be integrated
    into the framework simply by adding them to the registry, without any
    changes to the model-building code that uses this factory.
-   **Configuration-Driven Experimentation**: It enables model architectures
    to be defined in external configuration files (e.g., YAML or JSON), where
    the choice of FFN is specified by a single string. This greatly simplifies
    hyperparameter tuning and architectural A/B testing.
-   **Consistency and Reliability**: It provides a single, consistent interface
    for creating FFNs, reducing the risk of misconfiguration and ensuring that
    all layers adhere to a common set of standards.

Foundational Concepts:
The factory itself is an application of a well-established software design
pattern. The layers it produces, however, are based on significant research
in deep learning. It provides access to a curated set of FFNs, each with its
own mathematical underpinnings, including:

-   The standard "expand-then-contract" MLP from the original Transformer.
-   Advanced gated variants like GLU, GeGLU, and SwiGLU, which introduce
    dynamic, input-dependent information filtering.
-   Residual blocks that facilitate gradient flow in very deep networks.

By using this factory, a researcher can easily switch between these different
computational blocks to evaluate their impact on model performance.

References:
-   Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994). Design
    Patterns: Elements of Reusable Object-Oriented Software. Addison-Wesley.
-   Vaswani, A., et al. (2017). Attention Is All You Need. NIPS.
-   Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv preprint
    arXiv:2002.05202.

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
from .gelu_mlp_ffn import GELUMLPFFN
from .residual_block import ResidualBlock
from .swin_mlp import SwinMLP
from .counting_ffn import CountingFFN
from .logic_ffn import LogicFFN
from .gated_mlp import GatedMLP
from .orthoglu_ffn import OrthoGLUFFN
from .power_mlp_layer import PowerMLPLayer
from .kan_linear import KANLinear
from .tversky_projection import TverskyProjectionLayer
from .monarch_ffn import MonarchFFN

# ---------------------------------------------------------------------
# Type definition for FFN types
# ---------------------------------------------------------------------

FFNType = Literal[
    'counting',
    'differential',
    'gated_mlp',
    'geglu',
    'gelu_tanh',
    'glu',
    'kan',
    'logic',
    'mlp',
    'monarch',
    'orthoglu',
    'power_mlp',
    'residual',
    'swiglu',
    'swin_mlp',
    'tversky'
]

# ---------------------------------------------------------------------
# FFN layer registry mapping types to classes and parameter info
# ---------------------------------------------------------------------

FFN_REGISTRY: Dict[str, Dict[str, Any]] = {
    'counting': {
        'class': CountingFFN,
        'description': 'Feed-Forward Network that learns to count features in a sequence',
        'required_params': ['output_dim', 'count_dim'],
        'optional_params': {
            'counting_scope': 'local',
            'activation': 'gelu',
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': 'Sequence processing where feature frequency or position is important'
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
    'gated_mlp': {
        'class': GatedMLP,
        'description': 'Spatially-gated MLP using 1x1 convolutions, an alternative to self-attention',
        'required_params': ['filters'],
        'optional_params': {
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'attention_activation': 'relu',
            'output_activation': 'linear',
            'data_format': None
        },
        'use_case': 'Vision models, replacing attention with a computationally cheaper alternative'
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
    'gelu_tanh': {
        'class': GELUMLPFFN,
        'description': 'SD3-style GELU (tanh-approximation) MLP FeedForward (Dense -> gelu(approximate=True) -> Dense)',
        'required_params': ['hidden_dim'],
        'optional_params': {
            'output_dim': None,
            'dropout_rate': 0.0,
            'use_bias': True
        },
        'use_case': 'SD3 / MMDiT FeedForward; tanh-approximate GELU MLP, output_dim defaults to input dim'
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
    'kan': {
        'class': KANLinear,
        'description': (
            'Kolmogorov-Arnold Network linear layer with learnable per-connection '
            'univariate activations parameterized by B-splines. Supports N-D inputs via einsum.'
        ),
        'required_params': ['features'],
        'optional_params': {
            'grid_size': 5,
            'spline_order': 3,
            'grid_range': (-2.0, 2.0),
            'activation': 'swish',
            'base_trainable': True,
            'spline_trainable': True,
            'kernel_initializer': 'glorot_uniform',
            'base_scaler_initializer': 'ones',
            'epsilon': 1e-7
        },
        'use_case': 'Learnable per-connection univariate activations via B-splines (Kolmogorov-Arnold)'
    },
    'logic': {
        'class': LogicFFN,
        'description': 'Feed-Forward Network that performs soft logical reasoning',
        'required_params': ['output_dim', 'logic_dim'],
        'optional_params': {
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'temperature': 1.0
        },
        'use_case': 'Tasks requiring symbolic-like reasoning or feature interaction modeling'
    },
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
    'monarch': {
        'class': MonarchFFN,
        'description': (
            'Order-2 Monarch-structured FFN: each projection is a product of two '
            'block-diagonal matrices interleaved with a reshape/permute (Dao et al. 2022). '
            'Sub-quadratic parameter count; nblocks must divide input_dim, hidden_dim and output_dim.'
        ),
        'required_params': ['hidden_dim', 'output_dim'],
        'optional_params': {
            'nblocks': 4,
            'activation': 'gelu',
            'dropout_rate': 0.0,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None
        },
        'use_case': 'Parameter-efficient structured replacement for dense FFN projections'
    },
    'orthoglu': {
        'class': OrthoGLUFFN,
        'description': 'Orthogonally-regularized Gated Linear Unit for disciplined routing',
        'required_params': ['hidden_dim', 'output_dim'],
        'optional_params': {
            'activation': 'gelu',
            'dropout_rate': 0.0,
            'use_bias': True,
            'ortho_reg_factor': 1.0
        },
        'use_case': 'Deep networks requiring stable training and decorrelated features'
    },
    'power_mlp': {
        'class': PowerMLPLayer,
        'description': 'Dual-branch MLP with ReLUK and basis functions for enhanced expressiveness',
        'required_params': ['units'],
        'optional_params': {
            'k': 3,
            'kernel_initializer': 'he_normal',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'use_bias': True
        },
        'use_case': 'Tasks requiring approximation of complex functions with both sharp and smooth components'
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
    'swiglu': {
        'class': SwiGLUFFN,
        'description': 'SwiGLU Feed-Forward Network with gating mechanism',
        'required_params': ['output_dim'],
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
    'tversky': {
        'class': TverskyProjectionLayer,
        'description': (
            'Asymmetric Tversky-similarity projection layer. NOTE: operates on rank-2 inputs '
            '(batch, input_dim) only; output shape is (batch, units). Not suitable for rank-3 '
            '(batch, time, dim) consumers.'
        ),
        'required_params': ['units', 'num_features'],
        'optional_params': {
            'intersection_reduction': 'product',
            'difference_reduction': 'subtractmatch',
            'prototype_initializer': 'glorot_uniform',
            'feature_initializer': 'glorot_uniform',
            'contrast_initializer': 'ones'
        },
        'use_case': 'Asymmetric, psychologically-grounded similarity-based projection alternative to Dense (rank-2 only)'
    },
    'swin_mlp': {
        'class': SwinMLP,
        'description': 'Swin Transformer MLP with configurable activation and regularization',
        'required_params': ['hidden_dim'],
        'optional_params': {
            'use_bias': True,
            'output_dim': None,
            'activation': 'gelu',
            'dropout_rate': 0.0,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'activity_regularizer': None
        },
        'use_case': 'Swin Transformer architectures and vision_heads models'
    }
}


# ---------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------

def get_ffn_info() -> Dict[str, Dict[str, Any]]:
    """
    Get comprehensive information about all available FFN types.

    :return: Dict containing information about each FFN type, including
        description, required_params, optional_params, and use_case.
    :rtype: Dict[str, Dict[str, Any]]
    """
    return {ffn_type: info.copy() for ffn_type, info in FFN_REGISTRY.items()}


def validate_ffn_config(ffn_type: str, **kwargs: Any) -> None:
    """
    Validate FFN configuration parameters.

    :param ffn_type: Type of FFN to validate.
    :type ffn_type: str
    :param kwargs: Parameters to validate.
    :raises ValueError: If ffn_type is invalid or required parameters are missing.
    """
    if ffn_type not in FFN_REGISTRY:
        available_types = sorted(list(FFN_REGISTRY.keys()))
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

    positive_dims = ['hidden_dim', 'output_dim', 'count_dim', 'logic_dim', 'filters', 'units', 'features', 'num_features']
    for dim_param in positive_dims:
        if dim_param in kwargs and kwargs[dim_param] is not None:
            if kwargs[dim_param] <= 0:
                raise ValueError(f"{dim_param} must be positive, got {kwargs[dim_param]}")

    # Validate type-specific parameters
    if ffn_type == 'swiglu':
        if 'ffn_expansion_factor' in kwargs and kwargs['ffn_expansion_factor'] <= 0:
            raise ValueError(f"ffn_expansion_factor must be positive, got {kwargs['ffn_expansion_factor']}")
        if 'ffn_multiple_of' in kwargs and kwargs['ffn_multiple_of'] <= 0:
            raise ValueError(f"ffn_multiple_of must be positive, got {kwargs['ffn_multiple_of']}")
    elif ffn_type == 'counting':
        if 'counting_scope' in kwargs and kwargs['counting_scope'] not in ["global", "local", "causal"]:
            raise ValueError("counting_scope must be one of 'global', 'local', 'causal'")
    elif ffn_type == 'logic':
        if 'temperature' in kwargs and kwargs['temperature'] <= 0:
            raise ValueError(f"temperature must be positive, got {kwargs['temperature']}")
    elif ffn_type == 'gated_mlp':
        valid_activations = {"relu", "gelu", "swish", "silu", "linear"}
        if 'attention_activation' in kwargs and kwargs['attention_activation'] not in valid_activations:
            raise ValueError(f"attention_activation must be one of {valid_activations}")
        if 'output_activation' in kwargs and kwargs['output_activation'] not in valid_activations:
            raise ValueError(f"output_activation must be one of {valid_activations}")
    elif ffn_type == 'monarch':
        if 'nblocks' in kwargs:
            nblocks = kwargs['nblocks']
            if not isinstance(nblocks, int) or nblocks <= 0:
                raise ValueError(f"nblocks must be a positive integer, got {nblocks}")
    elif ffn_type == 'power_mlp':
        if 'k' in kwargs:
            k = kwargs['k']
            if not isinstance(k, int) or k <= 0:
                raise ValueError(f"k must be a positive integer, got {k}")
    elif ffn_type == 'kan':
        if 'grid_size' in kwargs and (not isinstance(kwargs['grid_size'], int) or kwargs['grid_size'] <= 0):
            raise ValueError(f"grid_size must be a positive integer, got {kwargs['grid_size']}")
        if 'spline_order' in kwargs and (not isinstance(kwargs['spline_order'], int) or kwargs['spline_order'] < 0):
            raise ValueError(f"spline_order must be a non-negative integer, got {kwargs['spline_order']}")
        if 'grid_range' in kwargs:
            gr = kwargs['grid_range']
            if not (isinstance(gr, (tuple, list)) and len(gr) == 2 and gr[0] < gr[1]):
                raise ValueError(f"grid_range must be a (low, high) tuple with low < high, got {gr}")
        if 'epsilon' in kwargs and kwargs['epsilon'] <= 0:
            raise ValueError(f"epsilon must be positive, got {kwargs['epsilon']}")
    elif ffn_type == 'tversky':
        valid_ir = {'product', 'min', 'mean'}
        if 'intersection_reduction' in kwargs and kwargs['intersection_reduction'] not in valid_ir:
            raise ValueError(
                f"intersection_reduction must be one of {sorted(valid_ir)}, "
                f"got '{kwargs['intersection_reduction']}'"
            )
        valid_dr = {'ignorematch', 'subtractmatch'}
        if 'difference_reduction' in kwargs and kwargs['difference_reduction'] not in valid_dr:
            raise ValueError(
                f"difference_reduction must be one of {sorted(valid_dr)}, "
                f"got '{kwargs['difference_reduction']}'"
            )

    # Validate activation functions are valid strings
    activation_params = ['activation', 'branch_activation', 'gate_activation', 'attention_activation', 'output_activation']
    for param in activation_params:
        if param in kwargs:
            activation = kwargs[param]
            if isinstance(activation, str) and activation != 'linear':
                try:
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

    :param ffn_type: Type of FFN layer to create. See ``FFNType`` for all supported types.
    :type ffn_type: FFNType
    :param name: Optional name for the layer.
    :type name: Optional[str]
    :param kwargs: Parameters specific to the FFN type. See individual layer
        documentation for parameter details.
    :return: Configured FFN layer instance.
    :rtype: keras.layers.Layer
    :raises ValueError: If ffn_type is invalid or required parameters are missing.
    :raises TypeError: If parameter types are incorrect.
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

    :param config: Configuration dictionary containing 'type' key and parameters.
    :type config: Dict[str, Any]
    :return: Configured FFN layer instance.
    :rtype: keras.layers.Layer
    :raises ValueError: If 'type' key is missing from config.
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
