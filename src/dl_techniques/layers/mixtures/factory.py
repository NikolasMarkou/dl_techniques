"""
A Factory Method design pattern providing a single, centralized entry point for
creating the differentiable clustering / mixture layers in this sub-package
(`RBFLayer`, `KMeansLayer`, `GMMLayer`). By abstracting the instantiation logic,
it decouples client code from the concrete layer implementations and enables
configuration-driven model construction, mirroring the `attention/`, `norms/`,
and `ffn/` factories.

Architectural Overview:
The factory operates on a registry-based design (`MIXTURE_REGISTRY`). This
registry maps a simple string identifier (the `mixture_type`) to the
corresponding Keras Layer class and its associated metadata (required
parameters, optional parameters with defaults, description, use case).

When called, the factory:
1.  **Validates** the requested `mixture_type` and ensures all required
    hyperparameters are present in ``**kwargs`` (centralized validation).
2.  **Retrieves** the Keras Layer class associated with the type.
3.  **Filters** unknown kwargs so per-type optional params never leak into the
    wrong constructor, then **instantiates** the class.

Supported layers:
-   ``rbf``    — :class:`RBFLayer`, a Radial Basis Function layer with learnable
    centers and a repulsion mechanism that keeps centers separated.
-   ``kmeans`` — :class:`KMeansLayer`, a differentiable soft K-means assignment
    layer with online centroid updates.
-   ``gmm``    — :class:`GMMLayer`, a differentiable Gaussian Mixture layer
    producing soft component responsibilities with an isometric regularizer.

References:
-   Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994). Design
    Patterns: Elements of Reusable Object-Oriented Software. Addison-Wesley.
"""

import keras
from typing import Dict, Any, Literal, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ...utils.logger import logger

from .radial_basis_function import RBFLayer
from .kmeans import KMeansLayer
from .gmm import GMMLayer

# ---------------------------------------------------------------------
# Type definition for mixture types
# ---------------------------------------------------------------------

MixtureType = Literal[
    'rbf',
    'kmeans',
    'gmm',
]

# DECISION plan-2026-07-21-845927c7/D-005: generic keras.layers.Layer constructor
# kwargs that every mixture class accepts and forwards via **kwargs. The per-type
# registry (below) only lists layer-specific params, so without this allowlist
# create_mixture_layer's valid_param_names filter would SILENTLY strip these
# (e.g. dtype='mixed_float16' dropped, layer built as float32, no warning). Mirrors
# norms/factory.py:_KERAS_BASE_PARAMS, the established fix for this exact drop.
_KERAS_BASE_PARAMS = frozenset({'name', 'dtype', 'trainable', 'activity_regularizer', 'autocast'})

# ---------------------------------------------------------------------
# Mixture layer registry mapping types to classes and parameter info
# ---------------------------------------------------------------------

MIXTURE_REGISTRY: Dict[str, Dict[str, Any]] = {
    'rbf': {
        'class': RBFLayer,
        'description': 'Radial Basis Function layer with learnable centers and inter-center repulsion',
        'required_params': ['units'],
        'optional_params': {
            # Mirrors RBFLayer.__init__ exactly: None means "resolve to 1/feature_dim
            # in build()" (D-001). Restoring a concrete 1.0 here reds
            # test_factory_registry_drift.py -- and would be wrong regardless, since
            # this registry advertises constructor defaults, not resolved values.
            'gamma_init': None,
            'repulsion_strength': 0.1,
            'min_center_distance': 1.0,
            'center_initializer': 'uniform',
            'center_constraint': None,
            'trainable_gamma': True,
            'safety_margin': 0.2,
            'kernel_regularizer': None,
            'gamma_regularizer': None,
            # Must mirror RBFLayer.__init__'s default exactly. Omitting it here would
            # make create_mixture_layer's valid_param_names filter SILENTLY DROP a
            # caller-supplied output_mode (test_factory_registry_drift.py also reds).
            'output_mode': 'basis',
        },
        'use_case': 'Localized RBF feature responses; soft prototype / kernel-based representations',
    },
    'kmeans': {
        'class': KMeansLayer,
        'description': 'Differentiable soft K-means layer with online centroid updates and repulsion',
        'required_params': ['n_clusters'],
        'optional_params': {
            'temperature': 0.1,
            'momentum': 0.9,
            'centroid_lr': 0.1,
            'repulsion_strength': 0.1,
            'min_distance': 1.0,
            'output_mode': 'assignments',
            'cluster_axis': -1,
            'centroid_initializer': 'orthonormal',
            'centroid_regularizer': None,
            'random_seed': None,
        },
        'use_case': 'Soft cluster assignments / learned codebooks as a differentiable layer',
    },
    'gmm': {
        'class': GMMLayer,
        'description': 'Differentiable Gaussian Mixture layer with soft responsibilities and isometric regularizer',
        'required_params': ['n_components'],
        'optional_params': {
            'temperature': 1.0,
            'isometric_regularizer_strength': 0.01,
            'variance_floor': 1e-3,
            'output_mode': 'assignments',
            'cluster_axis': -1,
            'mean_initializer': 'orthonormal',
            'log_variance_initializer': 'zeros',
            'mean_regularizer': None,
            'random_seed': None,
            'covariance_type': 'diagonal',
            'covariance_rank': 1,
            'factor_initializer': 'glorot_uniform',
        },
        'use_case': 'Soft probabilistic mixture assignments with learnable per-component variances',
    },
}


# ---------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------

def get_mixture_info() -> Dict[str, Dict[str, Any]]:
    """
    Get comprehensive information about all available mixture layer types.

    :return: Dict containing information about each mixture type, including
        description, required_params, optional_params, and use_case.
    :rtype: Dict[str, Dict[str, Any]]
    """
    return {mixture_type: info.copy() for mixture_type, info in MIXTURE_REGISTRY.items()}


def validate_mixture_config(mixture_type: str, **kwargs: Any) -> None:
    """
    Validate mixture layer configuration parameters.

    :param mixture_type: Type of mixture layer to validate.
    :type mixture_type: str
    :param kwargs: Parameters to validate.
    :raises ValueError: If mixture_type is invalid or required parameters are missing.
    """
    if mixture_type not in MIXTURE_REGISTRY:
        available_types = sorted(list(MIXTURE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown mixture type '{mixture_type}'. "
            f"Available types: {available_types}"
        )

    mixture_info = MIXTURE_REGISTRY[mixture_type]
    required_params = mixture_info['required_params']

    # Check for required parameters
    missing_params = [param for param in required_params if param not in kwargs]
    if missing_params:
        raise ValueError(
            f"Required parameters missing for {mixture_type}: {missing_params}. "
            f"Required: {required_params}"
        )

    # Validate positive integer count parameters
    count_params = ['units', 'n_clusters', 'n_components']
    for count_param in count_params:
        if count_param in kwargs and kwargs[count_param] is not None:
            value = kwargs[count_param]
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{count_param} must be a positive integer, got {value}")

    # Validate positive float parameters common across mixtures
    positive_floats = ['temperature', 'gamma_init', 'min_center_distance', 'min_distance', 'variance_floor']
    for float_param in positive_floats:
        if float_param in kwargs and kwargs[float_param] is not None:
            if kwargs[float_param] <= 0:
                raise ValueError(f"{float_param} must be positive, got {kwargs[float_param]}")

    # Validate non-negative parameters
    non_negative = ['repulsion_strength', 'isometric_regularizer_strength', 'safety_margin']
    for nn_param in non_negative:
        if nn_param in kwargs and kwargs[nn_param] is not None:
            if kwargs[nn_param] < 0:
                raise ValueError(f"{nn_param} must be non-negative, got {kwargs[nn_param]}")

    # Validate output_mode. DECISION plan-2026-07-20T160907-7de371a1/D-003: the legal
    # value set is per-mixture_type, NOT shared. RBFLayer reuses the kwarg NAME for
    # house-style consistency but has a disjoint vocabulary ('basis'/'normalized'), as
    # it has no reconstruction-mode analogue to 'mixture'. Do NOT collapse this back to
    # a single shared literal set: that would make the factory REJECT RBF's own legal
    # values with an error naming {'assignments','mixture'}. Do NOT generalize it into
    # MIXTURE_REGISTRY either -- the registry carries no validation-type metadata, so
    # that is schema design, explicitly cut from this plan's scope (F15).
    if 'output_mode' in kwargs and kwargs['output_mode'] is not None:
        valid_modes = (
            {'basis', 'normalized'} if mixture_type == 'rbf'
            else {'assignments', 'mixture'}
        )
        if kwargs['output_mode'] not in valid_modes:
            raise ValueError(
                f"output_mode must be one of {sorted(valid_modes)}, got '{kwargs['output_mode']}'"
            )


def create_mixture_layer(
        mixture_type: MixtureType,
        name: Optional[str] = None,
        **kwargs: Any
) -> keras.layers.Layer:
    """
    Factory function for creating mixture layers with a unified interface.

    This function provides a centralized way to create any mixture layer
    supported by this sub-package, with parameter validation, unknown-kwarg
    filtering, and consistent error handling.

    :param mixture_type: Type of mixture layer to create. See ``MixtureType``
        for all supported types ('rbf', 'kmeans', 'gmm').
    :type mixture_type: MixtureType
    :param name: Optional name for the layer.
    :type name: Optional[str]
    :param kwargs: Parameters specific to the mixture type. See the individual
        layer documentation for parameter details.
    :return: Configured mixture layer instance.
    :rtype: keras.layers.Layer
    :raises ValueError: If mixture_type is invalid or required parameters are missing.
    """
    try:
        # Validate configuration
        validate_mixture_config(mixture_type, **kwargs)

        # Get mixture info and class
        mixture_info = MIXTURE_REGISTRY[mixture_type]
        mixture_class = mixture_info['class']

        # Get all valid parameter names for this mixture_type. Union in the generic
        # keras.layers.Layer kwargs (_KERAS_BASE_PARAMS, D-005) so they are forwarded to
        # the layer instead of being silently dropped by the final_params filter below.
        valid_param_names = (
            set(mixture_info['required_params'])
            | set(mixture_info['optional_params'].keys())
            | _KERAS_BASE_PARAMS
        )

        # Start with defaults for all optional parameters, then user overrides
        params: Dict[str, Any] = {}
        params.update(mixture_info['optional_params'])
        params.update(kwargs)

        # Filter out any unknown parameters to avoid "Unrecognized keyword arguments" error
        final_params = {key: val for key, val in params.items() if key in valid_param_names}

        # Add name if provided
        if name is not None:
            final_params['name'] = name

        logger.info(f"Creating {mixture_type} mixture layer with parameters:")
        log_params = final_params.copy()
        for param_name, param_value in sorted(log_params.items()):
            if param_name == 'name':
                logger.info(f"  {param_name}: '{param_value}'")
            elif isinstance(param_value, str):
                logger.info(f"  {param_name}: '{param_value}'")
            elif param_value is None:
                logger.info(f"  {param_name}: None")
            else:
                logger.info(f"  {param_name}: {param_value}")

        # Create mixture layer using registry class directly (no if/elif chain)
        mixture_layer = mixture_class(**final_params)

        logger.debug(f"Successfully created {mixture_type} mixture layer: {mixture_layer.name}")
        return mixture_layer

    except (TypeError, ValueError) as e:
        mixture_info = MIXTURE_REGISTRY.get(mixture_type)
        if mixture_info:
            required_params = mixture_info.get('required_params', [])
            provided_params = list(kwargs.keys())
            class_name = mixture_info.get('class', type(None)).__name__
            error_msg = (
                f"Failed to create {mixture_type} mixture layer ({class_name}). "
                f"Required parameters: {required_params}. "
                f"Provided parameters: {provided_params}. "
                f"Check parameter compatibility and types. "
                f"Use get_mixture_info() for detailed parameter information. "
                f"Original error: {e}"
            )
        else:
            error_msg = (
                f"Failed to create mixture layer. Unknown mixture type "
                f"'{mixture_type}'. Original error: {e}"
            )

        logger.error(error_msg)
        raise ValueError(error_msg) from e


def create_mixture_from_config(config: Dict[str, Any]) -> keras.layers.Layer:
    """
    Create a mixture layer from a configuration dictionary.

    :param config: Configuration dictionary containing a 'type' key and parameters.
    :type config: Dict[str, Any]
    :return: Configured mixture layer instance.
    :rtype: keras.layers.Layer
    :raises ValueError: If 'type' key is missing from config.
    """
    if not isinstance(config, dict):
        raise ValueError(f"config must be a dictionary, got {type(config)}")

    if 'type' not in config:
        raise ValueError("Configuration must include 'type' key")

    config_copy = config.copy()
    mixture_type = config_copy.pop('type')

    logger.debug(
        f"Creating mixture from config - type: {mixture_type}, "
        f"params: {list(config_copy.keys())}"
    )

    return create_mixture_layer(mixture_type, **config_copy)

# ---------------------------------------------------------------------
