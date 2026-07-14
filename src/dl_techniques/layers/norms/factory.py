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
import inspect
from typing import Optional, Dict, Any, Literal, Set

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .rms_norm import RMSNorm
from .bias_free_batch_norm import BiasFreeBatchNorm
from .band_rms import BandRMS
from .adaptive_band_rms import AdaptiveBandRMS
from .band_logit_norm import BandLogitNorm
from .global_response_norm import GlobalResponseNormalization
from .logit_norm import LogitNorm
from .max_logit_norm import MaxLogitNorm, DecoupledMaxLogit, DMLPlus
from .dynamic_tanh import DynamicTanh
from .energy_layer_norm import EnergyLayerNorm
from .zero_centered_rms_norm import ZeroCenteredRMSNorm
from .zero_centered_band_rms_norm import ZeroCenteredBandRMSNorm
from .zero_centered_adaptive_band_rms_norm import ZeroCenteredAdaptiveBandRMS

# ---------------------------------------------------------------------

NormalizationType = Literal[
    'layer_norm', 'batch_norm', 'bias_free_batch_norm', 'rms_norm', 'zero_centered_rms_norm',
    'zero_centered_band_rms_norm', 'zero_centered_adaptive_band_rms_norm',
    'band_rms', 'adaptive_band_rms',
    'band_logit_norm', 'global_response_norm', 'logit_norm', 'max_logit_norm',
    'decoupled_max_logit', 'dml_plus_focal', 'dml_plus_center', 'dynamic_tanh',
    'energy_layer_norm'
]

# ---------------------------------------------------------------------
# Validation whitelist — DERIVED from the real constructors, never hand-maintained.
# ---------------------------------------------------------------------

# The class each type actually instantiates. Single source of truth, kept honest by
# `test_type_to_class_matches_what_the_builder_returns` (which builds every type and
# compares `type(layer)` against this map), so it cannot silently drift from the
# if/elif chain in `create_normalization_layer`.
_TYPE_TO_CLASS: Dict[str, type] = {
    'layer_norm': keras.layers.LayerNormalization,
    'batch_norm': keras.layers.BatchNormalization,
    'bias_free_batch_norm': BiasFreeBatchNorm,
    'rms_norm': RMSNorm,
    'zero_centered_rms_norm': ZeroCenteredRMSNorm,
    'zero_centered_band_rms_norm': ZeroCenteredBandRMSNorm,
    'zero_centered_adaptive_band_rms_norm': ZeroCenteredAdaptiveBandRMS,
    'band_rms': BandRMS,
    'adaptive_band_rms': AdaptiveBandRMS,
    'band_logit_norm': BandLogitNorm,
    'global_response_norm': GlobalResponseNormalization,
    'logit_norm': LogitNorm,
    'max_logit_norm': MaxLogitNorm,
    'decoupled_max_logit': DecoupledMaxLogit,
    'dml_plus_focal': DMLPlus,
    'dml_plus_center': DMLPlus,
    'dynamic_tanh': DynamicTanh,
    'energy_layer_norm': EnergyLayerNorm,
}

# Named parameters of `create_normalization_layer` ITSELF, valid for every type.
# `epsilon` is deliberately universal: the factory takes it for all types and adapts it
# per-layer (aliased to `eps` for GRN, dropped for `dynamic_tanh`) — see that function's
# docstring. Rejecting it for those two types made the validator disagree with the builder.
_FACTORY_LEVEL_PARAMS = frozenset({'name', 'epsilon'})

# Base `keras.layers.Layer` kwargs. Every target class takes `**kwargs` and forwards them,
# so these genuinely build; the validator must not reject what the builder accepts.
_KERAS_BASE_PARAMS = frozenset({'name', 'dtype', 'trainable', 'activity_regularizer', 'autocast'})

# Parameters the factory IGNORES. The rule below is "the validator must accept whatever
# the builder accepts" — but *builds without raising* is NOT the same as *has an effect*.
# These two sets are the params a caller may pass, that construct fine, and that the
# factory then throws away. Rejecting them is CORRECT and is NOT the drift being fixed:
# it tells a caller their value is doing nothing, instead of silently ignoring it.
#
# OVERWRITTEN: `create_normalization_layer` hard-assigns `model_type` for the two DML+
# variants, so a caller's value is clobbered.
_FACTORY_OWNED_PARAMS: Dict[str, frozenset] = {
    'dml_plus_focal': frozenset({'model_type'}),
    'dml_plus_center': frozenset({'model_type'}),
}

# DISCARDED: `DynamicTanh` has no epsilon, and the factory `pop`s it. A config-driven
# caller who sets `epsilon=1e-3` here and is not told would reasonably believe it applied.
# (Contrast `global_response_norm`, which ALIASES `epsilon` -> `eps`: there it is
# meaningful, so it is accepted — that one WAS drift, and is now fixed.)
_FACTORY_DROPPED_PARAMS: Dict[str, frozenset] = {
    'dynamic_tanh': frozenset({'epsilon'}),
}


def _accepted_params(normalization_type: str) -> Set[str]:
    """Return every kwarg `create_normalization_layer` genuinely accepts for a type.

    DERIVED from ``inspect.signature`` of the target class, NOT from a hand-maintained
    list. This is the whole point: a hand-kept whitelist drifts the moment someone adds a
    constructor argument, and the validator then rejects a parameter the builder happily
    accepts. That has now happened twice (F1, plan_2026-06-15_2485b951 — band/GRN
    initializers; and `gamma_constraint` on `energy_layer_norm`,
    plan_2026-07-14_e5955791/D-004), which is why the mechanism, not the list, was fixed.

    ``get_normalization_info()[t]['parameters']`` remains a DOCUMENTATION surface — a
    curated list of the parameters people commonly pass. It is deliberately NOT the
    validation whitelist any more, so it can be incomplete without breaking a caller.

    :param normalization_type: A registered normalization type.
    :type normalization_type: str
    :return: The set of accepted keyword-argument names.
    :rtype: Set[str]
    """
    cls = _TYPE_TO_CLASS[normalization_type]
    signature = inspect.signature(cls.__init__)
    named = {
        name for name, param in signature.parameters.items()
        if name != 'self'
        and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    }
    named |= _FACTORY_LEVEL_PARAMS
    named |= _KERAS_BASE_PARAMS
    named -= _FACTORY_OWNED_PARAMS.get(normalization_type, frozenset())
    named -= _FACTORY_DROPPED_PARAMS.get(normalization_type, frozenset())
    return named


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
        include 'layer_norm', 'batch_norm', 'bias_free_batch_norm', 'rms_norm',
        'zero_centered_rms_norm', 'zero_centered_band_rms_norm',
        'zero_centered_adaptive_band_rms_norm', 'band_rms', 'adaptive_band_rms',
        'band_logit_norm', 'global_response_norm', 'logit_norm', 'max_logit_norm',
        'decoupled_max_logit', 'dml_plus_focal', 'dml_plus_center', 'dynamic_tanh',
        and 'energy_layer_norm'.
        Use 'bias_free_batch_norm' for the variance-only, fixed-statistic layer that
        stays degree-1 homogeneous (``f(a*x)=a*f(x)``) at inference — no ``moving_mean``,
        no ``beta`` — as required by bias-free / Miyasawa denoisers.
    :type normalization_type: NormalizationType
    :param name: Optional name for the layer. If None, layer will use default naming.
    :type name: Optional[str]
    :param epsilon: Small constant for numerical stability. Defaults to 1e-6.
        Used by normalization layers that support an epsilon parameter. NOTE: this
        1e-6 default is imposed by the factory (via ``setdefault``) and may differ
        from a layer class's own default — the custom RMS-family classes default to
        1e-7, and Keras ``LayerNormalization``/``BatchNormalization`` default to
        1e-3. Pass ``epsilon`` explicitly to control it; instantiate the class
        directly to get its own default.
    :type epsilon: float
    :param kwargs: Additional keyword arguments specific to each normalization type.
        Common kwargs include axis, center, scale, use_scale, momentum (for batch_norm
        / bias_free_batch_norm), max_band_width, temperature, constant, alpha_init_value,
        and eps (for GRN).
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

    elif normalization_type == 'bias_free_batch_norm':
        # Variance-only, fixed-statistic normalization; degree-1 homogeneous at
        # inference (no moving_mean, no beta). See bias_free_batch_norm.py.
        layer_kwargs.setdefault('epsilon', epsilon)
        return BiasFreeBatchNorm(**layer_kwargs)

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

    elif normalization_type == 'zero_centered_adaptive_band_rms_norm':
        # Zero-centered adaptive RMS with log-transformed scaling
        layer_kwargs.setdefault('epsilon', epsilon)
        return ZeroCenteredAdaptiveBandRMS(**layer_kwargs)

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

    elif normalization_type == 'energy_layer_norm':
        # Energy Transformer layer norm (arXiv:2302.07253 eq. 1-2):
        # SCALAR gamma + VECTOR delta. See energy_layer_norm.py.
        layer_kwargs.setdefault('epsilon', epsilon)
        return EnergyLayerNorm(**layer_kwargs)

    else:
        supported_types = [
            'layer_norm', 'batch_norm', 'bias_free_batch_norm', 'rms_norm',
            'zero_centered_rms_norm',
            'zero_centered_band_rms_norm',
            'zero_centered_adaptive_band_rms_norm',
            'band_rms', 'adaptive_band_rms',
            'band_logit_norm', 'global_response_norm', 'logit_norm',
            'max_logit_norm', 'decoupled_max_logit', 'dml_plus_focal',
            'dml_plus_center', 'dynamic_tanh', 'energy_layer_norm'
        ]
        raise ValueError(
            f"Unknown normalization type: '{normalization_type}'. "
            f"Supported types: {', '.join(supported_types)}"
        )


# ---------------------------------------------------------------------


def get_normalization_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all supported normalization types and their parameters.

    .. warning::
       The ``'parameters'`` list is **documentation** — a curated set of the parameters
       callers commonly pass. It is **NOT** the validation whitelist and is **NOT**
       guaranteed exhaustive: ``layer_norm`` and ``batch_norm``, for instance, accept
       every Keras ``LayerNormalization`` / ``BatchNormalization`` kwarg, which this list
       does not enumerate.

       For the authoritative set of accepted kwargs, use
       :func:`validate_normalization_config`, whose whitelist is DERIVED from the target
       class's real constructor signature (see ``_accepted_params``). A hand-maintained
       whitelist drifted twice and made the validator reject parameters the builder
       accepts; deriving it makes that class of bug unrepresentable.

    :return: Dictionary mapping normalization type names to their parameter information,
        including description, commonly-used parameters, and usage notes.
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
        'bias_free_batch_norm': {
            'description': 'Variance-only, fixed-statistic normalization (no moving_mean, no beta); degree-1 homogeneous at inference',
            'parameters': ['axis', 'epsilon', 'momentum', 'use_scale'],
            'use_case': 'Bias-free / homogeneous architectures (e.g. Miyasawa denoisers) requiring f(a*x)=a*f(x) at inference'
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
        'zero_centered_adaptive_band_rms_norm': {
            'description': 'Zero-centered RMS with adaptive log-transformed RMS-based scaling',
            'parameters': ['max_band_width', 'axis', 'epsilon', 'band_initializer', 'band_regularizer'],
            'use_case': 'Advanced training stability combining zero-centering with input-adaptive scaling'
        },
        'band_rms': {
            'description': 'RMS normalization with bounded magnitude constraints',
            'parameters': ['max_band_width', 'axis', 'epsilon', 'band_initializer', 'band_regularizer'],
            'use_case': 'Training stability in deep networks with gradient control'
        },
        'adaptive_band_rms': {
            'description': 'Adaptive RMS with log-transformed RMS-based scaling',
            'parameters': ['max_band_width', 'axis', 'epsilon', 'band_initializer', 'band_regularizer'],
            'use_case': 'Advanced training stability with adaptive scaling'
        },
        'band_logit_norm': {
            'description': 'Band-constrained logit normalization for classification',
            'parameters': ['max_band_width', 'axis', 'epsilon'],
            'use_case': 'Classification tasks with logit magnitude control'
        },
        'global_response_norm': {
            'description': 'Global Response Normalization from ConvNeXt',
            'parameters': ['eps', 'gamma_initializer', 'beta_initializer',
                           'gamma_regularizer', 'beta_regularizer', 'activity_regularizer'],
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
            'parameters': ['axis', 'alpha_init_value', 'kernel_initializer',
                           'bias_initializer', 'kernel_regularizer', 'bias_regularizer',
                           'kernel_constraint', 'bias_constraint'],
            'use_case': 'Normalization-free transformer architectures'
        },
        'energy_layer_norm': {
            'description': 'Energy Transformer layer norm (arXiv:2302.07253 eq. 1-2): SCALAR gamma + VECTOR delta; g = dL/dx of a Lagrangian with a PSD Hessian',
            # DECISION plan_2026-07-14_e5955791/D-004 (SUPERSEDED in mechanism, kept for
            # history): `gamma_constraint` is a REAL ctor kwarg (57c9833e/D-010 — it pins
            # `gamma > 0`, which is what keeps the Lagrangian's Hessian PSD). It was added
            # to `EnergyLayerNorm` and NOT to this list, so `validate_normalization_config()`
            # REJECTED a parameter `create_normalization_layer()` happily accepted.
            # That was patched by adding the string here — which did NOT fix the mechanism,
            # and 17 more such disagreements were then found across `layer_norm`,
            # `batch_norm` and `global_response_norm`. The validator now DERIVES its
            # whitelist from the real ctor signature (`_accepted_params`), so this list is
            # documentation only and can no longer break a caller by being incomplete.
            'parameters': ['epsilon', 'gamma_initializer', 'delta_initializer',
                           'gamma_constraint'],
            'use_case': 'Energy Transformer blocks, where the norm must be the derivative of a Lagrangian for the energy-descent guarantee to hold'
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
    if normalization_type not in _TYPE_TO_CLASS:
        raise ValueError(f"Unknown normalization type: {normalization_type}")

    # DERIVED from the target class's real signature — NOT from
    # `get_normalization_info()['parameters']`, which is a curated documentation list and
    # drifts. Using a hand-maintained list as the whitelist made this function reject
    # parameters `create_normalization_layer` accepts, twice (see `_accepted_params`).
    # The invariant, pinned by `TestValidatorAgreesWithBuilder`:
    #     anything the BUILDER accepts, the VALIDATOR must accept.
    valid_params = _accepted_params(normalization_type)
    provided_params = set(kwargs.keys())

    # Check for invalid parameters
    invalid_params = provided_params - valid_params
    if invalid_params:
        raise ValueError(
            f"Invalid parameters for {normalization_type}: {invalid_params}. "
            f"Valid parameters: {sorted(valid_params)}"
        )

    # Type-specific validations
    if normalization_type in ['band_rms', 'adaptive_band_rms', 'band_logit_norm', 'zero_centered_band_rms_norm', 'zero_centered_adaptive_band_rms_norm']:
        if 'max_band_width' in kwargs:
            max_band_width = kwargs['max_band_width']
            if (not isinstance(max_band_width, (int, float))
                    or max_band_width <= 0 or max_band_width >= 1):
                raise ValueError(
                    f"max_band_width must be between 0 and 1, got {max_band_width}"
                )

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

    if normalization_type == 'bias_free_batch_norm':
        if 'momentum' in kwargs:
            momentum = kwargs['momentum']
            if not isinstance(momentum, (int, float)) or not (0.0 <= momentum <= 1.0):
                raise ValueError("momentum must be a number in [0, 1]")
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