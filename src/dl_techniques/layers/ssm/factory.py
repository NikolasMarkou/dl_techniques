"""State space layer factory: one registry, one construction path.

``SSM_REGISTRY`` maps 4 string keys to the SSM layers in this package plus
their metadata. ``create_ssm_layer`` is the single construction path: it
looks the key up, rejects any keyword the target type does not declare,
fills in the registry defaults, and constructs. Nothing on that path
filters and drops silently.

The registry's key set, the ``SsmType`` literals, and every entry's
``required_params`` / ``optional_params`` are public API, consumed by
config-driven callers and asserted by
``tests/test_layers/test_factory_registry_drift.py``. Adding, renaming or
removing one is a breaking change.

Public functions: ``get_ssm_info()``, ``list_ssm_types()``,
``get_ssm_requirements()``, ``validate_ssm_config()``,
``assemble_ssm_config()``, ``create_ssm_layer()``,
``create_ssm_from_config()``.

Registered types:

.. code-block:: text

    selective_ssm   SelectiveSSMLayer
    context_mamba   ContextMambaLayer
    mamba           MambaLayer (paper-named alias of SelectiveSSMLayer)
    mamba2          Mamba2Layer (multi-head SSD)
"""

import keras
from typing import Dict, Any, Literal, Mapping, Optional, List, Sequence

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

from .selective_ssm import SelectiveSSMLayer, MambaLayer
from .context_mamba import ContextMambaLayer
from .mamba2 import Mamba2Layer

# ---------------------------------------------------------------------
# Type Definitions
# ---------------------------------------------------------------------

SsmType = Literal['selective_ssm', 'context_mamba', 'mamba', 'mamba2']

# ---------------------------------------------------------------------
# SSM Layer Registry
# ---------------------------------------------------------------------

SSM_REGISTRY: Dict[str, Dict[str, Any]] = {
    'selective_ssm': {
        'class': SelectiveSSMLayer,
        'description': (
            'Input-dependent selective state space mixer (S6, Mamba v1): '
            'causal depthwise convolution followed by a sequential scan whose '
            'discretization step and B/C maps are projected from the input at '
            'every timestep. Linear complexity in sequence length.'
        ),
        'required_params': ['d_model'],
        'optional_params': {
            'd_state': 16,
            'd_conv': 4,
            'expand': 2,
            'dt_rank': 'auto',
            'dt_min': 0.001,
            'dt_max': 0.1,
            'dt_init': 'random',
            'dt_scale': 1.0,
            'dt_init_floor': 1e-4,
            'conv_bias': True,
            'use_bias': False,
            'layer_idx': None,
        },
        'use_case': (
            'Long-sequence temporal modeling where quadratic attention is '
            'prohibitive: video frame flows, token sequences, and the '
            'MambaLCT context scanner (via context_mamba).'
        ),
        'complexity': 'O(S * D * N) for S tokens, D channels, N states',
        'paper': 'Mamba: Linear-Time Sequence Modeling with Selective State Spaces',
    },
    'context_mamba': {
        'class': ContextMambaLayer,
        'description': (
            'Unidirectional temporal Context Mamba for MambaLCT: prepends '
            '(B, Nc, D) context tokens and appends one learnable (ones-initialized) '
            'bridge token '
            'around flattened (B, T, L, D) frame features, runs them through '
            'projection, normalization and a selective SSM, then splits back '
            'into enhanced features and the history-aggregated context update.'
        ),
        'required_params': ['d_model'],
        'optional_params': {
            'd_state': 16,
            'd_conv': 4,
            'expand': 2,
            'dt_rank': 'auto',
            'dt_min': 0.001,
            'dt_max': 0.1,
            'dt_init': 'random',
            'dt_scale': 1.0,
            'dt_init_floor': 1e-4,
            'conv_bias': True,
            'use_bias': False,
            'normalization_type': 'layer_norm',
            'norm_epsilon': 1e-5,
        },
        'use_case': (
            'MambaLCT long-term visual tracking: carry target-change cues '
            'from the first frame to the current one with linear cost.'
        ),
        'complexity': 'O((T*L + Nc) * D * N)',
        'paper': 'MambaLCT: Boosting Tracking via Long-term Context State Space Model',
    },
    'mamba': {
        'class': MambaLayer,
        'description': (
            'Paper-named alias of SelectiveSSMLayer: the Mamba v1 selective '
            'state space mixer with identical behavior and defaults. Prefer '
            'this key when porting paper-named configs.'
        ),
        'required_params': ['d_model'],
        'optional_params': {
            'd_state': 16,
            'd_conv': 4,
            'expand': 2,
            'dt_rank': 'auto',
            'dt_min': 0.001,
            'dt_max': 0.1,
            'dt_init': 'random',
            'dt_scale': 1.0,
            'dt_init_floor': 1e-4,
            'conv_bias': True,
            'use_bias': False,
            'layer_idx': None,
        },
        'use_case': (
            'Paper-named door onto the selective SSM for language and '
            'temporal sequence modeling.'
        ),
        'complexity': 'O(S * D * N) for S tokens, D channels, N states',
        'paper': 'Mamba: Linear-Time Sequence Modeling with Selective State Spaces',
    },
    'mamba2': {
        'class': Mamba2Layer,
        'description': (
            'Mamba-2 selective SSM on the State Space Duality framework: '
            'grouped multi-head recurrence with a parallel gated-MLP path, '
            'per-head decay and skip, and an RMSNorm gate.'
        ),
        'required_params': ['d_model'],
        'optional_params': {
            'd_state': 128,
            'd_conv': 4,
            'expand': 2,
            'headdim': 64,
            'ngroups': 1,
            'd_ssm': None,
            'rmsnorm': True,
            'norm_epsilon': 1e-5,
            'norm_before_gate': False,
            'dt_min': 0.001,
            'dt_max': 0.1,
            'dt_init_floor': 1e-4,
            'bias': False,
            'conv_bias': True,
        },
        'use_case': (
            'Long-sequence modeling with multi-head SSM structure and grouped '
            'B/C projections (Zamba2, HNet mixers, Mamba-2 LMs).'
        ),
        'complexity': 'O(S * H * P * N) for S tokens, H heads, P head dim, N states',
        'paper': 'Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality',
    },
}
"""
Registry of SSM layer implementations with metadata.

Each entry contains:
    - class: The actual layer class implementation.
    - description: Technical description of the mechanism.
    - required_params: List of mandatory parameters for instantiation.
    - optional_params: Dict of optional parameters with default values.
    - use_case: Scenarios and applications where this layer excels.
    - complexity: Computational complexity analysis.
    - paper: Reference to the original research paper.
"""


# ---------------------------------------------------------------------
# Public API Functions
# ---------------------------------------------------------------------

def get_ssm_info() -> Dict[str, Dict[str, Any]]:
    """Return the metadata of every registered SSM type.

    :return: Mapping from SSM type to its metadata.
    """
    return {ssm_type: info.copy() for ssm_type, info in SSM_REGISTRY.items()}


def list_ssm_types() -> List[str]:
    """Return all registered SSM type names.

    :return: Sorted list of SSM type names.
    """
    return sorted(SSM_REGISTRY.keys())


def get_ssm_requirements(ssm_type: str) -> Dict[str, Any]:
    """Return the parameter requirements for one SSM type.

    :param ssm_type: A registered SSM type name.
    :return: Dict with ``required_params`` and ``optional_params``.
    :raises ValueError: If ``ssm_type`` is unknown.
    """
    if ssm_type not in SSM_REGISTRY:
        raise ValueError(
            f"Unknown SSM type '{ssm_type}'. "
            f"Available types: {sorted(SSM_REGISTRY.keys())}"
        )
    info = SSM_REGISTRY[ssm_type]
    return {
        'required_params': list(info['required_params']),
        'optional_params': dict(info['optional_params']),
    }


#: Stable substring every strict dropped-key ``ValueError`` carries. Defined
#: before its users (unlike a later-defined constant, this cannot break if
#: the module is ever partially reordered).
STRICT_DROPPED_KEY_MARKER: str = "unsupported parameter(s)"


def validate_ssm_config(ssm_type: str, **kwargs: Any) -> None:
    """Check a configuration without building anything.

    :param ssm_type: The SSM layer type to validate against.
    :param kwargs: The parameters to validate for that type.
    :raises ValueError: If a keyword is undeclared, the type is unknown,
        a required parameter is missing, or a value violates its range.
    """
    _info = SSM_REGISTRY.get(ssm_type)
    if _info is not None:
        _declared = set(_info['required_params']) | set(
            _info['optional_params'].keys()
        )
        _undeclared = sorted(set(kwargs) - _declared)
        if _undeclared:
            raise ValueError(
                f"validate_ssm_config('{ssm_type}'): "
                f"{len(_undeclared)} {STRICT_DROPPED_KEY_MARKER} {_undeclared}. "
                f"'{ssm_type}' ({_info['class'].__name__}) accepts only "
                f"{sorted(_declared)}."
            )
    if ssm_type not in SSM_REGISTRY:
        raise ValueError(
            f"Unknown SSM type '{ssm_type}'. "
            f"Available types: {sorted(SSM_REGISTRY.keys())}"
        )

    info = SSM_REGISTRY[ssm_type]
    required = info['required_params']
    missing = [p for p in required if p not in kwargs]
    if missing:
        raise ValueError(
            f"Required parameters for '{ssm_type}' are missing: {missing}. "
            f"Required: {required}, Provided: {list(kwargs.keys())}"
        )

    for param in ('d_model', 'd_state', 'd_conv', 'expand', 'headdim', 'ngroups'):
        if param in kwargs and kwargs[param] is not None and kwargs[param] <= 0:
            raise ValueError(
                f"Parameter '{param}' must be positive, got {kwargs[param]}"
            )
    if 'dt_rank' in kwargs and isinstance(kwargs['dt_rank'], int):
        if kwargs['dt_rank'] <= 0:
            raise ValueError(
                f"Parameter 'dt_rank' must be positive, got {kwargs['dt_rank']}"
            )
    for param in ('dt_min', 'dt_max', 'dt_scale', 'dt_init_floor', 'norm_epsilon'):
        if param in kwargs and kwargs[param] is not None and kwargs[param] <= 0:
            raise ValueError(
                f"Parameter '{param}' must be positive, got {kwargs[param]}"
            )

    logger.debug(f"Validation successful for '{ssm_type}' with parameters: {kwargs}")


#: Keys :func:`create_ssm_layer` accepts that are not registry params.
_SSM_CONFIG_PASSTHROUGH_KEYS: Sequence[str] = ('name',)


def assemble_ssm_config(
        ssm_type: str,
        wrapper_config: Mapping[str, Any],
        caller_args: Optional[Mapping[str, Any]] = None,
        *,
        passthrough: Sequence[str] = _SSM_CONFIG_PASSTHROUGH_KEYS,
) -> Dict[str, Any]:
    """Filter a wrapper's generic defaults, then merge the caller's args.

    :param ssm_type: An ``SSM_REGISTRY`` key.
    :param wrapper_config: The wrapper's own generic config; filtered.
    :param caller_args: The caller's explicit args; never filtered.
    :param passthrough: Keys kept regardless of the registry intersection.
    :return: The assembled config dict for :func:`create_ssm_layer`.
    :raises ValueError: If ``ssm_type`` is not registered.
    """
    info = SSM_REGISTRY.get(ssm_type)
    if info is None:
        raise ValueError(
            f"Unknown ssm_type '{ssm_type}'. "
            f"Available: {sorted(SSM_REGISTRY)}."
        )
    accepted = (
        set(info['required_params'])
        | set(info['optional_params'])
        | set(passthrough)
    )
    config = {k: v for k, v in wrapper_config.items() if k in accepted}
    if caller_args:
        config.update(caller_args)
    return config


def create_ssm_layer(
        ssm_type: SsmType,
        name: Optional[str] = None,
        **kwargs: Any
) -> keras.layers.Layer:
    """Build one SSM layer. An undeclared keyword raises.

    :param ssm_type: The type of SSM layer to create.
    :param name: Optional name for the layer instance.
    :param kwargs: Type-specific parameters. Any key ``ssm_type`` does not
        accept raises rather than being silently dropped.
    :return: A fully configured SSM layer.
    :raises ValueError: If the type is unknown, required parameters are
        missing, values are out of range, a keyword is undeclared, or
        construction fails.
    """
    _info = SSM_REGISTRY.get(ssm_type)
    if _info is not None:
        _valid_param_names = set(_info['required_params']) | set(
            _info['optional_params'].keys()
        )
        dropped = sorted(set(kwargs) - _valid_param_names)
        if dropped:
            raise ValueError(
                f"create_ssm_layer('{ssm_type}'): "
                f"{len(dropped)} {STRICT_DROPPED_KEY_MARKER} {dropped}. "
                f"'{ssm_type}' ({_info['class'].__name__}) accepts only "
                f"{sorted(_valid_param_names)}. "
                f"Either you mistyped one of those names, or you chose the "
                f"wrong ssm_type for the parameters you are passing. If "
                f"these keys are a WRAPPER's own generic defaults rather than "
                f"an explicit request, pre-filter them with "
                f"assemble_ssm_config() instead of passing them here."
            )

    try:
        validate_ssm_config(ssm_type, **kwargs)

        info = SSM_REGISTRY[ssm_type]
        ssm_class = info['class']

        params = info['optional_params'].copy()
        params.update(kwargs)

        valid_param_names = set(info['required_params']) | set(
            info['optional_params'].keys()
        )
        final_params = {
            k: v for k, v in params.items() if k in valid_param_names
        }

        if name:
            final_params['name'] = name

        logger.info(
            f"Creating '{ssm_type}' SSM layer "
            f"({ssm_class.__name__}) with parameters: {final_params}"
        )

        return ssm_class(**final_params)

    except (TypeError, ValueError) as e:
        info = SSM_REGISTRY.get(ssm_type)
        if info:
            class_name = info['class'].__name__
            error_msg = (
                f"Failed to create '{ssm_type}' SSM layer "
                f"({class_name}). "
                f"Required parameters: {info['required_params']}. "
                f"Provided parameters: {list(kwargs.keys())}. "
                f"Please verify parameter compatibility. Original error: {e}"
            )
        else:
            error_msg = (
                f"Failed to create SSM layer. "
                f"Unknown type '{ssm_type}'. Error: {e}"
            )

        logger.error(error_msg)
        raise ValueError(error_msg) from e


def create_ssm_from_config(config: Dict[str, Any]) -> keras.layers.Layer:
    """Build an SSM layer from a single ``{'type': ..., ...}`` dict.

    :param config: Configuration dict with a ``'type'`` key naming the SSM
        type, plus that type's parameters.
    :return: The instantiated SSM layer.
    :raises ValueError: If ``config`` is not a dict, has no ``'type'`` key,
        or :func:`create_ssm_layer` raises.
    """
    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration must be a dictionary, got {type(config).__name__}. "
            f"Expected format: {{'type': 'ssm_type', ...}}"
        )

    if 'type' not in config:
        available_keys = list(config.keys()) if config else []
        raise ValueError(
            f"Configuration dictionary must include a 'type' key specifying the "
            f"SSM layer type. Available keys in config: {available_keys}. "
            f"Valid types: {sorted(SSM_REGISTRY.keys())}"
        )

    config_copy = config.copy()
    ssm_type = config_copy.pop('type')

    logger.debug(f"Creating SSM layer from config: {config}")
    return create_ssm_layer(ssm_type, **config_copy)

# ---------------------------------------------------------------------
