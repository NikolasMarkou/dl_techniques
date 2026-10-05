"""
A Factory Method design pattern providing a single, centralized entry point for
creating the low-rank adapter layers in this sub-package.

Architectural Overview:
    The factory operates on a registry-based design (``ADAPTER_REGISTRY``). This
    registry maps a simple string identifier (the ``adapter_type``) to the
    corresponding Keras Layer class and its associated metadata (required
    parameters, optional parameters with defaults, description, use case).

    When called, the factory:
    1.  **Validates** the requested ``adapter_type`` and rejects any keyword that
        type does not declare (strict -- it raises, never filters-and-drops).
    2.  **Retrieves** the Keras Layer class associated with the type.
    3.  **Fills in** the registry's optional-parameter defaults underneath the
        caller's values, then **instantiates** the class.

Why strict, and not filter-and-drop:
    A factory that silently discards an undeclared keyword converts a typo into
    a model that trains and behaves plausibly while missing the setting the
    caller asked for. Guide §9.2 records four measured instances of exactly that
    damage in this tree (a ``dropout=`` key landing on ``dropout_rate`` killed
    dropout across every vision encoder; ``max_seq_len``/``rope_theta`` landing
    on a type declaring no rotary parameter made a stack exactly
    permutation-equivariant). Any key not declared by the chosen type raises a
    ``ValueError`` whose message carries :data:`STRICT_DROPPED_KEY_MARKER`.

Supported layers:
    -   ``lora``    -- :class:`~dl_techniques.layers.adapters.lora.LoRAAdapter`,
        an additive low-rank delta with one independent ``A``/``B`` pair per
        adapter slot.

References:
    - Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994). Design
      Patterns: Elements of Reusable Object-Oriented Software. Addison-Wesley.
"""

import copy
import keras
from typing import Any, Dict, List, Literal, Optional, Sequence

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ...utils.logger import logger

from .lora import LoRAAdapter

# ---------------------------------------------------------------------
# Type definition for adapter types
# ---------------------------------------------------------------------

AdapterType = Literal[
    'lora',
]

#: Generic ``keras.layers.Layer`` constructor kwargs every adapter class accepts
#: and forwards via ``**kwargs``. The per-type ``optional_params`` below are
#: UNIONED with this set when validating, so passing ``name``/``dtype``/
#: ``trainable`` through a config dict is not reported as an undeclared key.
_KERAS_BASE_PARAMS = frozenset(
    {'name', 'dtype', 'trainable', 'activity_regularizer', 'autocast'}
)

#: Substring carried by the message raised when an undeclared keyword reaches
#: the factory. Kept as a module constant so a test can assert on the marker
#: instead of on the prose.
STRICT_DROPPED_KEY_MARKER: str = "unsupported parameter(s)"

#: Keys a wrapper may pass through untouched when pre-filtering its own generic
#: defaults via :func:`assemble_adapter_config`.
_ADAPTER_CONFIG_PASSTHROUGH_KEYS: Sequence[str] = ('name',)

# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------

ADAPTER_REGISTRY: Dict[str, Dict[str, Any]] = {
    'lora': {
        'class': LoRAAdapter,
        'description': (
            'Additive low-rank adapter delta with one independent A/B pair per '
            'adapter slot. The slot index is a static Python int supplied per call '
            'site, so one physical layer can specialize at many sites (Zamba2 depth '
            'positions) or across many sequential learning phases (Local Support '
            'Learning) without multiplying the shared layer\'s own parameter count. '
            'Returns a pure delta; the caller adds it onto its own base output.'
        ),
        'required_params': ['output_dim', 'rank', 'alpha', 'num_adapters'],
        'optional_params': {
            'kernel_initializer': 'glorot_uniform',
        },
        'use_case': (
            'Parameter-efficient adaptation of an existing projection. Multi-slot: '
            'the slots are whatever the caller needs to be independent -- depth '
            'positions of a weight-shared block, or the learning phases of a '
            'gated continual-learning adapter.'
        ),
        'complexity': 'O(2 * input_dim * rank * output_dim) FLOPs per slot; '
                      'O(num_adapters * (input_dim + output_dim) * rank) parameters',
        'paper': 'LoRA: Low-Rank Adaptation of Large Language Models',
    },
}


# DECISION: the Literal above and the registry keys are API. A type reachable by
# only passing an argument to the general class is deliberately NOT registered
# (guide §9.3), and a key added here without its Literal alias -- or the reverse
# -- is a drift defect, not a style issue. `tests/test_layers/test_adapters/
# test_the_adapter_registry_and_its_literal_agree.py` pins both directions.
def _check_registry_literal_consistency() -> None:
    """Fail at import if the registry keys and the ``AdapterType`` Literal disagree.

    Raises ``RuntimeError`` rather than ``assert``: ``python -O`` strips asserts,
    and a stripped assert here would ship a dispatcher that silently cannot
    build a registered type.

    :raises RuntimeError: If the two vocabularies differ in either direction.
    """
    from typing import get_args

    literal_names = set(get_args(AdapterType))
    registry_names = set(ADAPTER_REGISTRY.keys())

    missing_from_literal = sorted(registry_names - literal_names)
    missing_from_registry = sorted(literal_names - registry_names)

    if missing_from_literal or missing_from_registry:
        raise RuntimeError(
            "AdapterType and ADAPTER_REGISTRY disagree. "
            f"In the registry but not the Literal: {missing_from_literal}. "
            f"In the Literal but not the registry: {missing_from_registry}. "
            "Add the key to ADAPTER_REGISTRY and the name to AdapterType in "
            "the same commit."
        )


_check_registry_literal_consistency()

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def get_adapter_info() -> Dict[str, Dict[str, Any]]:
    """Return every registered adapter type's metadata entry.

    The returned dict is a **deep copy**. A shallow copy would hand callers the
    live registry's nested ``optional_params`` dict, and one caller mutating
    ``info['lora']['optional_params']['rank']`` would corrupt dispatch for every
    other caller process-wide.

    :return: A deep copy of the registry, keyed by adapter type.
    :rtype: Dict[str, Dict[str, Any]]
    """
    return copy.deepcopy(ADAPTER_REGISTRY)


def _declared_params(adapter_type: str) -> set:
    """The set of keyword names ``adapter_type`` accepts.

    :param adapter_type: The adapter type whose declared parameters to compute.
    :type adapter_type: str
    :return: Required parameter names unioned with optional ones and the Keras
        base kwargs.
    :rtype: set
    :raises KeyError: If ``adapter_type`` is not in the registry.
    """
    info = ADAPTER_REGISTRY[adapter_type]
    return set(info['required_params']) | set(info['optional_params'].keys()) | set(
        _KERAS_BASE_PARAMS
    )


def _validate_positive_int(name: str, value: Any) -> None:
    """Raise unless ``value`` is a genuine positive ``int``.

    ``bool`` is rejected explicitly: it is an ``int`` subclass, so
    ``rank=True`` would otherwise pass as ``rank=1``.

    :param name: Parameter name, for the message.
    :type name: str
    :param value: The value to check.
    :type value: Any
    :raises ValueError: If ``value`` is not a positive Python/NumPy integer.
    """
    import numbers

    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(
            f"{name} must be a positive integer, got {value!r} "
            f"(type {type(value).__name__})"
        )
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_adapter_config(adapter_type: str, **kwargs: Any) -> None:
    """Check a configuration without building anything.

    A pre-flight check for a caller who wants to fail early, or who validates
    here and then constructs the layer class directly. Returns ``None`` on
    success -- the exception is the only signal.

    Checks run in this order: undeclared keyword, unknown type, missing required
    parameter, then value checks.

    :param adapter_type: The adapter type to validate against.
    :type adapter_type: str
    :param kwargs: The parameters to validate for that type.
    :type kwargs: Any
    :raises ValueError: If a keyword is not declared by the type, if
        ``adapter_type`` is unknown, if a required parameter is missing, or if a
        value violates a range or type constraint.
    """
    # DECISION: the undeclared-key check is deliberately duplicated in
    # create_adapter_layer. Removing either copy changes WHICH failure fires
    # first -- here a caller gets the undeclared-key message without paying for
    # construction; there the same check guards the construction path itself.
    # Collapsing them into one helper would make one of the two entry points
    # lose its message. Mirrors layers/attention/factory.py.
    if adapter_type not in ADAPTER_REGISTRY:
        available_types = list(ADAPTER_REGISTRY.keys())
        raise ValueError(
            f"Unknown adapter type '{adapter_type}'. Available types: {available_types}"
        )

    _info = ADAPTER_REGISTRY[adapter_type]
    _declared = _declared_params(adapter_type)
    _undeclared = sorted(set(kwargs) - _declared)
    if _undeclared:
        raise ValueError(
            f"validate_adapter_config('{adapter_type}'): "
            f"{len(_undeclared)} {STRICT_DROPPED_KEY_MARKER} {_undeclared}. "
            f"'{adapter_type}' ({_info['class'].__name__}) accepts only "
            f"{sorted(_declared)}."
        )

    required = _info['required_params']
    missing = [p for p in required if p not in kwargs]
    if missing:
        raise ValueError(
            f"Required parameters for '{adapter_type}' are missing: {missing}. "
            f"Required: {required}, Provided: {list(kwargs.keys())}"
        )

    # Value checks. These live in the factory rather than relying on the target
    # class's own __init__ to reject them: guide §9.2 is explicit that "validate"
    # must not mean "let the constructor happen to notice".
    _validate_positive_int('output_dim', kwargs['output_dim'])
    _validate_positive_int('rank', kwargs['rank'])
    _validate_positive_int('num_adapters', kwargs['num_adapters'])

    alpha = kwargs['alpha']
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError(
            f"alpha must be a positive number, got {alpha!r} "
            f"(type {type(alpha).__name__})"
        )
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")


def assemble_adapter_config(
        adapter_type: str,
        wrapper_config: Dict[str, Any],
        caller_args: Optional[Dict[str, Any]] = None,
        *,
        passthrough: Sequence[str] = _ADAPTER_CONFIG_PASSTHROUGH_KEYS,
) -> Dict[str, Any]:
    """Merge a wrapper's generic defaults under a caller's explicit arguments.

    A wrapper that offers a flat block of settings -- some of which the chosen
    adapter type does not accept -- cannot pass that block straight to
    :func:`create_adapter_layer`, because the factory is strict by design. This
    splits the block into the part the type declares (which goes through and is
    checked) and the part it does not (which is the wrapper's own noise and is
    dropped silently, since rejecting it would break every wrapper that carries
    a superset of knobs).

    ``caller_args`` passes through **unfiltered**: those are the caller's
    explicit requests, and an undeclared key among them must still raise.

    :param adapter_type: The adapter type the config is destined for.
    :type adapter_type: str
    :param wrapper_config: The wrapper's own settings block. Keys the type does
        not declare are dropped.
    :type wrapper_config: Dict[str, Any]
    :param caller_args: The caller's explicit arguments, forwarded unfiltered.
    :type caller_args: Optional[Dict[str, Any]]
    :param passthrough: ``wrapper_config`` keys forwarded regardless of whether
        the type declares them.
    :type passthrough: Sequence[str]
    :return: The merged configuration dict.
    :rtype: Dict[str, Any]
    :raises ValueError: If ``adapter_type`` is unknown.
    """
    if adapter_type not in ADAPTER_REGISTRY:
        available_types = list(ADAPTER_REGISTRY.keys())
        raise ValueError(
            f"Unknown adapter type '{adapter_type}'. Available types: {available_types}"
        )

    declared = _declared_params(adapter_type)
    keep_passthrough = set(passthrough)

    merged: Dict[str, Any] = {}
    dropped: List[str] = []
    for key, value in wrapper_config.items():
        if key in declared or key in keep_passthrough:
            merged[key] = value
        else:
            dropped.append(key)

    if dropped:
        logger.debug(
            f"assemble_adapter_config('{adapter_type}'): dropping "
            f"{len(dropped)} wrapper key(s) the type does not declare: "
            f"{sorted(dropped)}"
        )

    if caller_args:
        merged.update(caller_args)

    return merged


def create_adapter_layer(
        adapter_type: AdapterType,
        name: Optional[str] = None,
        **kwargs: Any
) -> keras.layers.Layer:
    """Build one adapter layer. An undeclared keyword raises.

    The single construction path for every registered type. It looks
    ``adapter_type`` up, rejects any keyword that type does not declare, fills
    the registry defaults in under the caller's values, and constructs.

    Nothing here filters and drops. A wrapper offering generic conveniences that
    only some types accept pre-filters them through
    :func:`assemble_adapter_config` first; whatever it then passes here is
    treated as an explicit request and is checked.

    :param adapter_type: The type of adapter layer to create.
    :type adapter_type: AdapterType
    :param name: Optional name for the layer instance.
    :type name: Optional[str]
    :param kwargs: Type-specific parameters. See :func:`get_adapter_info` for
        details. Any key ``adapter_type`` does not accept raises rather than
        being silently dropped.
    :return: A fully configured and instantiated adapter layer.
    :rtype: keras.layers.Layer
    :raises ValueError: If ``adapter_type`` is invalid, a required parameter is
        missing, a parameter value is out of range, a supplied keyword is not a
        parameter of ``adapter_type`` (the message then carries
        :data:`STRICT_DROPPED_KEY_MARKER`), or layer construction fails.
    """
    validate_adapter_config(adapter_type, **kwargs)

    info = ADAPTER_REGISTRY[adapter_type]
    params = info['optional_params'].copy()
    params.update(kwargs)
    params['name'] = name

    layer_class = info['class']
    try:
        return layer_class(**params)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Failed to create {adapter_type} layer "
            f"({layer_class.__name__}). Check for parameter incompatibility. "
            f"Custom args: {list(kwargs.keys())}. Original error: {e}"
        ) from e


def create_adapter_from_config(config: Dict[str, Any]) -> keras.layers.Layer:
    """Build an adapter layer from a single ``{'type': ..., ...}`` dict.

    Pops ``'type'`` off a copy of ``config`` and splats the rest into
    :func:`create_adapter_layer`, so every rule that function enforces applies
    here too -- including the strict keyword check. The input dict is not
    mutated.

    :param config: Configuration dict with a ``'type'`` key naming the adapter
        type, plus that type's parameters.
    :type config: Dict[str, Any]
    :return: The instantiated adapter layer.
    :rtype: keras.layers.Layer
    :raises ValueError: If ``config`` is not a dict, has no ``'type'`` key, or
        for any reason :func:`create_adapter_layer` raises.
    """
    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration must be a dictionary, got {type(config).__name__}. "
            f"Expected format: {{'type': 'adapter_type', ...}}"
        )

    if 'type' not in config:
        available_keys = list(config.keys()) if config else []
        raise ValueError(
            f"Configuration dictionary must include a 'type' key specifying the "
            f"adapter layer type. Available keys in config: {available_keys}. "
            f"Valid adapter types: {list(ADAPTER_REGISTRY.keys())}"
        )

    config_copy = config.copy()
    adapter_type = config_copy.pop('type')

    logger.debug(f"Creating adapter layer from config: {config}")
    return create_adapter_layer(adapter_type, **config_copy)


def list_adapter_types() -> List[str]:
    """List every registered adapter type key.

    Sorted alphabetically, so the order is stable across runs and does not follow
    the registry's insertion order.

    :return: The registry's keys, sorted.
    :rtype: List[str]
    """
    return sorted(list(ADAPTER_REGISTRY.keys()))


def get_adapter_requirements(adapter_type: str) -> Dict[str, Any]:
    """Return one adapter type's registry entry.

    :param adapter_type: The adapter type to look up.
    :type adapter_type: str
    :return: The registry entry for ``adapter_type``.
    :rtype: Dict[str, Any]
    :raises ValueError: If ``adapter_type`` is not in the registry.
    """
    if adapter_type not in ADAPTER_REGISTRY:
        available_types = list(ADAPTER_REGISTRY.keys())
        raise ValueError(
            f"Unknown adapter type '{adapter_type}'. Available types: {available_types}"
        )
    return copy.deepcopy(ADAPTER_REGISTRY[adapter_type])
