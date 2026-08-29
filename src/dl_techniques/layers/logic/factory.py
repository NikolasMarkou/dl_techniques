"""
Build any layer in `dl_techniques.layers.logic` from a string key.

``create_logic_layer("logic", ...)`` returns a fresh, unbuilt layer. The
four keys and the class each one builds live in ``LOGIC_REGISTRY``, which
also carries the defaults the factory fills in and the enum values it
checks. Direct class imports keep working; this is for callers that read
their layer type out of a config file.

**Dispatch flow:**

.. code-block:: text

    config dict ──► create_logic_from_config()
                    copies the dict, then pops 'type' from the copy
                                 │
                                 ▼
      create_logic_layer(layer_type, name=None, **kwargs)
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────┐
    │ try:                                                     │
    │   validate_logic_config(layer_type, **kwargs)            │
    │     ├─ layer_type not in LOGIC_REGISTRY ───────► raise   │
    │     ├─ a required_params name is missing ──────► raise   │
    │     ├─ an undeclared keyword is present ───────► raise   │
    │     ├─ a positive-int name is <= 0, or a bool ─► raise   │
    │     ├─ a positive-float name is <= 0 ──────────► raise   │
    │     └─ an enum_params value is not allowed ────► raise   │
    │     │                                                    │
    │     ▼                                                    │
    │   cls    = LOGIC_REGISTRY[layer_type]['class']           │
    │   final  = optional_params defaults, then kwargs         │
    │            validate already rejected every key the       │
    │            entry does not declare, so nothing here       │
    │            is filtered out                               │
    │     │                                                    │
    │     ▼                                                    │
    │   final['name'] = name   (only when name is given)       │
    │   logger.debug, one line per parameter                   │
    │     │                                                    │
    │     ▼                                                    │
    │   layer = cls(**final)                                   │
    ├──────────────────────────────────────────────────────────┤
    │ except (TypeError, ValueError) as e:                     │
    │   every raise above lands here and comes back out as     │
    │   one ValueError, chained with `from e`. Two messages:   │
    │     layer_type in LOGIC_REGISTRY -> names the class,     │
    │       its required params and the params you passed      │
    │     layer_type unknown ──────────► names only the type   │
    └──────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                        keras.layers.Layer

A keyword the target type does not declare is a ``ValueError``, not a
default. Measured: ``create_logic_layer("logic", bogus_key=1)`` raises,
naming the key, the count and the accepted set. The check lives in
``validate_logic_config``, so calling that directly rejects the same key.
Every message carries ``STRICT_UNSUPPORTED_KEY_MARKER``, which is what a
test should match on.

**Migration.** Until 2026-08-29 this factory filtered such a keyword out
and said nothing, so a call that quietly lost a setting now fails loudly.
The fix is the same either way: read the accepted set off the error, or
off ``get_logic_info()``, and correct the spelling. If a wrapper is
handing over its own generic defaults rather than an explicit request,
filter them against ``get_logic_info()[type]["optional_params"]`` before
the call.

**The four registry keys:**

.. code-block:: text

    type key        class                        optional  enums
    --------------  ---------------------------  --------  -----
    arithmetic      LearnableArithmeticOperator        18      3
    logic           LearnableLogicOperator             14      1
    circuit_depth   CircuitDepthLayer                  17      3
    neural_circuit  LearnableNeuralCircuit             18      4

    Every entry has an empty required_params, so every key
    constructs with no arguments at all. The optional column
    is len(optional_params) and every one of those has a
    default in the registry.

**What each key is for:**

.. code-block:: text

    type key        use case
    --------------  ----------------------------------------
    arithmetic      Learnable elementwise arithmetic between
                    two same-shape tensors (or unary, with
                    caveats — see README).
    logic           Soft logical combination of two same-
                    shape tensors interpreted as fuzzy truth
                    values.
    circuit_depth   Drop-in mid-network expert ensemble that
                    preserves tensor shape (rank >= 2).
    neural_circuit  Deep compositional reasoning block —
                    shape-preserving, rank >= 2.

**The values validate_logic_config checks by name:**

.. code-block:: text

    type key        parameter                allowed values
    --------------  -----------------------  ----------------------
    arithmetic      safe_divide_mode         hard_clamp, smooth
    arithmetic      selection_mode           global, per_channel
    arithmetic      exponent_clip_mode       hard, smooth
    logic           selection_mode           global, per_channel
    circuit_depth   circuit_routing          classic, output_only
    circuit_depth   selection_mode           global, per_channel
    circuit_depth   channel_mix              None, dense
    neural_circuit  circuit_routing          classic, output_only
    neural_circuit  apply_sigmoid_per_depth  all, first_only, none
    neural_circuit  selection_mode           global, per_channel
    neural_circuit  channel_mix              None, dense

    Anything else is checked by the class itself, and the
    error comes back wrapped.

Three of the four keys reach ``LearnableArithmeticOperator``:
``arithmetic`` builds one directly, ``circuit_depth`` builds
``num_arithmetic_ops`` of them inside each stage, and ``neural_circuit``
stacks those stages. That class was dead on its forward pass between
2026-08-25 and the repair in this package's history, so those three keys
raised ``NameError`` on the first call while ``logic`` kept working.

When not to use this factory: if you want an FFN-shaped learnable logic
block over a single feature vector, ``(B, T, D) -> (B, T, D)``, use
``dl_techniques.layers.ffn.LogicFFN``. The classes here take a tensor of
any rank >= 2 and give back the same shape.

References:
    - Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994). "Design
      Patterns". The registry-backed factory this file follows.

    - Liu, H., Simonyan, K., & Yang, Y. (2018). "DARTS: Differentiable
      Architecture Search". The soft selection the arithmetic and logic
      layers use.

    - Zadeh, L. A. (1965). "Fuzzy sets". Information and Control. The
      source of the soft gate forms.
"""

import copy
import keras
from typing import Any, Dict, Literal, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

from .logic_operators import LearnableLogicOperator
from .arithmetic_operators import LearnableArithmeticOperator
from .neural_circuit import CircuitDepthLayer, LearnableNeuralCircuit

# ---------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------

LogicLayerType = Literal[
    "arithmetic",
    "logic",
    "circuit_depth",
    "neural_circuit",
]

# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------

LOGIC_REGISTRY: Dict[str, Dict[str, Any]] = {
    "arithmetic": {
        "class": LearnableArithmeticOperator,
        "description": (
            "Differentiable, learnable arithmetic operator (DARTS-style soft "
            "selection over add/multiply/subtract/divide/power/max/min)."
        ),
        "required_params": [],
        "optional_params": {
            "operation_types": None,
            "use_temperature": True,
            "temperature_init": 1.0,
            "use_scaling": True,
            "scaling_init": 1.0,
            "operation_initializer": "zeros",
            "temperature_initializer": None,
            "scaling_initializer": None,
            "epsilon": 1e-7,
            "power_clip_range": (1e-7, 10.0),
            "exponent_clip_range": (-2.0, 2.0),
            "softplus_temperature": True,
            "safe_divide_mode": "hard_clamp",
            "gumbel_softmax": False,
            "gumbel_hard": False,
            "entropy_coefficient": 0.0,
            "selection_mode": "global",
            "exponent_clip_mode": "hard",
        },
        "enum_params": {
            "safe_divide_mode": {"hard_clamp", "smooth"},
            "selection_mode": {"global", "per_channel"},
            "exponent_clip_mode": {"hard", "smooth"},
        },
        "use_case": (
            "Learnable elementwise arithmetic between two same-shape tensors "
            "(or unary, with caveats — see README)."
        ),
    },
    "logic": {
        "class": LearnableLogicOperator,
        "description": (
            "Differentiable, learnable fuzzy logic operator over "
            "and/or/xor/not/nand/nor with sigmoid input normalization."
        ),
        "required_params": [],
        "optional_params": {
            "force_clip_when_no_sigmoid": False,
            "operation_types": None,
            "use_temperature": True,
            "temperature_init": 1.0,
            "operation_initializer": "zeros",
            "temperature_initializer": None,
            "apply_sigmoid": True,
            "softplus_temperature": True,
            "gumbel_softmax": False,
            "gumbel_hard": False,
            "entropy_coefficient": 0.0,
            "allow_unary_degenerate": False,
            "selection_mode": "global",
            "yager_p": 2.0,
        },
        "enum_params": {
            "selection_mode": {"global", "per_channel"},
        },
        "use_case": (
            "Soft logical combination of two same-shape tensors interpreted "
            "as fuzzy truth values."
        ),
    },
    "circuit_depth": {
        "class": CircuitDepthLayer,
        "description": (
            "Single MoE-style depth layer combining parallel logic and "
            "arithmetic operators with learnable input routing and output "
            "fusion."
        ),
        "required_params": [],
        "optional_params": {
            "force_logic_input_clip": False,
            "load_balance_coefficient": None,
            "num_logic_ops": 2,
            "num_arithmetic_ops": 2,
            "use_residual": True,
            "logic_op_types": None,
            "arithmetic_op_types": None,
            "routing_initializer": "zeros",
            "combination_initializer": "zeros",
            "circuit_routing": "output_only",
            "apply_sigmoid": True,
            "gate_entropy_coefficient": 0.0,
            "channel_mix": None,
            "selection_mode": "global",
            "diversity_coefficient": 0.0,
            "inner_logic_kwargs": None,
            "inner_arithmetic_kwargs": None,
        },
        "enum_params": {
            "circuit_routing": {"output_only", "classic"},
            "selection_mode": {"global", "per_channel"},
            "channel_mix": {None, "dense"},
        },
        "use_case": (
            "Drop-in mid-network expert ensemble that preserves tensor shape "
            "(rank >= 2)."
        ),
    },
    "neural_circuit": {
        "class": LearnableNeuralCircuit,
        "description": (
            "Stacked CircuitDepthLayer pipeline with optional layer "
            "normalization between depth levels."
        ),
        "required_params": [],
        "optional_params": {
            "load_balance_coefficient": None,
            "circuit_depth": 3,
            "num_logic_ops_per_depth": 2,
            "num_arithmetic_ops_per_depth": 2,
            "use_residual": True,
            "use_layer_norm": False,
            "logic_op_types": None,
            "arithmetic_op_types": None,
            "routing_initializer": "zeros",
            "combination_initializer": "zeros",
            "circuit_routing": "output_only",
            "apply_sigmoid_per_depth": "first_only",
            "gate_entropy_coefficient": 0.0,
            "channel_mix": None,
            "selection_mode": "global",
            "diversity_coefficient": 0.0,
            "inner_logic_kwargs": None,
            "inner_arithmetic_kwargs": None,
        },
        "enum_params": {
            "circuit_routing": {"output_only", "classic"},
            "apply_sigmoid_per_depth": {"first_only", "all", "none"},
            "selection_mode": {"global", "per_channel"},
            "channel_mix": {None, "dense"},
        },
        "use_case": (
            "Deep compositional reasoning block — shape-preserving, "
            "rank >= 2."
        ),
    },
}


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

#: The stable substring every undeclared-key ``ValueError`` raised by
#: :func:`validate_logic_config` carries. Guards match on this constant
#: instead of retyping the phrase, so rewording the message cannot blind
#: them. `layers/ffn/factory.py` defines its own copy for its own message;
#: the two are independent on purpose, so a reword there cannot change what
#: a guard here matches.
STRICT_UNSUPPORTED_KEY_MARKER: str = "unsupported parameter(s)"


def get_logic_info() -> Dict[str, Dict[str, Any]]:
    """
    Return a deep copy of the whole registry.

    Every key of every entry comes back, including ``class``, so the
    caller can read the target class as well as the ``description``,
    ``required_params``, ``optional_params``, ``enum_params`` and
    ``use_case`` fields. It is a copy, so editing the result does not
    change what the factory builds.

    :return: One entry per layer type, keyed by the type string.
    :rtype: Dict[str, Dict[str, Any]]
    """
    return copy.deepcopy(LOGIC_REGISTRY)


def validate_logic_config(layer_type: str, **kwargs: Any) -> None:
    """
    Check the arguments a ``create_logic_layer`` call would use.

    This catches an unknown type, a missing required parameter, a keyword
    the entry does not declare, a non-positive count or scale, and a value
    outside the allowed set of an enum parameter. The undeclared-keyword
    message carries ``STRICT_UNSUPPORTED_KEY_MARKER``. Range and enum
    checks are by NAME, so a name this type does not declare never reaches
    them -- it is rejected first. The module docstring lists exactly what
    is checked.

    :param layer_type: One of the keys in ``LOGIC_REGISTRY``.
    :type layer_type: str
    :param kwargs: The parameters the layer would be built with.
    :type kwargs: Any
    :return: Nothing. It either passes or raises.
    :rtype: None
    :raises ValueError: If the type is unknown, a required parameter is
        missing, a keyword is not declared by the entry, a count is not a
        positive int, a scale is not positive, or an enum value is not
        allowed.
    """
    if layer_type not in LOGIC_REGISTRY:
        available = sorted(LOGIC_REGISTRY.keys())
        raise ValueError(
            f"Unknown logic layer type '{layer_type}'. "
            f"Available types: {available}"
        )

    info = LOGIC_REGISTRY[layer_type]
    required = info["required_params"]
    missing = [p for p in required if p not in kwargs]
    if missing:
        raise ValueError(
            f"Required parameters missing for {layer_type}: {missing}. "
            f"Required: {required}"
        )

    # DECISION plan-2026-08-29T112804-aff039c4/D-002 -- raise, never
    # filter-and-drop. Subtract from `kwargs`, not from the merged
    # parameter dict the factory builds: that dict already carries the
    # registry defaults, so it can never expose a caller's typo. The
    # rejection lives here rather than in the layer constructor, which
    # takes **kwargs and would name the Keras base class instead of this
    # factory. See decisions.md D-002.
    declared = set(required) | set(info["optional_params"])
    unsupported = sorted(set(kwargs) - declared)
    if unsupported:
        raise ValueError(
            f"create_logic_layer('{layer_type}'): {len(unsupported)} "
            f"{STRICT_UNSUPPORTED_KEY_MARKER} {unsupported}. "
            f"'{layer_type}' ({info['class'].__name__}) accepts only "
            f"{sorted(declared)}. Nothing is dropped here: check the "
            f"spelling against get_logic_info()."
        )

    # These names must be positive whole numbers wherever they appear.
    positive_ints = [
        "num_logic_ops",
        "num_arithmetic_ops",
        "circuit_depth",
        "num_logic_ops_per_depth",
        "num_arithmetic_ops_per_depth",
    ]
    for name in positive_ints:
        if name in kwargs and kwargs[name] is not None:
            value = kwargs[name]
            # bool is a subclass of int — reject it explicitly.
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"{name} must be a positive integer, got {value!r}"
                )

    # These names must be positive numbers wherever they appear.
    positive_floats = ["temperature_init", "scaling_init", "epsilon"]
    for name in positive_floats:
        if name in kwargs and kwargs[name] is not None:
            value = kwargs[name]
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    # Checking the enums here names the bad value. Left to the class, the
    # same mistake comes back wrapped in the generic "Failed to create
    # logic layer" message.
    enum_params: Dict[str, set] = info.get("enum_params", {})
    for name, allowed in enum_params.items():
        if name in kwargs and kwargs[name] not in allowed:
            raise ValueError(
                f"{name}={kwargs[name]!r} is not a valid value. "
                f"Allowed: {sorted(a if a is not None else 'None' for a in allowed)}"
            )


def create_logic_layer(
        layer_type: LogicLayerType,
        name: Optional[str] = None,
        **kwargs: Any,
) -> keras.layers.Layer:
    """
    Build one layer from `dl_techniques.layers.logic` by type string.

    Anything you do not pass takes its default from the registry entry.
    Anything you pass that the type does not declare is a ``ValueError``
    naming the key and the accepted set; nothing is dropped. Every error
    raised on the way, including one from the class constructor, comes
    back as a single ``ValueError`` naming the type and what you passed.

    :param layer_type: ``'arithmetic'``, ``'logic'``, ``'circuit_depth'``
        or ``'neural_circuit'``.
    :type layer_type: LogicLayerType
    :param name: Name for the layer. ``None`` lets Keras pick one.
    :type name: Optional[str]
    :param kwargs: Parameters for that layer type. See
        ``get_logic_info()`` or the class docstring.
    :type kwargs: Any
    :return: A fresh, unbuilt layer.
    :rtype: keras.layers.Layer
    :raises ValueError: On an unknown type, a keyword the type does not
        declare, a failed validation, or any error the constructor itself
        raises.

    Example:
        .. code-block:: python

            import keras
            from dl_techniques.layers.logic import create_logic_layer

            layer = create_logic_layer(
                'neural_circuit', circuit_depth=2, name='circuit'
            )
            y = layer(keras.random.normal((4, 16)))
            y.shape  # (4, 16)
    """
    try:
        validate_logic_config(layer_type, **kwargs)

        info = LOGIC_REGISTRY[layer_type]
        cls = info["class"]

        # validate_logic_config has already rejected every key outside the
        # entry's schema, so this merge cannot carry an undeclared name.
        final_params: Dict[str, Any] = dict(info["optional_params"])
        final_params.update(kwargs)

        if name is not None:
            final_params["name"] = name

        logger.debug(f"Creating logic layer '{layer_type}' ({cls.__name__}):")
        for k in sorted(final_params.keys()):
            logger.debug(f"  {k}: {final_params[k]!r}")

        layer = cls(**final_params)
        logger.debug(f"Created {layer_type} layer: {layer.name}")
        return layer

    except (TypeError, ValueError) as e:
        info = LOGIC_REGISTRY.get(layer_type)
        if info is not None:
            class_name = info["class"].__name__
            msg = (
                f"Failed to create logic layer '{layer_type}' ({class_name}). "
                f"Required: {info['required_params']}. "
                f"Provided: {list(kwargs.keys())}. "
                f"Original error: {e}"
            )
        else:
            msg = (
                f"Failed to create logic layer — unknown type '{layer_type}'. "
                f"Original error: {e}"
            )
        logger.error(msg)
        raise ValueError(msg) from e


def create_logic_from_config(config: Dict[str, Any]) -> keras.layers.Layer:
    """
    Build one layer from a config dict carrying a ``'type'`` key.

    The dict is copied before ``'type'`` is removed, so the caller's dict
    is left alone. Every remaining key is passed to
    :func:`create_logic_layer`.

    :param config: ``'type'`` plus the parameters for that type.
    :type config: Dict[str, Any]
    :return: A fresh, unbuilt layer.
    :rtype: keras.layers.Layer
    :raises ValueError: If ``config`` is not a dict, has no ``'type'``, or
        the underlying ``create_logic_layer`` call fails.

    Example:
        .. code-block:: python

            from dl_techniques.layers.logic import (
                create_logic_from_config,
            )

            layer = create_logic_from_config({
                'type': 'circuit_depth',
                'num_logic_ops': 3,
            })
    """
    if not isinstance(config, dict):
        raise ValueError(f"config must be a dict, got {type(config)}")
    if "type" not in config:
        raise ValueError("config must include a 'type' key")

    cfg = dict(config)
    layer_type = cfg.pop("type")
    return create_logic_layer(layer_type, **cfg)

# ---------------------------------------------------------------------
