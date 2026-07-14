"""
Factory Method for `dl_techniques.layers.logic`.

Provides a single, centralized entry point for creating learnable logic /
arithmetic / circuit layers via a string identifier (`layer_type`), mirroring
the registry-based design used in `layers/ffn/factory.py` and
`layers/norms/factory.py`.

Architectural Overview
----------------------
The factory operates on a registry-based design (``LOGIC_REGISTRY``). The
registry maps a string identifier (the ``layer_type``) to the concrete Keras
layer class and its associated metadata: a short description, required
parameters, optional parameters with defaults, and a recommended use-case.

When called, the factory performs the following steps:

1. **Validation**: consults the registry to validate the ``layer_type`` and
   that all required hyperparameters are present in ``**kwargs``.
2. **Class retrieval**: looks up the corresponding Keras layer class.
3. **Instantiation**: merges defaults with user kwargs, filters unknown keys,
   and constructs the layer.

This factory is intentionally narrow — it exposes only the public layer
contracts of the `logic/` package and does **not** wrap or alter their
behavior. Direct class imports remain fully supported.

When NOT to use this factory
----------------------------
If you need FFN-shaped learnable logic operating on a single feature vector
``(B, T, D) -> (B, T, D)`` use ``dl_techniques.layers.ffn.LogicFFN`` instead.
The classes in this package operate on tensors of arbitrary rank (>= 2)
without altering channel dimensionality.

References
----------
- Gamma et al. (1994). Design Patterns. Addison-Wesley.
- Liu, Simonyan, Yang (2018). "DARTS: Differentiable Architecture Search".
- Zadeh (1965). "Fuzzy sets". Information and Control.
"""

import copy
import keras
from typing import Any, Dict, Literal, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

from .arithmetic_operators import LearnableArithmeticOperator
from .logic_operators import LearnableLogicOperator
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

def get_logic_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all available logic layer types.

    :return: Dict mapping layer-type string to a copy of its registry entry
        (description, required_params, optional_params, use_case).
    :rtype: Dict[str, Dict[str, Any]]
    """
    return copy.deepcopy(LOGIC_REGISTRY)


def validate_logic_config(layer_type: str, **kwargs: Any) -> None:
    """
    Validate `create_logic_layer` arguments before instantiation.

    :param layer_type: One of the keys in ``LOGIC_REGISTRY``.
    :type layer_type: str
    :param kwargs: Layer-specific parameters.
    :raises ValueError: If ``layer_type`` is unknown, a required parameter is
        missing, or a numeric constraint is violated.
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

    # Common positive-int validations
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

    # Positive-float validations
    positive_floats = ["temperature_init", "scaling_init", "epsilon"]
    for name in positive_floats:
        if name in kwargs and kwargs[name] is not None:
            value = kwargs[name]
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    # G3 (plan_2026-05-13_e33114da): enum pre-validation with helpful error.
    # Catches typos before construction surface them through generic wrapped
    # "Failed to create logic layer" message.
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
    Construct a layer from `dl_techniques.layers.logic` by string type.

    :param layer_type: One of ``'arithmetic'``, ``'logic'``, ``'circuit_depth'``,
        ``'neural_circuit'``.
    :type layer_type: LogicLayerType
    :param name: Optional layer name.
    :type name: Optional[str]
    :param kwargs: Layer-specific parameters. See ``get_logic_info()`` or the
        individual class docstrings.
    :return: A fresh, unbuilt Keras layer instance.
    :rtype: keras.layers.Layer
    :raises ValueError: On unknown ``layer_type``, missing required parameters,
        or any downstream construction error.
    """
    try:
        validate_logic_config(layer_type, **kwargs)

        info = LOGIC_REGISTRY[layer_type]
        cls = info["class"]

        valid_params = set(info["required_params"]) | set(info["optional_params"].keys())

        params: Dict[str, Any] = {}
        params.update(info["optional_params"])
        params.update(kwargs)

        final_params = {k: v for k, v in params.items() if k in valid_params}

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
    Create a logic layer from a config dict with a ``'type'`` key.

    :param config: Dict with ``'type'`` plus layer-specific parameters.
    :type config: Dict[str, Any]
    :return: Configured layer instance.
    :rtype: keras.layers.Layer
    :raises ValueError: If ``config`` is not a dict or is missing ``'type'``.
    """
    if not isinstance(config, dict):
        raise ValueError(f"config must be a dict, got {type(config)}")
    if "type" not in config:
        raise ValueError("config must include a 'type' key")

    cfg = dict(config)
    layer_type = cfg.pop("type")
    return create_logic_layer(layer_type, **cfg)

# ---------------------------------------------------------------------
