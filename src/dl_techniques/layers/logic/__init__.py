"""
Public API for `dl_techniques.layers.logic`.

Exposes four differentiable, learnable layer primitives plus a string-keyed
factory function. All classes remain importable from their original module
paths — this `__init__` only re-exports already-decorated symbols (no extra
registration with `keras.saving`).
"""

from .arithmetic_operators import LearnableArithmeticOperator
from .logic_operators import LearnableLogicOperator
from .neural_circuit import CircuitDepthLayer, LearnableNeuralCircuit
from .factory import (
    LOGIC_REGISTRY,
    LogicLayerType,
    create_logic_from_config,
    create_logic_layer,
    get_logic_info,
    validate_logic_config,
)

__all__ = [
    "LearnableArithmeticOperator",
    "LearnableLogicOperator",
    "CircuitDepthLayer",
    "LearnableNeuralCircuit",
    "LogicLayerType",
    "LOGIC_REGISTRY",
    "create_logic_layer",
    "create_logic_from_config",
    "get_logic_info",
    "validate_logic_config",
]
