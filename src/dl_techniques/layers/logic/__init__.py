"""
Public API for `dl_techniques.layers.logic`.

Four learnable layer classes plus a string-keyed factory. Every class is
also importable from the module it is defined in. This file only re-exports
already decorated symbols; it registers nothing of its own, so importing
from here and importing from the module give you the same class object.
The four `keras.saving` keys are `dl_techniques.layers.logic.<module>><ClassName>`
-- resolved 2026-08-29 with `keras.saving.get_registered_name`, not copied from
any doc. Each module passes its own dotted path to `register_dl_technique`, so
the key names the defining module: `arithmetic_operators`, `logic_operators` and
`neural_circuit` (which owns two of the four). Until 2026-08-29 all four shared
the coarse hand-chosen string `"dl_techniques.layers"`; that was the last batch
of ad-hoc package strings in `src/` and it was normalized onto the module-path
rule the other 710 sites already followed. Each also
keeps the `Custom><ClassName>` alias bound to the identical object (verified
`a is b`).

**Architecture Overview:**

.. code-block:: text

    caller
      │
      ├─ create_logic_layer(type, **kwargs)
      │    type = 'arithmetic'      ─────────┐
      │           'logic'                    │
      │           'circuit_depth'            │
      │           'neural_circuit'           │
      │    the 4 keys live in LOGIC_REGISTRY │
      │                                      ▼
      └─ direct import ──► LearnableArithmeticOperator
                           LearnableLogicOperator
                           CircuitDepthLayer
                           LearnableNeuralCircuit

    create_logic_from_config(dict) reads 'type' out of a
    copy of the dict and calls create_logic_layer.
    Both operator classes take rank >= 1; the two circuit
    classes need rank >= 2. All four preserve the shape.

Counts, re-derived from the live module rather than copied from any doc:
`__all__` exports 10 names — the 4 classes drawn above, the 4 functions
`create_logic_layer`, `create_logic_from_config`, `get_logic_info` and
`validate_logic_config`, and 2 module-level objects, the `LogicLayerType`
alias and the `LOGIC_REGISTRY` dict. `LOGIC_REGISTRY` has 4 entries. Their
`optional_params` hold 18, 14, 17 and 18 defaults for `arithmetic`,
`logic`, `circuit_depth` and `neural_circuit`; every `required_params` is
empty, so every key builds with no arguments at all.

The detail lives in the modules. Read `factory.py` for the dispatch and
the registry tables, `logic_operators.py` for the 18 fuzzy gates,
`arithmetic_operators.py` for the 7 arithmetic operations, and
`neural_circuit.py` for the two shape-preserving stacking layers.
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
