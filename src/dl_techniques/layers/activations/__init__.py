"""
Public surface of ``dl_techniques.layers.activations``.

Re-exports every activation layer in the package, the three serializable
activation functions, and the six factory helpers. Import from here rather
than from the individual modules::

    from dl_techniques.layers.activations import GoLU, create_activation_layer

Three groups, matching the sections of ``__all__`` below:

- **Factory utilities** (``factory.py``). Build a layer from a registry key
  or from a config dict. ``ActivationType`` is the ``Literal`` alias listing
  every valid key.
- **Activation functions** (``gelu_tanh.py``, ``soft_value_range.py``).
  Plain callables, registered with Keras so they survive a ``.keras``
  round-trip. Everything else exported here is a Layer.
- **Layer classes**. One entry per activation module.

Watch the name ``resolve_activation``. Two different functions in this
package carry it. The one exported here is ``gelu_tanh.resolve_activation``,
which extends ``keras.activations.get`` with the tanh-GELU spellings.
``common.resolve_activation``, which turns an activation spec into a
callable and rejects Layer instances, is **not** exported and must be
imported as ``from .common import resolve_activation``.

``__all__`` is a contract: 31 names. Every name in it must also appear in an
import above, or ``from ... import *`` raises ``AttributeError`` on that
name. Add the import and the ``__all__`` entry in the same change.
"""

from .factory import (
    create_activation_from_config,
    create_activation_layer,
    resolve_activation_layer,
    get_activation_info,
    ActivationType,
    validate_activation_config
)

# Explicitly export layer classes for direct import
from .adaptive_softmax import AdaptiveTemperatureSoftmax
from .basis_function import BasisFunction
from .differentiable_step import DifferentiableStep
from .expanded_activations import (
    GELU, SiLU, xATLU, xGELU, xSiLU, EluPlusOne
)
from .gelu_tanh import gelu_tanh, resolve_activation
from .golu import GoLU
from .hard_sigmoid import HardSigmoid
from .hard_swish import HardSwish
from .mish import Mish, SaturatedMish
from .monotonicity_layer import MonotonicityLayer
from .relu_k import ReLUK
from .routing_probabilities import RoutingProbabilitiesLayer
from .soft_value_range import soft_value_range, SoftValueRange
from .sparsemax import Sparsemax
from .squash import SquashLayer
from .thresh_max import ThreshMax
from .probability_output import ProbabilityOutput

__all__ = [
    # Factory Utilities
    "ActivationType",
    "get_activation_info",
    "create_activation_layer",
    "resolve_activation_layer",
    "validate_activation_config",
    "create_activation_from_config",

    # Activation Functions (serializable, registered)
    "gelu_tanh",
    "resolve_activation",
    "soft_value_range",

    # Layer Classes
    "AdaptiveTemperatureSoftmax",
    "BasisFunction",
    "DifferentiableStep",
    "GELU", "SiLU", "xATLU", "xGELU", "xSiLU", "EluPlusOne",
    "GoLU",
    "HardSigmoid",
    "HardSwish",
    "Mish", "SaturatedMish",
    "MonotonicityLayer",
    "ReLUK",
    "RoutingProbabilitiesLayer",
    "SoftValueRange",
    "Sparsemax",
    "SquashLayer",
    "ThreshMax",
    "ProbabilityOutput",
]