"""
Build any activation layer in this package from a short string key.

``create_activation_layer('mish')`` returns a ``Mish`` layer.
``ACTIVATION_REGISTRY`` maps each key to the class, the parameters that key
requires, and the optional parameters with their factory defaults. The
registry is the source of truth for what exists; ``README.md`` keeps a
derived table of the same keys.

Three entry points:

- :func:`create_activation_layer` -- the builder. String key plus kwargs.
- :func:`resolve_activation_layer` -- the same, but falls back to
  ``keras.layers.Activation`` for plain Keras names such as ``'sigmoid'``
  that are not registry keys.
- :func:`create_activation_from_config` -- the same, from a dict carrying a
  ``'type'`` key.

Two keys share one class. ``'routing_probabilities'`` and
``'hierarchical_routing'`` both build ``RoutingProbabilitiesLayer``; the
first is the deterministic door, the second the trainable one.

Unknown parameters are rejected, never ignored. Every failure inside
:func:`create_activation_layer` surfaces as a ``ValueError`` naming the
activation type, the class, and the keys you passed.
"""

import keras
from typing import Dict, Any, Literal, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# DECISION plan-2026-08-27T103353-60745fe0/D-011
# Every class imported below registers BARE (no `package=`), giving a
# module-independent key. Do NOT add `package=` for guide-v2 s14 Pitfall 3:
# measured 697 bare registrations across src/, 0 colliding keys, and a changed
# key breaks load_model for every stored .keras holding these layers (34/34
# exact round-trip today). Repo-wide migration, not a local edit. See D-011.
#
# SUPERSEDED 2026-08-29 -- the repo-wide migration this anchor asked for HAPPENED.
# The ruling above is kept verbatim; only its cited
# measurement is re-taken, because a measured claim is superseded with a new
# measurement, not silently edited.
#   measured 2026-08-27: 697 bare registrations across src/, 0 colliding keys
#   measured 2026-08-29: 0 bare registrations across src/
#                        (`grep -rc "^@keras.saving.register_keras_serializable()" src/ --include=*.py`),
#                        744 `@register_dl_technique` sites in their place
# Every class imported below now registers PACKAGE-QUALIFIED, e.g.
# `dl_techniques.layers.activations.golu>GoLU`. The key-break the anchor warned
# about did not occur, because `register_dl_technique` also binds the legacy
# `Custom>ClassName` as an alias to the same object: all 21 registered classes
# reachable from this package resolve under BOTH keys to the IDENTICAL object
# (`a is b`), measured 2026-08-29. What remains forbidden is a bare `package=` on
# the stock decorator -- that moves the key WITHOUT minting the alias.

from .adaptive_softmax import AdaptiveTemperatureSoftmax
from .basis_function import BasisFunction
from .differentiable_step import DifferentiableStep
from .expanded_activations import (
    GELU, SiLU, xATLU, xGELU, xSiLU, EluPlusOne
)
from .golu import GoLU
from .hard_sigmoid import HardSigmoid
from .hard_swish import HardSwish
from .mish import Mish, SaturatedMish
from .monotonicity_layer import MonotonicityLayer
from .relu_k import ReLUK
from .routing_probabilities import RoutingProbabilitiesLayer
from .soft_value_range import SoftValueRange
from .sparsemax import Sparsemax
from .squash import SquashLayer
from .thresh_max import ThreshMax

# ---------------------------------------------------------------------

# Type definition for Activation types
ActivationType = Literal[
    'adaptive_softmax',
    'basis_function',
    'differentiable_step',
    'elu_plus_one',
    'gelu',
    'golu',
    'hard_sigmoid',
    'hard_swish',
    'hierarchical_routing',
    'mish',
    'monotonicity',
    'relu',
    'relu_k',
    'routing_probabilities',
    'saturated_mish',
    'silu',
    'soft_value_range',
    'sparsemax',
    'squash',
    'thresh_max',
    'xatlu',
    'xgelu',
    'xsilu'
]

# ---------------------------------------------------------------------

# Activation layer registry mapping types to classes and parameter info
ACTIVATION_REGISTRY: Dict[str, Dict[str, Any]] = {
    'adaptive_softmax': {
        'class': AdaptiveTemperatureSoftmax,
        'description': 'Softmax with dynamic temperature based on input entropy.',
        'required_params': [],
        'optional_params': {
            'min_temp': 0.1,
            'max_temp': 1.0,
            'entropy_threshold': 0.5,
            'eps': 1e-7,
            'polynomial_coeffs': None
        },
        'use_case': (
            'Maintains sharpness in softmax for large output spaces, '
            'improving retrieval tasks.'
        )
    },
    'basis_function': {
        'class': BasisFunction,
        'description': 'Implements b(x) = x * sigmoid(x), equivalent to Swish/SiLU.',
        'required_params': [],
        'optional_params': {},
        'use_case': 'Used in PowerMLP architectures for smooth, non-linear transformations.'
    },
    'differentiable_step': {
        'class': DifferentiableStep,
        'description': 'Learnable, differentiable approximation of a step function (tanh-based).',
        'required_params': [],
        'optional_params': {
            'shift_regularizer': None,
            'shift_constraint': None,
            'axis': -1,
            'slope_initializer': 'ones',
            'shift_initializer': 'zeros',
            # shift_regularizer / shift_constraint are listed above with the
            # value None so the strict kwarg check accepts them. None does NOT
            # override the class defaults: DifferentiableStep reads None as
            # "use mine". Measured -- a factory-built layer and a directly
            # constructed one both end up with L2(1e-3) and
            # ValueRangeConstraint(-1, +1).
        },
        'use_case': 'Learnable binary gates, soft thresholding, or feature selection.'
    },
    'elu_plus_one': {
        'class': EluPlusOne,
        'description': 'Enhanced ELU activation: ELU(x) + 1 + epsilon.',
        'required_params': [],
        'optional_params': {},
        'use_case': 'Ensures outputs are strictly positive, useful for rate parameters in distributions.'
    },
    'gelu': {
        'class': GELU,
        'description': 'Gaussian Error Linear Unit, a smooth, non-monotonic activation.',
        'required_params': [],
        'optional_params': {},
        'use_case': 'State-of-the-art activation for Transformer-based models.'
    },
    'golu': {
        'class': GoLU,
        'description': 'Gompertz Linear Unit, a self-gated activation using an asymmetrical Gompertz curve.',
        'required_params': [],
        'optional_params': {'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0},
        'use_case': (
            'Asymmetrical self-gated activation intended to create smoother '
            'loss landscapes and improve model generalization.'
        )
    },
    'hard_sigmoid': {
        'class': HardSigmoid,
        'description': 'Hard-sigmoid activation, a computationally efficient approximation of sigmoid.',
        'required_params': [],
        'optional_params': {},
        'use_case': 'Efficient gating in mobile networks and squeeze-and-excitation modules.'
    },
    'hard_swish': {
        'class': HardSwish,
        'description': 'Hard-swish activation, a computationally efficient variant of Swish/SiLU.',
        'required_params': [],
        'optional_params': {},
        'use_case': 'High-performance activation for mobile-optimized models like MobileNetV3.'
    },
    'hierarchical_routing': {
        # RoutingProbabilitiesLayer in trainable mode. This key replaced the
        # standalone HierarchicalRoutingLayer.
        'class': RoutingProbabilitiesLayer,
        'description': 'Trainable hierarchical probability tree for O(log N) classification.',
        'required_params': ['output_dim'],
        'optional_params': {
            'normalize': True,
            'input_normalization': None,
            'kernel_constraint': None,
            'bias_regularizer': None,
            'bias_constraint': None,
            'axis': -1,
            'epsilon': 1e-7,
            # This diverges from RoutingProbabilitiesLayer's own constructor
            # default ('deterministic') on purpose. This key is the trainable
            # door onto that class; the sibling key 'routing_probabilities'
            # below is the deterministic one. Do not "correct" it to match the
            # constructor. INTENTIONAL_OVERRIDES in
            # tests/test_layers/test_factory_registry_drift.py pins this value
            # and fails if it is edited.
            'mode': 'trainable',
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None
        },
        'use_case': 'Efficient classification for very large output spaces (e.g., language modeling).'
    },
    'mish': {
        'class': Mish,
        'description': 'A self-regularized, non-monotonic activation: x * tanh(softplus(x)).',
        'required_params': [],
        'optional_params': {},
        'use_case': (
            'Smooth activation that can outperform ReLU and Swish in deep '
            'vision and NLP models.'
        )
    },
    'monotonicity': {
        'class': MonotonicityLayer,
        'description': 'Enforces monotonic (non-decreasing) constraints on predictions.',
        'required_params': [],
        'optional_params': {
            'method': 'cumulative_softplus',
            'axis': -1,
            'min_spacing': None,
            'max_spacing': None,
            'value_range': None,
            'clip_inputs': None,
            'input_clip_range': (-20.0, 20.0),
            'epsilon': 1e-7
        },
        'use_case': 'Quantile regression, survival analysis, dose-response modeling.'
    },
    'relu': {
        'class': keras.layers.ReLU,
        'description': 'Rectified Linear Unit, the most common activation function.',
        'required_params': [],
        'optional_params': {
            'max_value': None,
            'negative_slope': 0.0,
            'threshold': 0.0
        },
        'use_case': (
            'Default activation for hidden layers in many types of neural '
            'networks due to its simplicity and effectiveness.'
        )
    },
    'relu_k': {
        'class': ReLUK,
        'description': 'Powered ReLU activation: max(0, x)^k.',
        'required_params': [],
        'optional_params': {'k': 3},
        'use_case': 'Creates more aggressive non-linearities than standard ReLU.'
    },
    'routing_probabilities': {
        'class': RoutingProbabilitiesLayer,
        'description': 'A non-trainable hierarchical routing layer using cosine basis patterns.',
        'required_params': [],
        'optional_params': {
            'normalize': True,
            'input_normalization': None,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'kernel_regularizer': None,
            'kernel_constraint': None,
            'bias_initializer': 'zeros',
            'bias_regularizer': None,
            'bias_constraint': None,
            'output_dim': None,
            'axis': -1,
            'epsilon': 1e-7,
            'mode': 'deterministic',
        },
        'use_case': (
            'Parameter-free alternative to softmax for multi-class '
            'classification, introducing a structured, hierarchical bias.'
        )
    },
    'saturated_mish': {
        'class': SaturatedMish,
        'description': 'Mish variant that smoothly saturates for large positive inputs.',
        'required_params': [],
        'optional_params': {'alpha': 3.0, 'beta': 0.5},
        'use_case': (
            'Prevents activation explosion in very deep networks by '
            'saturating the Mish function.'
        )
    },
    'silu': {
        'class': SiLU,
        'description': 'Sigmoid Linear Unit (SiLU/Swish), defined as x * sigmoid(x).',
        'required_params': [],
        'optional_params': {},
        'use_case': 'Self-gated activation that often outperforms ReLU in deep networks.'
    },
    'soft_value_range': {
        'class': SoftValueRange,
        'description': (
            'Smooth, monotone softplus map into [min_value, max_value] whose '
            'derivative is never structurally zero.'
        ),
        'required_params': ['min_value'],
        'optional_params': {
            'max_value': None,
            'sharpness': 50.0,
            'relative_sharpness': True
        },
        'use_case': (
            'Differentiable stand-in for ops.clip when a hard clip would kill '
            'the gradient outside the interval, e.g. bounded regression heads.'
        )
    },
    'sparsemax': {
        'class': Sparsemax,
        'description': 'Projects logits onto the probability simplex using Euclidean projection (L2).',
        'required_params': [],
        'optional_params': {'axis': -1},
        'use_case': (
            'Produces sparse probability distributions (with exact zeros), '
            'ideal for interpretable attention mechanisms.'
        )
    },
    'squash': {
        'class': SquashLayer,
        'description': 'Squashing non-linearity for Capsule Networks.',
        'required_params': [],
        'optional_params': {'axis': -1, 'epsilon': None},
        'use_case': (
            'Core non-linearity for Capsule Networks, normalizing vector '
            'outputs to represent probabilities.'
        )
    },
    'thresh_max': {
        'class': ThreshMax,
        'description': 'Sparse softmax variant using a differentiable step function.',
        'required_params': [],
        'optional_params': {
            'slope_regularizer': None,
            'slope_constraint': None,
            'axis': -1,
            'slope': 10.0,
            'epsilon': 1e-12,
            'trainable_slope': False,
            'slope_initializer': 'ones',
            # slope_regularizer / slope_constraint are listed above with the
            # value None so the strict kwarg check accepts them. None does NOT
            # override the class defaults: ThreshMax reads None as "use mine".
            # Measured -- a factory-built layer and a directly constructed one
            # both end up with L2_custom and ValueRangeConstraint(1.0, 50.0).
        },
        'use_case': 'Creates sparse, confident probability distributions as an alternative to softmax.'
    },
    'xatlu': {
        'class': xATLU,
        'description': 'Expanded ArcTan Linear Unit with a trainable alpha parameter.',
        'required_params': [],
        'optional_params': {
            'alpha_initializer': 'zeros',
            'alpha_regularizer': None,
            'alpha_constraint': None
        },
        'use_case': (
            'Expanded activation with an arctan gate; provides adaptable '
            'gating for specialized tasks.'
        )
    },
    'xgelu': {
        'class': xGELU,
        'description': 'Expanded Gaussian Error Linear Unit with a trainable alpha parameter.',
        'required_params': [],
        'optional_params': {
            'alpha_initializer': 'zeros',
            'alpha_regularizer': None,
            'alpha_constraint': None
        },
        'use_case': (
            'Extends GELU with a trainable parameter to adapt the gating '
            'range, enhancing flexibility.'
        )
    },
    'xsilu': {
        'class': xSiLU,
        'description': 'Expanded Sigmoid Linear Unit with a trainable alpha parameter.',
        'required_params': [],
        'optional_params': {
            'alpha_initializer': 'zeros',
            'alpha_regularizer': None,
            'alpha_constraint': None
        },
        'use_case': 'Extends SiLU/Swish with a trainable parameter to adapt the gating range.'
    }
}


# Public API functions
def get_activation_info() -> Dict[str, Dict[str, Any]]:
    """
    Return the registry contents, one entry per activation type.

    Each entry carries ``description``, ``required_params``,
    ``optional_params`` and ``use_case``.

    Treat the result as read-only. The copy is one level deep, so the nested
    ``optional_params`` dict is the same object the registry holds. Measured:
    after ``get_activation_info()['relu_k']['optional_params']['k'] = 99``,
    ``ACTIVATION_REGISTRY['relu_k']['optional_params']['k']`` is 99 and the
    factory builds ``ReLUK(k=99)``.

    :return: One entry per activation type, keyed by type name.
    :rtype: Dict[str, Dict[str, Any]]
    """
    return {
        act_type: info.copy() for act_type, info in ACTIVATION_REGISTRY.items()
    }


def validate_activation_config(activation_type: str, **kwargs: Any) -> None:
    """
    Check that ``activation_type`` exists and its parameters make sense.

    Two checks run for every type: the key is in ``ACTIVATION_REGISTRY``, and
    every name in that entry's ``required_params`` is present in ``kwargs``.
    After that, ten types get a hand-written value check
    (``adaptive_softmax``, ``differentiable_step``, ``golu``,
    ``hierarchical_routing``, ``monotonicity``, ``relu``, ``relu_k``,
    ``routing_probabilities``, ``saturated_mish``, ``thresh_max``), and the
    three expanded activations get their initializer / regularizer /
    constraint strings resolved to catch typos. Every other type passes
    through on the two generic checks alone.

    Returns ``None`` on success; it raises or it says nothing.

    :func:`create_activation_layer` calls this. Called directly, it raises
    ``TypeError`` for a wrong type; called through the factory, that
    ``TypeError`` comes back re-wrapped as a ``ValueError``.

    :param activation_type: Registry key to validate.
    :type activation_type: str
    :param kwargs: Parameters to validate against that key.
    :raises ValueError: If ``activation_type`` is not a registry key, a
        required parameter is missing, or a value is out of range.
    :raises TypeError: If a parameter has the wrong type (``axis`` for
        ``differentiable_step``, ``max_value`` for ``relu``).
    """
    if activation_type not in ACTIVATION_REGISTRY:
        available_types = sorted(list(ACTIVATION_REGISTRY.keys()))
        raise ValueError(
            f"Unknown activation type '{activation_type}'. "
            f"Available types: {available_types}"
        )

    # Validate required parameters exist
    required = ACTIVATION_REGISTRY[activation_type]['required_params']
    for param in required:
        if param not in kwargs:
            raise ValueError(
                f"Missing required parameter '{param}' for activation "
                f"type '{activation_type}'."
            )

    # Specific parameter logic validation
    if activation_type == 'adaptive_softmax':
        min_temp = kwargs.get('min_temp', 0.1)
        max_temp = kwargs.get('max_temp', 1.0)
        entropy_threshold = kwargs.get('entropy_threshold', 0.5)
        if min_temp <= 0.0:
            raise ValueError(f"min_temp must be positive, got {min_temp}")
        if max_temp <= 0.0:
            raise ValueError(f"max_temp must be positive, got {max_temp}")
        if min_temp > max_temp:
            raise ValueError(
                f"min_temp ({min_temp}) must be <= max_temp ({max_temp})"
            )
        if entropy_threshold < 0.0:
            raise ValueError(
                f"entropy_threshold must be non-negative, got {entropy_threshold}"
            )

    elif activation_type == 'differentiable_step':
        axis = kwargs.get('axis', -1)
        if axis is not None and not isinstance(axis, int):
            raise TypeError(f"axis must be int or None, got {type(axis)}")

    elif activation_type == 'golu':
        for param in ['alpha', 'beta', 'gamma']:
            val = kwargs.get(param, 1.0)
            if val <= 0.0:
                raise ValueError(f"{param} must be positive, got {val}")

    elif activation_type == 'hierarchical_routing':
        output_dim = kwargs.get('output_dim')
        if not isinstance(output_dim, int) or output_dim <= 1:
            raise ValueError(
                f"output_dim must be an integer > 1, got {output_dim}"
            )

    elif activation_type == 'monotonicity':
        method = kwargs.get('method', 'cumulative_softplus')
        valid_methods = [
            "cumulative_softplus", "exponential", "sigmoid",
            "normalized_softmax", "squared", "cumulative_exp"
        ]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid monotonicity method '{method}'. "
                f"Must be one of {valid_methods}"
            )
        if method in ["sigmoid", "normalized_softmax"]:
            if "value_range" not in kwargs or kwargs["value_range"] is None:
                raise ValueError(
                    f"value_range (min, max) is required for method '{method}'"
                )
            if len(kwargs["value_range"]) != 2:
                raise ValueError("value_range must be a tuple of (min, max)")

    elif activation_type == 'relu':
        max_val = kwargs.get('max_value')
        if max_val is not None and not isinstance(max_val, (int, float)):
            raise TypeError("max_value must be a number or None")

    elif activation_type == 'relu_k':
        k = kwargs.get('k', 3)
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")

    elif activation_type == 'routing_probabilities':
        output_dim = kwargs.get('output_dim')
        if output_dim is not None:
            if not isinstance(output_dim, int) or output_dim <= 1:
                raise ValueError(
                    f"output_dim must be integer > 1, got {output_dim}"
                )

    elif activation_type == 'saturated_mish':
        alpha = kwargs.get('alpha', 3.0)
        beta = kwargs.get('beta', 0.5)
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError("alpha and beta must be positive")

    elif activation_type == 'soft_value_range':
        min_value = kwargs.get('min_value')
        max_value = kwargs.get('max_value')
        sharpness = kwargs.get('sharpness', 50.0)
        if sharpness <= 0.0:
            raise ValueError(f"sharpness must be positive, got {sharpness}")
        if max_value is not None and max_value < min_value:
            raise ValueError(
                f"max_value ({max_value}) must be >= min_value ({min_value})"
            )

    elif activation_type == 'thresh_max':
        slope = kwargs.get('slope', 10.0)
        if slope <= 0:
            raise ValueError(f"slope must be positive, got {slope}")

    # Validate generic object params for expanded activations
    if activation_type in ['xatlu', 'xgelu', 'xsilu']:
        for param in ['alpha_initializer', 'alpha_regularizer', 'alpha_constraint']:
            if param in kwargs and isinstance(kwargs[param], str):
                try:
                    if 'initializer' in param:
                        keras.initializers.get(kwargs[param])
                    elif 'regularizer' in param:
                        keras.regularizers.get(kwargs[param])
                    elif 'constraint' in param:
                        keras.constraints.get(kwargs[param])
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid {param}: {e}")


STRICT_DROPPED_KEY_MARKER: str = "unsupported parameter(s)"
"""Substring every strict-drop rejection message carries.

DUPLICATION, deliberate and gated: the identical literal is defined in
`layers/attention/factory.py`, `layers/ffn/factory.py`,
`layers/embedding/factory.py` and `layers/sampling.py`. It is not centralised
because every candidate home is either a peer package (importing one factory from
another drags that package's whole layer tree into this module's import graph) or
a new shared module (this plan's abstraction budget is 0). The lockstep is NOT
hand-maintained: `tests/test_layers/test_activations/test_activation_factory.py::
TestStrictDroppedKeys::test_marker_is_identical_across_all_five_factories` fails
if any copy drifts.
"""

# Base `keras.layers.Layer` kwargs. Every registered activation class takes
# `**kwargs` and forwards them to `Layer.__init__`, so these five build and are
# NOT rejected by the strict check below. `layers/norms/factory.py` holds the
# same frozenset under the same name, copied for the same import-graph reason as
# the marker above and pinned identical by the same test class.
_KERAS_BASE_PARAMS = frozenset(
    {'name', 'dtype', 'trainable', 'activity_regularizer', 'autocast'}
)


def create_activation_layer(
    activation_type: ActivationType,
    name: Optional[str] = None,
    **kwargs: Any
) -> keras.layers.Layer:
    """
    Build one activation layer from its registry key.

    Registry defaults are applied first, then your ``kwargs`` override them.
    A name passed as ``name=`` wins over one passed inside ``kwargs``.

    **Architecture Overview:**

    .. code-block:: text

        activation_type, name, **kwargs
                       │
                       ▼
        ┌──────────────────────────────┐
        │ registry lookup (.get)       │
        └──────────────┬───────────────┘
                       ▼  entry, or None if unknown
        ┌──────────────────────────────┐
        │ reject undeclared kwargs     │──► ValueError
        └──────────────┬───────────────┘     (carries the marker)
                       ▼
        ┌──────────────────────────────┐
        │ validate_activation_config   │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │ defaults, then kwargs        │
        │ keep declared names only     │
        │ apply name=                  │
        └──────────────┬───────────────┘
                       ▼  final_params
        ┌──────────────────────────────┐
        │ act_class(**final_params)    │
        └──────────────┬───────────────┘
                       ▼
              keras.layers.Layer

    Only the undeclared-kwarg rejection escapes as itself. Everything from
    ``validate_activation_config`` down sits inside a ``try`` that catches
    ``TypeError`` and ``ValueError`` and re-raises one ``ValueError`` naming
    the type, the class and the keys you passed.

    :param activation_type: Registry key, e.g. ``'mish'`` or ``'sparsemax'``.
    :type activation_type: ActivationType
    :param name: Layer name. Overrides any ``name`` in ``kwargs``.
    :type name: Optional[str]
    :param kwargs: Parameters for that activation type, plus any of
        ``name``, ``dtype``, ``trainable``, ``activity_regularizer``,
        ``autocast``.
    :return: A built activation layer.
    :rtype: keras.layers.Layer
    :raises ValueError: Every failure path. An undeclared parameter gives a
        message carrying :data:`STRICT_DROPPED_KEY_MARKER`; an unknown type,
        a bad value or a constructor error gives a message beginning
        ``Failed to create <type> layer``.

    Note:
        ``TypeError`` never escapes this function, even though
        :func:`validate_activation_config` raises one. Measured:
        ``create_activation_layer('relu', max_value='big')`` and
        ``create_activation_layer('differentiable_step', axis='x')`` both
        come back as ``ValueError``, because the ``except`` clause below
        catches ``TypeError`` and re-raises it as ``ValueError``. Catch
        ``ValueError`` alone.
    """
    # DECISION plan-2026-08-18T140459-7991552f/D-017
    # Reject undeclared kwargs HERE, before the try, so the re-wrapper below cannot bury the marker.
    # Do NOT restore the old `or k in kwargs` clause (it made the registry side of the filter below
    # unreachable) or drop the _KERAS_BASE_PARAMS exemption (`trainable=`/`dtype=` build today).
    # Not a silent-drop fix -- Keras 3 already rejected the key. See decisions.md D-017.
    _info = ACTIVATION_REGISTRY.get(activation_type)
    if _info is not None:
        _valid_param_names = (
            set(_info['required_params'])
            | set(_info['optional_params'].keys())
            | _KERAS_BASE_PARAMS
        )
        dropped = sorted(set(kwargs) - _valid_param_names)
        if dropped:
            raise ValueError(
                f"create_activation_layer('{activation_type}'): "
                f"{len(dropped)} {STRICT_DROPPED_KEY_MARKER} {dropped}. "
                f"'{activation_type}' ({_info['class'].__name__}) accepts only "
                f"{sorted(_valid_param_names)}. Either you mistyped one of "
                f"those names, or you chose the wrong activation_type for the "
                f"parameters you are passing."
            )

    try:
        # Validate configuration first
        validate_activation_config(activation_type, **kwargs)

        # Get activation info and class from registry
        act_info = ACTIVATION_REGISTRY[activation_type]
        act_class = act_info['class']

        # Prepare parameters: start with defaults, override with user kwargs
        params = {}
        params.update(act_info['optional_params'])

        # Add required parameters only if they are in kwargs
        for req in act_info['required_params']:
            if req in kwargs:
                params[req] = kwargs[req]

        # Update with remaining kwargs
        params.update(kwargs)

        # Names the class constructor accepts: the registry's own, plus the
        # base Layer kwargs. The strict check at the top of this function has
        # already rejected everything outside this set, so by here the filter
        # below drops nothing. It stays as a second line of defence.
        valid_param_names = (
            set(act_info['required_params']) |
            set(act_info['optional_params'].keys())
        )

        valid_param_names |= _KERAS_BASE_PARAMS

        final_params = {
            k: v for k, v in params.items()
            if k in valid_param_names
        }

        if name is not None:
            final_params['name'] = name

        # Log creation
        logger.info(f"Creating {activation_type} layer.")
        logger.debug(f"Params: {final_params}")

        # Instantiate
        activation_layer = act_class(**final_params)

        return activation_layer

    except (TypeError, ValueError) as e:
        # Provide enhanced error reporting
        act_info = ACTIVATION_REGISTRY.get(activation_type)
        class_name = act_info.get('class', type(None)).__name__ if act_info else "Unknown"

        error_msg = (
            f"Failed to create {activation_type} layer ({class_name}). "
            f"Provided keys: {list(kwargs.keys())}. "
            f"Error: {e}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg) from e


# ---------------------------------------------------------------------

def resolve_activation_layer(
    activation_type: str,
    name: Optional[str] = None,
    **kwargs: Any
) -> keras.layers.Layer:
    """
    Build an activation layer from either a registry key or a Keras name.

    A registry key goes to :func:`create_activation_layer`. Anything else is
    handed to ``keras.layers.Activation``, which covers the standard Keras
    strings such as ``'sigmoid'``, ``'tanh'``, ``'linear'`` and ``'softmax'``.

    Use this when a layer's ``activation`` argument should accept both, so a
    caller can write ``'mish'`` or ``'sigmoid'`` without knowing which side
    of the line it falls on.

    ``kwargs`` are dropped on the Keras fallback path. Measured:
    ``resolve_activation_layer('sigmoid')`` returns an ``Activation`` layer;
    ``resolve_activation_layer('mish')`` returns a ``Mish`` layer.

    :param activation_type: Either a key in ``ACTIVATION_REGISTRY`` or any
        string ``keras.activations.get`` accepts.
    :type activation_type: str
    :param name: Layer name, applied on both paths.
    :type name: Optional[str]
    :param kwargs: Forwarded to :func:`create_activation_layer` for a registry
        key. Ignored for a plain Keras activation.
    :return: A built activation layer.
    :rtype: keras.layers.Layer
    :raises ValueError: If ``activation_type`` is neither a registry key nor a
        name Keras recognises. The message then comes from
        ``keras.activations.get``, not from this factory.
    """
    if activation_type in ACTIVATION_REGISTRY:
        return create_activation_layer(activation_type, name=name, **kwargs)
    return keras.layers.Activation(activation_type, name=name)


# ---------------------------------------------------------------------

def create_activation_from_config(config: Dict[str, Any]) -> keras.layers.Layer:
    """
    Build an activation layer from a config dict.

    ``config['type']`` is the registry key. Every other entry is passed to
    :func:`create_activation_layer` as a keyword argument, so
    ``{'type': 'relu_k', 'k': 2}`` builds ``ReLUK(k=2)``. The dict is copied
    before ``'type'`` is popped, so the caller's dict is left alone.

    There is no ``name`` shortcut here. Put ``name`` in the dict like any
    other parameter.

    :param config: Dict with a ``'type'`` key plus that type's parameters.
    :type config: Dict[str, Any]
    :return: A built activation layer.
    :rtype: keras.layers.Layer
    :raises ValueError: If ``config`` is not a dict, has no ``'type'`` key, or
        :func:`create_activation_layer` rejects the rest.
    """
    if not isinstance(config, dict):
        raise ValueError(f"config must be a dictionary, got {type(config)}")

    if 'type' not in config:
        raise ValueError("Configuration must include a 'type' key")

    config_copy = config.copy()
    activation_type = config_copy.pop('type')

    return create_activation_layer(activation_type, **config_copy)

# ---------------------------------------------------------------------
