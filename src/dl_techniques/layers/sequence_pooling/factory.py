"""
Sequence Pooling Layer Factory.

A registry-driven factory for creating sequence-pooling layers with a unified
interface, type safety, parameter validation, and detailed metadata. The factory
mirrors the canonical ``attention/`` sub-package template and exposes the four
pooling layers in this package through a single, consistent construction surface.

The factory supports four pooling mechanisms:
    - ``attention``: learnable, content-aware attention-weighted pooling.
    - ``mode``: pooling to the dominant mode of the tokens by mean-shift
      iterations, robust to a minority of far-away tokens.
    - ``weighted``: learnable, content-agnostic per-position weighted pooling.
    - ``sequence``: a unified facade dispatching 18 strategies and 4 aggregation
      methods (internally composing the attention and weighted poolers).

Key Features:
    - Type-safe pooling-layer creation with parameter validation.
    - Unified interface across all pooling mechanisms.
    - Detailed parameter documentation and error handling.
    - Support for both dictionary-based and direct configuration.
    - Integration with the dl_techniques logging system.
    - Complete compatibility with Keras 3 serialization.
"""

import copy

import keras
from typing import Any, Dict, List, Literal, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.validation import positive_float, positive_int
from .weighted_pooling import WeightedPooling
from .sequence_pooling import SequencePooling, check_strategy_config
from .attention_pooling import AttentionPooling
from .mode_pooling import ModePooling

# ---------------------------------------------------------------------
# Type Definitions
# ---------------------------------------------------------------------

SequencePoolingType = Literal['attention', 'mode', 'weighted', 'sequence']
"""
Type alias for supported sequence-pooling mechanisms.

This literal type provides IDE autocompletion and type checking for valid
pooling-layer types supported by the factory.
"""

# ---------------------------------------------------------------------
# Sequence Pooling Layer Registry
# ---------------------------------------------------------------------

SEQUENCE_POOLING_REGISTRY: Dict[str, Dict[str, Any]] = {
    'attention': {
        'class': AttentionPooling,
        'description': (
            'Learnable, content-aware attention pooling. Each token is passed '
            'through a tanh projection and scored against a learnable context '
            'vector to produce per-token importance weights via softmax; the '
            'output is the weighted sum of the input tokens, optionally across '
            'multiple averaged attention heads.'
        ),
        'required_params': [],
        'optional_params': {
            'hidden_dim': 256,
            'num_heads': 1,
            'dropout_rate': 0.0,
            'use_bias': True,
            'temperature': 1.0,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
        },
        'use_case': (
            'Collapsing a sequence into a single summary vector when relevance '
            'is input-dependent: sentence/document embeddings, sequence '
            'classification heads, and any task where the model should '
            'dynamically focus on the most informative tokens.'
        ),
    },

    'mode': {
        'class': ModePooling,
        'description': (
            'Mode pooling, with no weights unless learnable_bandwidth is set. '
            'The tokens are the keys of a Gaussian (or von Mises-Fisher) kernel '
            'density; start points (the masked mean '
            'and, optionally, tokens at evenly spaced ranks) move for num_steps '
            'mean-shift steps against the fixed tokens and the start with the '
            'highest density is returned, so a minority of far-away tokens does '
            'not pull the summary the way a mean does. The bandwidth is the '
            'median pairwise distance or a given value, optionally scaled by a '
            'learnable factor.'
        ),
        'required_params': [],
        'optional_params': {
            'num_steps': 5,
            'bandwidth': 'auto',
            'bandwidth_scale': 1.0,
            'learnable_bandwidth': False,
            'num_starts': 1,
            'kernel': 'gaussian',
            'exclude_positions': None,
            'return_density': False,
        },
        'use_case': (
            'Summarising a token set that holds a dominant cluster plus '
            'outliers or a second minority cluster: robust sentence or patch '
            'embeddings, and pooling where the mean lands between modes. '
            'return_density also gives a log density usable as a confidence.'
        ),
    },

    'weighted': {
        'class': WeightedPooling,
        'description': (
            'Learnable, content-agnostic per-position weighted pooling. A scalar '
            'learnable weight is assigned to each position up to max_seq_len; the '
            'weights for the current sequence length are softmax-normalised and '
            'used to compute a weighted sum, capturing positional importance '
            'patterns rather than input-dependent relevance.'
        ),
        'required_params': [],
        'optional_params': {
            'max_seq_len': 512,
            'dropout_rate': 0.0,
            'temperature': 1.0,
            'initializer': 'ones',
            'regularizer': None,
        },
        'use_case': (
            'A middle ground between simple mean pooling and full attention '
            'pooling for fixed-length or positionally-structured sequences, '
            'where token position carries consistent importance signal.'
        ),
    },

    'sequence': {
        'class': SequencePooling,
        'description': (
            'Unified, configurable pooling facade exposing 18 strategies '
            '(positional, statistical, learnable, top-k, and special) and 4 '
            'aggregation methods (concat, add, multiply, weighted_sum). One or '
            'more strategies are applied and their outputs combined, internally '
            'composing the attention and weighted poolers for learnable modes.'
        ),
        'required_params': [],
        'optional_params': {
            'strategy': 'mean',
            'exclude_positions': None,
            'aggregation_method': 'concat',
            'attention_hidden_dim': 256,
            'attention_num_heads': 1,
            'attention_dropout_rate': 0.0,
            'weighted_max_seq_len': 512,
            'top_k': 10,
            'temperature': 1.0,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
        },
        'use_case': (
            'The default choice when bridging a sequence encoder (Transformer, '
            'LSTM) to a fixed-size downstream head, and when experimenting with '
            'or combining multiple pooling strategies behind a single interface.'
        ),
    },
}
"""
Registry of sequence-pooling layer implementations with metadata.

Each entry contains:
    - class: The actual layer class implementation.
    - description: Technical description of the pooling mechanism.
    - required_params: List of mandatory parameters for instantiation.
    - optional_params: Dict of optional parameters with default values.
    - use_case: Scenarios and applications where this pooling excels.
"""


STRICT_DROPPED_KEY_MARKER: str = "unsupported parameter(s)"
"""Substring every undeclared-keyword ``ValueError`` from this factory carries.

The same literal is defined in the other strict factories (attention, ffn,
embedding, activations, sampling, ssm, adapters); importing a peer factory would
pull its layer tree into this module's import graph.
``tests/test_layers/test_activations/test_activation_factory.py::TestStrictDroppedKeys::
test_marker_is_identical_in_every_factory`` finds every definition in ``src``
and fails if one drifts.
"""

# Base ``keras.layers.Layer`` kwargs. All four pooling classes forward
# ``**kwargs`` to ``Layer.__init__``, so these build and are not refused. The
# same frozenset lives in ``norms/factory.py`` and is pinned identical by the
# same test class as the marker above.
_KERAS_BASE_PARAMS = frozenset(
    {'name', 'dtype', 'trainable', 'activity_regularizer', 'autocast'}
)


# ---------------------------------------------------------------------
# Public API Functions
# ---------------------------------------------------------------------

def get_sequence_pooling_info() -> Dict[str, Dict[str, Any]]:
    """
    Retrieve metadata for all available sequence-pooling layer types.

    Provides per-type metadata including the technical description, parameter
    specifications, and use cases for every supported pooling mechanism.

    Returns:
        A dictionary mapping each pooling type to its metadata (description,
        required_params, optional_params, use_case). Each entry is a deep
        copy (the layer class itself is shared), so callers cannot mutate the
        registry.
    """
    return {
        pool_type: copy.deepcopy(info)
        for pool_type, info in SEQUENCE_POOLING_REGISTRY.items()
    }


def validate_sequence_pooling_config(
        pooling_type: str,
        **kwargs: Any
) -> None:
    """
    Validate sequence-pooling configuration parameters.

    Performs type-existence checking, required-parameter completeness, and light
    value-range validation on any numeric parameters that are present.

    Args:
        pooling_type: The pooling-layer type to validate against.
        **kwargs: Parameter dictionary to validate for the specified type.

    Raises:
        ValueError: If pooling_type is not supported, a keyword is not
            declared for it (the message carries
            :data:`STRICT_DROPPED_KEY_MARKER`), required parameters are
            missing, a provided parameter value violates its constraint, or a
            ``'sequence'`` strategy or aggregation name is unknown.
    """
    # DECISION plan-2026-10-14T062616-81938e81/D-200: refuse an undeclared
    # keyword here, in the one function both entry points run, instead of
    # filtering it out in create_sequence_pooling_layer. The filter turned
    # `dropout=0.5` (for `dropout_rate`) into a pooler with no dropout and no
    # error. Do NOT restore the filter; the Keras base kwargs stay exempt.
    info = SEQUENCE_POOLING_REGISTRY.get(pooling_type)
    if info is not None:
        declared = (
            set(info['required_params'])
            | set(info['optional_params'])
            | _KERAS_BASE_PARAMS
        )
        undeclared = sorted(set(kwargs) - declared)
        if undeclared:
            raise ValueError(
                f"validate_sequence_pooling_config('{pooling_type}'): "
                f"{len(undeclared)} {STRICT_DROPPED_KEY_MARKER} {undeclared}. "
                f"'{pooling_type}' ({info['class'].__name__}) accepts only "
                f"{sorted(declared)}."
            )

    if pooling_type not in SEQUENCE_POOLING_REGISTRY:
        available_types = list(SEQUENCE_POOLING_REGISTRY.keys())
        raise ValueError(
            f"Unknown sequence pooling type '{pooling_type}'. "
            f"Available types: {available_types}"
        )

    info = SEQUENCE_POOLING_REGISTRY[pooling_type]
    required = info['required_params']
    missing = [p for p in required if p not in kwargs]
    if missing:
        raise ValueError(
            f"Required parameters for '{pooling_type}' are missing: {missing}. "
            f"Required: {required}, Provided: {list(kwargs.keys())}"
        )

    # The layers' own rules (utils/validation.py), so the factory refuses
    # exactly what the constructors refuse.
    positive_int_params = [
        'hidden_dim', 'num_heads', 'max_seq_len', 'attention_hidden_dim',
        'attention_num_heads', 'weighted_max_seq_len', 'top_k', 'num_starts'
    ]
    for param in positive_int_params:
        if param in kwargs:
            positive_int(kwargs[param], param, f"sequence pooling '{pooling_type}'")
    for param in ('temperature', 'bandwidth_scale'):
        if param in kwargs:
            positive_float(
                kwargs[param], param, f"sequence pooling '{pooling_type}'"
            )

    # Validate probability/rate parameters (0.0 to 1.0)
    rate_params = ['dropout_rate', 'attention_dropout_rate']
    for param in rate_params:
        if param in kwargs and not (0.0 <= kwargs[param] <= 1.0):
            raise ValueError(
                f"Parameter '{param}' must be between 0.0 and 1.0, "
                f"got {kwargs[param]}"
            )

    if pooling_type == 'sequence':
        check_strategy_config(
            kwargs.get('strategy', info['optional_params']['strategy']),
            kwargs.get(
                'aggregation_method',
                info['optional_params']['aggregation_method'],
            ),
        )

    logger.debug(
        f"Validation successful for '{pooling_type}' with parameters: {kwargs}"
    )


def create_sequence_pooling_layer(
        pooling_type: SequencePoolingType,
        name: Optional[str] = None,
        **kwargs: Any
) -> keras.layers.Layer:
    """
    Factory function for creating sequence-pooling layers.

    Provides a centralized, type-safe way to instantiate any sequence-pooling
    layer in this package, with parameter validation, default-value handling,
    and detailed error reporting. Dispatch is pure registry lookup (no if/elif).

    Args:
        pooling_type: The type of pooling layer to create
            (``'attention'``, ``'mode'``, ``'weighted'``, or ``'sequence'``).
        name: Optional name for the layer instance.
        **kwargs: Type-specific parameters for the pooling layer, plus any of
            ``dtype``, ``trainable``, ``activity_regularizer``, ``autocast``.
            See ``get_sequence_pooling_info()`` for per-type parameter specs.

    Returns:
        A fully configured and instantiated sequence-pooling layer.

    Raises:
        ValueError: Every failure path. A keyword the type does not declare
            gives a message carrying :data:`STRICT_DROPPED_KEY_MARKER`; an
            unknown type, a bad value or a constructor error gives a message
            beginning ``Failed to create`` or the validator's own message.
    """
    validate_sequence_pooling_config(pooling_type, **kwargs)

    info = SEQUENCE_POOLING_REGISTRY[pooling_type]
    pool_class = info['class']

    # Defaults first, then the caller's values (every key is declared).
    params = dict(info['optional_params'])
    params.update(kwargs)
    if name:
        params['name'] = name

    logger.info(
        f"Creating '{pooling_type}' sequence pooling layer "
        f"({pool_class.__name__}) with parameters: {params}"
    )

    try:
        return pool_class(**params)
    except (TypeError, ValueError) as e:
        error_msg = (
            f"Failed to create '{pooling_type}' sequence pooling layer "
            f"({pool_class.__name__}). "
            f"Provided parameters: {list(kwargs.keys())}. Original error: {e}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def create_sequence_pooling_from_config(
        config: Dict[str, Any]
) -> keras.layers.Layer:
    """
    Create a sequence-pooling layer from a configuration dictionary.

    Convenience function for instantiating pooling layers from dictionary-based
    configurations, useful for loading architectures from JSON/YAML files,
    hyperparameter optimization, and configuration-driven model building.

    Args:
        config: Configuration dictionary containing a ``'type'`` key specifying
            the pooling-layer type and additional keys for layer-specific
            parameters.

    Returns:
        Instantiated and configured sequence-pooling layer.

    Raises:
        ValueError: If config is missing the required ``'type'`` key.
        TypeError: If config parameter types are invalid.
    """
    config_copy = config.copy()
    try:
        pooling_type = config_copy.pop('type')
    except KeyError as e:
        available_keys = list(config.keys()) if config else []
        raise ValueError(
            f"Configuration dictionary must include a 'type' key specifying the "
            f"sequence pooling layer type. Available keys in config: "
            f"{available_keys}. "
            f"Valid types: {list(SEQUENCE_POOLING_REGISTRY.keys())}"
        ) from e

    logger.debug(f"Creating sequence pooling layer from config: {config}")
    return create_sequence_pooling_layer(pooling_type, **config_copy)


def list_sequence_pooling_types() -> List[str]:
    """
    Get a list of all supported sequence-pooling layer types.

    Returns:
        Alphabetically sorted list of supported pooling-layer types.
    """
    return sorted(list(SEQUENCE_POOLING_REGISTRY.keys()))
