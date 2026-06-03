"""
Sequence Pooling Layers Module.

This package collapses a sequence of token vectors
``(batch, seq_len, hidden_dim)`` into a fixed-size summary representation. It
exposes three pooling layers and a registry-driven factory:

    - :class:`AttentionPooling` — learnable, content-aware attention pooling.
    - :class:`WeightedPooling` — learnable, content-agnostic per-position pooling.
    - :class:`SequencePooling` — a unified facade over 18 strategies and 4
      aggregation methods (composing the two leaf poolers for learnable modes).

It provides a factory interface (`create_sequence_pooling_layer`) for unified,
config-driven instantiation alongside direct access to all layer classes and the
``PoolingStrategy`` / ``AggregationMethod`` type aliases.
"""

# Factory and Utility Functions
from .factory import (
    create_sequence_pooling_layer,
    create_sequence_pooling_from_config,
    validate_sequence_pooling_config,
    get_sequence_pooling_info,
    list_sequence_pooling_types,
    SequencePoolingType,
    SEQUENCE_POOLING_REGISTRY,
)

# Layer Classes
from .attention_pooling import AttentionPooling
from .weighted_pooling import WeightedPooling
from .sequence_pooling import SequencePooling

# Type Aliases
from .sequence_pooling import PoolingStrategy, AggregationMethod

__all__ = [
    # Factory Interface
    "create_sequence_pooling_layer",
    "create_sequence_pooling_from_config",
    "validate_sequence_pooling_config",
    "get_sequence_pooling_info",
    "list_sequence_pooling_types",
    "SequencePoolingType",
    "SEQUENCE_POOLING_REGISTRY",

    # Layer Classes
    "AttentionPooling",
    "WeightedPooling",
    "SequencePooling",

    # Type Aliases
    "PoolingStrategy",
    "AggregationMethod",
]
