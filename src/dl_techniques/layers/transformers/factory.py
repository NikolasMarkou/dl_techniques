"""Factory for transformer blocks.

This module provides a config-driven construction function for the transformer
blocks in this package. The factory validates the configuration against each
block's constructor signature and rejects unrecognized keys.

Registered block types:
- ``"transformer_layer"``: :class:`TransformerLayer`
- ``"transformer_decoder_layer"``: :class:`TransformerDecoderLayer`
- ``"swin_transformer_block"``: :class:`SwinTransformerBlock`
- ``"swin_conv_block"``: :class:`SwinConvBlock`
- ``"perceiver_transformer_layer"``: :class:`PerceiverTransformerLayer`
- ``"eomt_transformer"``: :class:`EomtTransformer`
- ``"adaln_zero_conditional_block"``: :class:`AdaLNZeroConditionalBlock`
- ``"free_transformer_layer"``: :class:`FreeTransformerLayer`
- ``"gated_linear_attention_block"``: :class:`GatedLinearAttentionBlock`
- ``"area_attention_block"``: :class:`AreaAttentionBlock`
- ``"energy_transformer"``: :class:`EnergyTransformer`
- ``"hopfield_network"``: :class:`HopfieldNetwork`

The accepted keys of each type are its block's own ``__init__`` keywords plus
the Keras base-layer keywords ``name``, ``dtype`` and ``trainable``; the
required keys are the ``__init__`` parameters without a default. Both are read
off the constructor, so the factory cannot drift from the class. A
``ValueError`` names an unrecognized or a missing key.

The public entry point is :func:`create_transformer_block`.
"""

import inspect
from typing import Any, Dict, FrozenSet, Optional, Tuple

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------

from .transformer import (
    TransformerLayer,
    AttentionType,
    FFNType,
    NormalizationType,
    NormalizationPositionType,
)
from .transformer_decoder import TransformerDecoderLayer
from .swin_transformer_block import SwinTransformerBlock
from .swin_conv_block import SwinConvBlock
from .perceiver_transformer import PerceiverTransformerLayer
from .eomt_transformer import EomtTransformer
from .adaln_zero import AdaLNZeroConditionalBlock
from .free_transformer import BinaryMapper, FreeTransformerLayer
from .gated_linear_attention_block import GatedLinearAttentionBlock
from .area_attention_block import AreaAttentionBlock
from .energy_transformer import EnergyTransformer, HopfieldNetwork

# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------

# DECISION plan-2026-10-14T062616-81938e81/D-217
# The registry maps a type to its class and nothing else. Do NOT bring back a
# hand-written required/optional key list per entry: that is what this module
# shipped with, nothing compared it to the constructors, and 11 of its 12 keys
# could not build their block (``transformer_layer`` demanded ``dim`` and
# ``max_seq_len``, which ``TransformerLayer`` does not take). The key sets are
# read from each ``__init__`` by ``_constructor_keywords``. See decisions.md D-217.
TRANSFORMER_BLOCK_REGISTRY: Dict[str, type] = {
    "transformer_layer": TransformerLayer,
    "transformer_decoder_layer": TransformerDecoderLayer,
    "swin_transformer_block": SwinTransformerBlock,
    "swin_conv_block": SwinConvBlock,
    "perceiver_transformer_layer": PerceiverTransformerLayer,
    "eomt_transformer": EomtTransformer,
    "adaln_zero_conditional_block": AdaLNZeroConditionalBlock,
    "free_transformer_layer": FreeTransformerLayer,
    "gated_linear_attention_block": GatedLinearAttentionBlock,
    "area_attention_block": AreaAttentionBlock,
    "energy_transformer": EnergyTransformer,
    "hopfield_network": HopfieldNetwork,
}

#: Keras base-layer keywords every block forwards to ``keras.layers.Layer``
#: through ``**kwargs``.
_LAYER_BASE_KEYWORDS = frozenset({"name", "dtype", "trainable"})


def _constructor_keywords(cls: type) -> Tuple[FrozenSet[str], Tuple[str, ...]]:
    """
    The keywords ``cls.__init__`` names, and those of them without a default.

    :param cls: A block class from :data:`TRANSFORMER_BLOCK_REGISTRY`.
    :type cls: type
    :return: ``(accepted, required)``: every positional-or-keyword and
        keyword-only parameter except ``self``, and the subset with no default,
        in signature order.
    :rtype: Tuple[FrozenSet[str], Tuple[str, ...]]
    """
    kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    params = [
        param for name, param in inspect.signature(cls.__init__).parameters.items()
        if name != "self" and param.kind in kinds
    ]
    accepted = frozenset(param.name for param in params)
    required = tuple(
        param.name for param in params if param.default is inspect.Parameter.empty
    )
    return accepted, required


# ---------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------

def create_transformer_block(
    config: Dict[str, Any],
    *,
    name: Optional[str] = None,
) -> Any:
    """Create a transformer block from a configuration dictionary.

    The ``config`` dict must contain a ``"type"`` key matching one of the
    registered block types in ``TRANSFORMER_BLOCK_REGISTRY``. All other keys
    are passed as constructor arguments to the corresponding class.

    A key that is neither one of the class's ``__init__`` keywords nor
    ``name``/``dtype``/``trainable`` raises ``ValueError`` naming it, and so
    does a missing ``__init__`` parameter that has no default.

    :param config: Configuration dictionary with a ``"type"`` key and
        constructor arguments for the selected block type.
    :type config: Dict[str, Any]
    :param name: Optional name for the created layer. If provided, overrides
        any ``"name"`` key in ``config``.
    :type name: Optional[str]
    :return: An instance of the requested transformer block class.
    :rtype: keras.layers.Layer
    :raises ValueError: If ``"type"`` is missing or unknown, if ``config``
        contains an unrecognized key, or if it lacks a required one.
    """
    if "type" not in config:
        raise ValueError(
            "Transformer block config must contain a 'type' key. "
            f"Available types: {sorted(TRANSFORMER_BLOCK_REGISTRY)}"
        )

    block_type = config["type"]
    cls = TRANSFORMER_BLOCK_REGISTRY.get(block_type)
    if cls is None:
        raise ValueError(
            f"Unknown transformer block type '{block_type}'. "
            f"Available: {sorted(TRANSFORMER_BLOCK_REGISTRY)}"
        )

    accepted, required = _constructor_keywords(cls)
    valid_keys = accepted | _LAYER_BASE_KEYWORDS | {"type"}

    for key in config:
        if key not in valid_keys:
            raise ValueError(
                f"Unrecognized key '{key}' for transformer block type '{block_type}'. "
                f"Valid keys: {sorted(valid_keys)}"
            )

    for req in required:
        if req not in config:
            raise ValueError(
                f"Missing required parameter '{req}' for transformer block type '{block_type}'. "
                f"Required: {list(required)}"
            )

    kwargs = {k: v for k, v in config.items() if k not in ("type", "name")}
    if name is not None:
        kwargs["name"] = name

    return cls(**kwargs)


__all__ = [
    "TRANSFORMER_BLOCK_REGISTRY",
    "create_transformer_block",
    # Re-export type aliases for convenience
    "AttentionType",
    "FFNType",
    "NormalizationType",
    "NormalizationPositionType",
]