"""Factory for transformer blocks.

This module provides a config-driven construction function for the transformer
blocks in this package. The factory validates the configuration against each
block's declared parameter schema and rejects unrecognized keys.

Registered block types:
- ``"transformer_layer"``: :class:`TransformerLayer`
- ``"transformer_decoder_layer"``: :class:`TransformerDecoderLayer`
- ``"swin_transformer_block"``: :class:`SwinTransformerBlock`
- ``"swin_conv_block"``: :class:`SwinConvBlock`
- ``"perceiver_transformer_layer"``: :class:`PerceiverTransformerLayer`
- ``"eomt_transformer"``: :class:`EomtTransformer`
- ``"adaln_zero_conditional_block"``: :class:`AdaLNZeroConditionalBlock`
- ``"free_transformer_layer"``: :class:`FreeTransformerLayer`
- ``"progressive_focused_transformer_block"``: :class:`PFTBlock`
- ``"gated_linear_attention_block"``: :class:`GatedLinearAttentionBlock`
- ``"area_attention_block"``: :class:`AreaAttentionBlock`
- ``"energy_transformer"``: :class:`EnergyTransformer`
- ``"hopfield_network"``: :class:`HopfieldNetwork`

Each entry declares its ``required_params`` and ``optional_params``. A
``ValueError`` is raised if the config contains a key not in the union of the
two sets.

The public entry point is :func:`create_transformer_block`.
"""

from typing import Any, Dict, Optional

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
from .progressive_focused_transformer import PFTBlock
from .gated_linear_attention_block import GatedLinearAttentionBlock
from .area_attention_block import AreaAttentionBlock
from .energy_transformer import EnergyTransformer, HopfieldNetwork

# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------

# Each entry must have:
# - "cls": the class to instantiate
# - "required_params": list of required config keys (strings)
# - "optional_params": list of optional config keys (strings)
# The union of required + optional is the accepted key set.
# Keys not in this union raise ValueError at construction time.

TRANSFORMER_BLOCK_REGISTRY = {
    "transformer_layer": {
        "cls": TransformerLayer,
        "required_params": [
            "dim",
            "num_heads",
            "max_seq_len",
        ],
        "optional_params": [
            "head_dim",
            "ffn_type",
            "ffn_args",
            "intermediate_size",
            "attention_type",
            "attention_args",
            "normalization_type",
            "normalization_position",
            "norm_args",
            "attention_norm_args",
            "ffn_norm_args",
            "dropout_rate",
            "attention_dropout_rate",
            "use_bias",
            "kernel_initializer",
            "bias_initializer",
            "kernel_regularizer",
            "bias_regularizer",
            "layer_scale_init",
            "stochastic_depth_rate",
        ],
    },
    "transformer_decoder_layer": {
        "cls": TransformerDecoderLayer,
        "required_params": [
            "dim",
            "num_heads",
            "max_seq_len",
        ],
        "optional_params": [
            "head_dim",
            "ffn_type",
            "ffn_args",
            "intermediate_size",
            "attention_type",
            "attention_args",
            "cross_attention_type",
            "cross_attention_args",
            "normalization_type",
            "normalization_position",
            "norm_args",
            "attention_norm_args",
            "ffn_norm_args",
            "cross_attention_norm_args",
            "dropout_rate",
            "attention_dropout_rate",
            "use_bias",
            "kernel_initializer",
            "bias_initializer",
            "kernel_regularizer",
            "bias_regularizer",
            "layer_scale_init",
            "stochastic_depth_rate",
        ],
    },
    "swin_transformer_block": {
        "cls": SwinTransformerBlock,
        "required_params": [
            "dim",
            "num_heads",
            "window_size",
        ],
        "optional_params": [
            "head_dim",
            "ffn_type",
            "ffn_args",
            "intermediate_size",
            "shift_size",
            "attention_type",
            "attention_args",
            "normalization_type",
            "norm_args",
            "dropout_rate",
            "attention_dropout_rate",
            "use_bias",
            "kernel_initializer",
            "bias_initializer",
            "kernel_regularizer",
            "bias_regularizer",
            "layer_scale_init",
            "stochastic_depth_rate",
        ],
    },
    "swin_conv_block": {
        "cls": SwinConvBlock,
        "required_params": [
            "dim",
            "num_heads",
            "window_size",
        ],
        "optional_params": [
            "head_dim",
            "ffn_type",
            "ffn_args",
            "intermediate_size",
            "shift_size",
            "conv_kernel_size",
            "attention_type",
            "attention_args",
            "normalization_type",
            "norm_args",
            "dropout_rate",
            "attention_dropout_rate",
            "use_bias",
            "kernel_initializer",
            "bias_initializer",
            "kernel_regularizer",
            "bias_regularizer",
            "layer_scale_init",
            "stochastic_depth_rate",
        ],
    },
    "perceiver_transformer_layer": {
        "cls": PerceiverTransformerLayer,
        "required_params": [
            "dim",
            "num_heads",
            "num_latents",
            "latent_dim",
        ],
        "optional_params": [
            "head_dim",
            "ffn_type",
            "ffn_args",
            "intermediate_size",
            "attention_type",
            "attention_args",
            "normalization_type",
            "norm_args",
            "dropout_rate",
            "attention_dropout_rate",
            "use_bias",
            "kernel_initializer",
            "bias_initializer",
            "kernel_regularizer",
            "bias_regularizer",
            "layer_scale_init",
            "stochastic_depth_rate",
        ],
    },
    "eomt_transformer": {
        "cls": EomtTransformer,
        "required_params": [
            "dim",
            "num_heads",
            "num_object_queries",
        ],
        "optional_params": [
            "head_dim",
            "ffn_type",
            "ffn_args",
            "intermediate_size",
            "attention_type",
            "attention_args",
            "normalization_type",
            "norm_args",
            "dropout_rate",
            "attention_dropout_rate",
            "use_bias",
            "kernel_initializer",
            "bias_initializer",
            "kernel_regularizer",
            "bias_regularizer",
            "layer_scale_init",
            "stochastic_depth_rate",
            "use_masked_attention",
        ],
    },
    "adaln_zero_conditional_block": {
        "cls": AdaLNZeroConditionalBlock,
        "required_params": [
            "dim",
            "num_heads",
            "cond_dim",
        ],
        "optional_params": [
            "head_dim",
            "ffn_type",
            "ffn_args",
            "intermediate_size",
            "attention_type",
            "attention_args",
            "normalization_type",
            "norm_args",
            "dropout_rate",
            "attention_dropout_rate",
            "use_bias",
            "kernel_initializer",
            "bias_initializer",
            "kernel_regularizer",
            "bias_regularizer",
            "layer_scale_init",
            "stochastic_depth_rate",
        ],
    },
    "free_transformer_layer": {
        "cls": FreeTransformerLayer,
        "required_params": [
            "dim",
            "num_heads",
            "max_seq_len",
        ],
        "optional_params": [
            "head_dim",
            "ffn_type",
            "ffn_args",
            "intermediate_size",
            "attention_type",
            "attention_args",
            "normalization_type",
            "norm_args",
            "dropout_rate",
            "attention_dropout_rate",
            "use_bias",
            "kernel_initializer",
            "bias_initializer",
            "kernel_regularizer",
            "bias_regularizer",
            "layer_scale_init",
            "stochastic_depth_rate",
            "num_register_tokens",
            "use_bias_free_attention",
        ],
    },
    "progressive_focused_transformer_block": {
        "cls": PFTBlock,
        "required_params": [
            "dim",
            "num_heads",
        ],
        "optional_params": [
            "head_dim",
            "ffn_type",
            "ffn_args",
            "intermediate_size",
            "attention_type",
            "attention_args",
            "normalization_type",
            "norm_args",
            "dropout_rate",
            "attention_dropout_rate",
            "use_bias",
            "kernel_initializer",
            "bias_initializer",
            "kernel_regularizer",
            "bias_regularizer",
            "layer_scale_init",
            "stochastic_depth_rate",
            "local_window_size",
            "global_ratio",
        ],
    },
    "gated_linear_attention_block": {
        "cls": GatedLinearAttentionBlock,
        "required_params": [
            "dim",
            "num_heads",
            "max_seq_len",
        ],
        "optional_params": [
            "head_dim",
            "conv_kernel_size",
            "dropout_rate",
            "activation",
            "normalization_type",
            "q_norm_args",
            "k_norm_args",
            "v_norm_args",
            "ffn_type",
            "ffn_args",
            "intermediate_size",
            "chunk_size",
            "use_bias",
            "kernel_initializer",
            "bias_initializer",
            "kernel_regularizer",
            "bias_regularizer",
        ],
    },
    "area_attention_block": {
        "cls": AreaAttentionBlock,
        "required_params": [
            "dim",
            "num_heads",
        ],
        "optional_params": [
            "head_dim",
            "ffn_type",
            "ffn_args",
            "intermediate_size",
            "attention_type",
            "attention_args",
            "normalization_type",
            "norm_args",
            "dropout_rate",
            "attention_dropout_rate",
            "use_bias",
            "kernel_initializer",
            "bias_initializer",
            "kernel_regularizer",
            "bias_regularizer",
            "layer_scale_init",
            "stochastic_depth_rate",
        ],
    },
    "energy_transformer": {
        "cls": EnergyTransformer,
        "required_params": [
            "dim",
            "num_heads",
            "energy_steps",
        ],
        "optional_params": [
            "head_dim",
            "ffn_type",
            "ffn_args",
            "intermediate_size",
            "attention_type",
            "attention_args",
            "normalization_type",
            "norm_args",
            "dropout_rate",
            "attention_dropout_rate",
            "use_bias",
            "kernel_initializer",
            "bias_initializer",
            "kernel_regularizer",
            "bias_regularizer",
            "layer_scale_init",
            "stochastic_depth_rate",
            "energy_lr",
            "energy_init_scale",
        ],
    },
    "hopfield_network": {
        "cls": HopfieldNetwork,
        "required_params": [
            "dim",
            "num_heads",
            "beta",
        ],
        "optional_params": [
            "head_dim",
            "ffn_type",
            "ffn_args",
            "intermediate_size",
            "attention_type",
            "attention_args",
            "normalization_type",
            "norm_args",
            "dropout_rate",
            "attention_dropout_rate",
            "use_bias",
            "kernel_initializer",
            "bias_initializer",
            "kernel_regularizer",
            "bias_regularizer",
            "layer_scale_init",
            "stochastic_depth_rate",
            "update_steps",
            "energy_lr",
        ],
    },
}


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

    Unrecognized keys (not in the union of ``required_params`` and
    ``optional_params`` for the selected type) raise ``ValueError`` naming
    the offending key.

    :param config: Configuration dictionary with a ``"type"`` key and
        constructor arguments for the selected block type.
    :type config: Dict[str, Any]
    :param name: Optional name for the created layer. If provided, overrides
        any ``"name"`` key in ``config``.
    :type name: Optional[str]
    :return: An instance of the requested transformer block class.
    :rtype: keras.layers.Layer
    :raises ValueError: If ``"type"`` is missing, unknown, or if ``config``
        contains unrecognized keys.
    """
    if "type" not in config:
        raise ValueError(
            "Transformer block config must contain a 'type' key. "
            f"Available types: {sorted(TRANSFORMER_BLOCK_REGISTRY)}"
        )

    block_type = config["type"]
    info = TRANSFORMER_BLOCK_REGISTRY.get(block_type)
    if info is None:
        raise ValueError(
            f"Unknown transformer block type '{block_type}'. "
            f"Available: {sorted(TRANSFORMER_BLOCK_REGISTRY)}"
        )

    cls = info["cls"]
    valid_keys = set(info["required_params"]) | set(info["optional_params"]) | {"type", "name"}

    for key in config:
        if key not in valid_keys:
            raise ValueError(
                f"Unrecognized key '{key}' for transformer block type '{block_type}'. "
                f"Valid keys: {sorted(valid_keys)}"
            )

    # Check required params are present
    for req in info["required_params"]:
        if req not in config:
            raise ValueError(
                f"Missing required parameter '{req}' for transformer block type '{block_type}'. "
                f"Required: {info['required_params']}"
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