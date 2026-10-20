"""Guard: ``create_transformer_block`` builds every block it registers.

Why this file exists
--------------------
Until plan-2026-10-14T062616-81938e81 step 7.1 (D-217) the registry carried a
hand-written ``required_params`` / ``optional_params`` list per block, and no
test ever called the factory. Measured at 43c9576d6, 11 of its 12 keys could
not build their block at all: ``transformer_layer`` required ``dim`` and
``max_seq_len`` (``TransformerLayer`` takes ``hidden_size`` and
``intermediate_size`` and has no ``max_seq_len``), ``energy_transformer``
required ``energy_steps`` (the class takes ``embed_dim``, ``head_dim``,
``hopfield_dim``), and so on; only ``gated_linear_attention_block`` matched its
constructor. The factory now reads the accepted and required keywords off each
block's ``__init__``, so the lists cannot drift again.

What is asserted, per registered key
------------------------------------
1. The block builds from exactly its constructor's required arguments, and
   each value reaches the block's ``get_config``.
2. Every keyword the constructor names is accepted (all passed at their own
   defaults), so a real argument is never refused.
3. An unknown key raises ``ValueError`` naming it; a missing required key
   raises ``ValueError`` naming it (not the class's ``TypeError``).

The registered key set itself is pinned, so adding or deleting a block must
update this file.
"""

import inspect

import pytest

from dl_techniques.layers.transformers import (
    AdaLNZeroConditionalBlock,
    AreaAttentionBlock,
    EnergyTransformer,
    EomtTransformer,
    FreeTransformerLayer,
    GatedLinearAttentionBlock,
    HopfieldNetwork,
    PerceiverTransformerLayer,
    SwinConvBlock,
    SwinTransformerBlock,
    TRANSFORMER_BLOCK_REGISTRY,
    TransformerDecoderLayer,
    TransformerLayer,
    create_transformer_block,
)

#: The class each key must build, written out here rather than read back from
#: the registry, so the registry is checked against something it does not hold.
EXPECTED_CLASS = {
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

#: The smallest constructor arguments that build each block: its required
#: (no-default) ``__init__`` parameters and nothing else.
REQUIRED_ARGS = {
    "transformer_layer": {"hidden_size": 16, "num_heads": 2, "intermediate_size": 32},
    "transformer_decoder_layer": {"hidden_size": 16, "num_heads": 2, "intermediate_size": 32},
    "swin_transformer_block": {"dim": 16, "num_heads": 2},
    "swin_conv_block": {"conv_dim": 32, "trans_dim": 32},  # head_dim defaults to 32
    "perceiver_transformer_layer": {"dim": 16},
    "eomt_transformer": {"hidden_size": 16},
    "adaln_zero_conditional_block": {"dim": 16, "num_heads": 2, "dim_head": 8, "mlp_dim": 32},
    "free_transformer_layer": {"hidden_size": 16, "num_heads": 2, "intermediate_size": 32},
    "gated_linear_attention_block": {"dim": 16, "num_heads": 2, "max_seq_len": 8},
    "area_attention_block": {"dim": 16},
    "energy_transformer": {"embed_dim": 16, "num_heads": 2, "head_dim": 8, "hopfield_dim": 32},
    "hopfield_network": {"dim": 16, "hopfield_dim": 32},
}

KEYS = sorted(REQUIRED_ARGS)


def _block_class(key):
    return EXPECTED_CLASS[key]


def _named_params(cls):
    """``{name: Parameter}`` for every keyword-settable ``__init__`` parameter."""
    kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    return {
        name: param
        for name, param in inspect.signature(cls.__init__).parameters.items()
        if name != "self" and param.kind in kinds
    }


def test_the_registered_keys_are_pinned():
    assert sorted(TRANSFORMER_BLOCK_REGISTRY) == KEYS == sorted(EXPECTED_CLASS)


@pytest.mark.parametrize("key", KEYS)
def test_the_required_arguments_are_the_constructor_ones(key):
    """The table above is the constructor's, not a second hand-kept list."""
    required = {
        name for name, param in _named_params(_block_class(key)).items()
        if param.default is inspect.Parameter.empty
    }
    assert required == set(REQUIRED_ARGS[key])


@pytest.mark.parametrize("key", KEYS)
def test_every_block_builds_from_its_required_arguments(key):
    block = create_transformer_block({"type": key, **REQUIRED_ARGS[key]}, name=f"probe_{key}")
    assert type(block) is _block_class(key)
    assert block.name == f"probe_{key}"
    config = block.get_config()
    for arg, value in REQUIRED_ARGS[key].items():
        assert config[arg] == value, (key, arg)


@pytest.mark.parametrize("key", KEYS)
def test_every_constructor_keyword_is_accepted(key):
    config = {"type": key, **REQUIRED_ARGS[key]}
    for name, param in _named_params(_block_class(key)).items():
        if param.default is not inspect.Parameter.empty:
            config[name] = param.default
    block = create_transformer_block(config)
    assert type(block) is _block_class(key)


@pytest.mark.parametrize("key", KEYS)
def test_an_unknown_key_is_refused_by_name(key):
    with pytest.raises(ValueError, match="not_a_block_argument"):
        create_transformer_block(
            {"type": key, **REQUIRED_ARGS[key], "not_a_block_argument": 1}
        )


@pytest.mark.parametrize("key", KEYS)
def test_a_missing_required_key_is_refused_by_name(key):
    missing = sorted(REQUIRED_ARGS[key])[0]
    config = {"type": key, **REQUIRED_ARGS[key]}
    del config[missing]
    with pytest.raises(ValueError, match=f"'{missing}'"):
        create_transformer_block(config)


def test_an_unknown_type_is_refused():
    with pytest.raises(ValueError, match="Unknown transformer block type"):
        create_transformer_block({"type": "no_such_block"})


def test_a_layer_base_keyword_reaches_the_block():
    block = create_transformer_block(
        {"type": "area_attention_block", "dim": 16, "dtype": "float64"}
    )
    assert block.dtype_policy.compute_dtype == "float64"
