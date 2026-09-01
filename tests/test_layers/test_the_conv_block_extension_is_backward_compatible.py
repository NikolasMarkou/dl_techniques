"""Guard: widening ``standard_blocks.ConvBlock`` changed nothing for its existing consumers.

``ConvBlock`` gained ``groups``, ``use_bias`` and an activation-off route
(``activation_type='linear'`` via ``resolve_activation_layer``). Three ``src/``
packages already consume the class:
``layers/heads/vision/factory.py``, ``models/vision/fractalnet/model.py`` and
``layers/fractal_block.py`` — the last of which rebuilds copies through
``ConvBlock.from_config(...)``, i.e. it is sensitive to the ``get_config()`` key
set itself and not only to the constructor.

The pinned numbers below (``29416`` / ``94762`` and the two epsilon censuses) were
measured at commit ``607ffcea9``, BEFORE the widening. They are the whole point of
this module: a difference of one parameter means a default changed. The census is
read with the repo's own oracle ``tests/norm_epsilon_oracle.py::_epsilon_of`` over
``_flatten_layers(include_self=True)`` — a hand-rolled walker was tried first and
silently under-counted the head to ``{}``.

Reference: ``plans/plan-2026-09-01T055648-e6d380a5`` invariants I3 and I6,
decisions D-001/D-004/D-009.
"""

from typing import Any, Dict

import keras
import numpy as np
import pytest

from dl_techniques.layers.standard_blocks import ConvBlock
from dl_techniques.layers.heads.vision.factory import create_vision_head
from dl_techniques.models.vision.fractalnet.model import create_fractal_net

from ..norm_epsilon_oracle import _epsilon_of

# The `get_config()` key set of `standard_blocks.ConvBlock` measured at 607ffcea9,
# before this plan touched the class. 17 keys.
BASELINE_CONFIG_KEYS = frozenset({
    'activation_kwargs', 'activation_type', 'dropout_rate', 'dtype', 'filters',
    'kernel_initializer', 'kernel_regularizer', 'kernel_size', 'name',
    'normalization_kwargs', 'normalization_type', 'padding', 'pool_size',
    'pool_type', 'strides', 'trainable', 'use_pooling',
})


def _norm_epsilon_census(model: keras.layers.Layer) -> Dict[str, int]:
    """Bucket every normalization epsilon in ``model``'s whole sub-layer tree."""
    census: Dict[str, int] = {}
    for sub in model._flatten_layers(include_self=True, recursive=True):
        if "norm" not in type(sub).__name__.lower():
            continue
        epsilon = _epsilon_of(sub)
        if epsilon is None:
            continue
        key = f"{epsilon:.0e}"
        census[key] = census.get(key, 0) + 1
    return census


def _build_vision_head() -> keras.layers.Layer:
    keras.utils.set_random_seed(0)
    head = create_vision_head("detection", num_classes=4, hidden_dim=32)
    head(keras.ops.zeros((1, 16, 16, 32)))
    return head


def _build_fractalnet() -> keras.Model:
    keras.utils.set_random_seed(0)
    return create_fractal_net("micro", num_classes=10, input_shape=(32, 32, 3))


# --------------------------------------------------------------------------
# (a) the config key set grew by exactly {'groups', 'use_bias'}
# --------------------------------------------------------------------------

def test_the_config_key_set_grew_by_exactly_groups_and_use_bias() -> None:
    config = ConvBlock(filters=8).get_config()
    added = set(config) - set(BASELINE_CONFIG_KEYS)
    removed = set(BASELINE_CONFIG_KEYS) - set(config)
    assert added == {"groups", "use_bias"}, (
        f"config key delta is {added!r}, expected exactly "
        f"{{'groups', 'use_bias'}}"
    )
    assert removed == set(), f"config lost keys: {removed!r}"
    assert config["groups"] == 1, f"groups default is {config['groups']!r}, expected 1"
    assert config["use_bias"] is True, (
        f"use_bias default is {config['use_bias']!r}, expected True"
    )


def test_the_new_parameters_are_real_init_parameters() -> None:
    """`fractal_block.py` rebuilds via `from_config`, so these cannot be kwargs-only."""
    block = ConvBlock(filters=8, groups=2, use_bias=False)
    assert block.groups == 2
    assert block.use_bias is False
    assert block.conv.groups == 2
    assert block.conv.use_bias is False
    config = block.get_config()
    assert config["groups"] == 2
    assert config["use_bias"] is False


def test_a_non_positive_groups_is_rejected() -> None:
    with pytest.raises(ValueError, match="groups must be positive"):
        ConvBlock(filters=64, groups=0)
    with pytest.raises(ValueError, match="groups must be positive"):
        ConvBlock(filters=64, groups=-1)


# --------------------------------------------------------------------------
# (b) the two foreign consumers are bit-for-bit the same build
# --------------------------------------------------------------------------

def test_the_vision_head_reproduces_its_pre_widening_build() -> None:
    head = _build_vision_head()
    assert head.count_params() == 29416, (
        f"vision head count_params() is {head.count_params()}, "
        f"pinned at 29416 (measured pre-widening at 607ffcea9)"
    )
    assert _norm_epsilon_census(head) == {"1e-06": 3}, (
        f"vision head norm-epsilon census is {_norm_epsilon_census(head)}, "
        f"pinned at {{'1e-06': 3}}"
    )


def test_the_fractalnet_build_reproduces_its_pre_widening_build() -> None:
    model = _build_fractalnet()
    assert model.count_params() == 94762, (
        f"fractalnet count_params() is {model.count_params()}, "
        f"pinned at 94762 (measured pre-widening at 607ffcea9)"
    )
    assert _norm_epsilon_census(model) == {"1e-06": 7}, (
        f"fractalnet norm-epsilon census is {_norm_epsilon_census(model)}, "
        f"pinned at {{'1e-06': 7}}"
    )


# --------------------------------------------------------------------------
# (c) activation_type='linear' is an EXACT, weightless identity
# --------------------------------------------------------------------------

def test_a_linear_activation_is_an_exact_weightless_identity() -> None:
    keras.utils.set_random_seed(0)
    block = ConvBlock(filters=4, kernel_size=3, activation_type="linear")
    x = keras.ops.convert_to_tensor(
        np.random.default_rng(0).standard_normal((2, 8, 8, 3)).astype("float32")
    )
    out = block(x, training=False)

    # The same pipeline with the activation step omitted entirely.
    reference = block.norm(block.conv(x), training=False)

    delta = float(keras.ops.max(keras.ops.abs(out - reference)))
    assert delta == 0.0, (
        f"activation_type='linear' is not an exact identity: "
        f"max|out - conv->bn| == {delta!r}, expected exactly 0.0"
    )

    activation_weights = list(block.activation.weights)
    assert activation_weights == [], (
        f"the 'linear' activation contributed weights: {activation_weights!r}"
    )

    relu_block = ConvBlock(filters=4, kernel_size=3, activation_type="relu")
    relu_block(x, training=False)
    assert block.count_params() == relu_block.count_params(), (
        "the 'linear' route changed the parameter count relative to 'relu'"
    )


def test_a_registry_activation_still_routes_through_the_registry() -> None:
    """`resolve_activation_layer` must not have demoted registry keys to Keras strings."""
    from dl_techniques.layers.activations.expanded_activations import SiLU

    block = ConvBlock(filters=4, activation_type="silu")
    assert isinstance(block.activation, SiLU), (
        f"activation_type='silu' built {type(block.activation).__name__}, "
        f"expected the registry class SiLU"
    )


# --------------------------------------------------------------------------
# (d) I6 — activation_kwargs aimed at a non-registry name must RAISE
# --------------------------------------------------------------------------

def test_activation_kwargs_for_a_non_registry_activation_raises() -> None:
    with pytest.raises(ValueError, match="ACTIVATION_REGISTRY"):
        ConvBlock(filters=8, activation_type="linear", activation_kwargs={"k": 3})
    with pytest.raises(ValueError, match="ACTIVATION_REGISTRY"):
        ConvBlock(filters=8, activation_type="sigmoid", activation_kwargs={"axis": -1})


def test_activation_kwargs_for_a_registry_activation_still_works() -> None:
    block = ConvBlock(filters=8, activation_type="relu_k", activation_kwargs={"k": 3})
    assert block.activation_kwargs == {"k": 3}
    assert block.get_config()["activation_kwargs"] == {"k": 3}


# --------------------------------------------------------------------------
# (e)/(f) D-009 — `fractal_block.py` rebuilds copies via `from_config`
# --------------------------------------------------------------------------

def test_the_from_config_round_trip_reconstructs_an_equivalent_layer() -> None:
    original = ConvBlock(
        filters=16,
        kernel_size=3,
        groups=2,
        use_bias=False,
        activation_type="linear",
        normalization_kwargs={"epsilon": 1e-3, "momentum": 0.97},
    )
    rebuilt = ConvBlock.from_config(original.get_config())
    for key in ("filters", "kernel_size", "groups", "use_bias", "activation_type"):
        assert getattr(rebuilt, key) == getattr(original, key), (
            f"round trip changed {key}: "
            f"{getattr(original, key)!r} -> {getattr(rebuilt, key)!r}"
        )
    assert rebuilt.normalization_kwargs == {"epsilon": 1e-3, "momentum": 0.97}
    assert rebuilt.conv.groups == 2
    assert rebuilt.conv.use_bias is False


def test_an_old_style_seventeen_key_config_still_constructs() -> None:
    """A config written by the PRE-widening class carries neither new key."""
    config: Dict[str, Any] = ConvBlock(filters=12).get_config()
    del config["groups"]
    del config["use_bias"]
    assert set(config) == set(BASELINE_CONFIG_KEYS), (
        f"the reduced config is not the pre-widening 17-key set: "
        f"{set(config) ^ set(BASELINE_CONFIG_KEYS)!r}"
    )
    rebuilt = ConvBlock.from_config(config)
    assert rebuilt.groups == 1, f"old config gave groups={rebuilt.groups!r}, expected 1"
    assert rebuilt.use_bias is True, (
        f"old config gave use_bias={rebuilt.use_bias!r}, expected True"
    )
