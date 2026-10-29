"""Each projection of a library attention layer starts from its own random draw.

A seedless ``Initializer`` INSTANCE handed to several sub-layers replays one underlying sample at every
site (``initializers/clone.py``), and the smaller draw is a scaled prefix of the larger. Eleven attention
layers resolved ONE instance in ``__init__`` and gave it to every projection, so their ``qkv`` and
``proj`` (or ``q``, ``k``, ``v`` and ``o``) kernels were the same random numbers:

=====================  =====================================  ===========================
registry type          sites that shared one instance         pairs replayed before
=====================  =====================================  ===========================
``anchor``             query, key, value, query_token, output 10 of 10
``differential``       q1, k1, q2, k2, v, proj                15 of 15
``lighthouse``         wq, wk, wv, wo                         6 of 6
``non_local``          depthwise, query, key, value, output   10 of 10
``perceiver``          q, kv, proj of the wrapped engine      3 of 3
``performer``          qkv, output                            1 of 1
``rpc``                qkv, output                            1 of 1
``shared_weights_cross``  qkv, proj                           1 of 1
``single_window``      qkv, proj                              1 of 3
``window``             qkv, proj (inside single_window)       1 of 3
``window_band``        qkv, proj                              1 of 1
``window_zigzag``      qkv, proj                              1 of 1
``wave_field``         qkv, output                            1 of 3
=====================  =====================================  ===========================

Measured at HEAD 3bd2c0c59 by ``step19_3/layer_replay_census_before.txt``. The first projection of each
layer keeps the layer's own initializer (so its seeded draw is unmoved) and every other projection takes
``clone_initializer(...)`` at its own site (decision D-056, carve-out C-8).

What this file pins:

* the replay oracle is empty on the default arm (a string) and on an instance arm (a seedless
  ``HeNormal()``), with the number of compared pairs pinned so a layer that stops exposing kernels cannot
  pass vacuously;
* the first projection keeps the layer's own initializer instance and every other one has its own instance
  of the same class (the seed accounting of C-8);
* the oracle is not blind: ``multi_head`` at its default still replays one pair (the deliberate D-160
  default), a SEEDED instance replays by contract (the exemption in ``clone.py``), and both are asserted;
* the weight paths are unchanged, so archives keyed by path still load.
"""

import itertools
from typing import Dict, List, Tuple

import keras
import numpy as np
import pytest

from dl_techniques.layers.attention.factory import create_attention_layer
from tests.test_models.initializer_replay_oracle import replayed_pairs

# ---------------------------------------------------------------------

_TOKENS = (2, 16, 64)
_MAP = (2, 8, 8, 32)


class Row:
    """One layer under test: how to build it, what to call it with, and its measured shape of the fix."""

    def __init__(self, attention_type: str, kwargs: dict, shape: Tuple[int, ...], compared: int,
                 sites: List[Tuple[str, str]], paths: List[str], call: bool = True,
                 owner: str = "", offset: int = 0, deterministic: bool = True) -> None:
        self.attention_type = attention_type
        self.kwargs = kwargs
        self.shape = shape
        self.compared = compared
        self.sites = sites
        self.paths = paths
        #: ``False`` when the layer has no plain call (``shared_weights_cross`` needs split sizes).
        self.call = call
        #: Dotted path of the sub-layer that holds the instance the first site keeps ("" = the layer).
        #: The window layers forward the raw argument, so the inner ``SingleWindowAttention`` resolves its own.
        self.owner = owner
        #: Default seeds taken before the first site's initializer (the window wrapper resolves one it never uses).
        self.offset = offset
        #: ``False`` when two calls differ on their own (``performer`` redraws its random features per call).
        self.deterministic = deterministic

    def build(self, **changes) -> keras.layers.Layer:
        layer = create_attention_layer(self.attention_type, **{**self.kwargs, **changes})
        x = np.random.default_rng(0).normal(size=self.shape).astype("float32")
        if self.call:
            layer(x)
        else:
            layer.build(self.shape)
        return layer


def _dense_paths(*names: str) -> List[str]:
    return [f"{name}/{leaf}" for name in names for leaf in ("kernel", "bias")]


_WINDOW = {"dim": 64, "window_size": 4, "num_heads": 4}

ROWS: Dict[str, Row] = {
    "anchor": Row(
        "anchor", {"dim": 64, "num_heads": 4}, _TOKENS, 10,
        [("query_proj", "kernel_initializer"), ("key_proj", "kernel_initializer"),
         ("value_proj", "kernel_initializer"), ("query_token_proj", "kernel_initializer"),
         ("output_proj", "kernel_initializer")],
        _dense_paths("query_proj", "key_proj", "value_proj", "query_token_proj", "output_proj")),
    "differential": Row(
        "differential", {"dim": 64, "num_heads": 4, "head_dim": 16}, _TOKENS, 15,
        [("q1_dense", "kernel_initializer"), ("k1_dense", "kernel_initializer"),
         ("q2_dense", "kernel_initializer"), ("k2_dense", "kernel_initializer"),
         ("v_dense", "kernel_initializer"), ("proj", "kernel_initializer")],
        ["lambda_param"] + _dense_paths("q1", "k1", "q2", "k2", "v", "proj")),
    "lighthouse": Row(
        "lighthouse", {"dim": 64, "num_heads": 4}, _TOKENS, 6,
        [("wq", "kernel_initializer"), ("wk", "kernel_initializer"), ("wv", "kernel_initializer"),
         ("wo", "kernel_initializer")],
        ["wq/kernel", "wk/kernel", "wv/kernel", "wo/kernel"]),
    "non_local": Row(
        "non_local", {"attention_channels": 32}, _MAP, 10,
        [("depthwise_conv", "depthwise_initializer"), ("query_conv", "kernel_initializer"),
         ("key_conv", "kernel_initializer"), ("value_conv", "kernel_initializer"),
         ("output_conv", "kernel_initializer")],
        ["depthwise_conv/kernel", "output_norm/gamma", "output_norm/beta", "output_norm/moving_mean",
         "output_norm/moving_variance", "query_conv/kernel", "key_conv/kernel", "value_conv/kernel",
         "output_conv/kernel"]),
    "perceiver": Row(
        "perceiver", {"dim": 64}, _TOKENS, 3,
        [("cross_attention.q_dense", "kernel_initializer"), ("cross_attention.kv_dense", "kernel_initializer"),
         ("cross_attention.proj_dense", "kernel_initializer")],
        [f"cross_attention/{p}" for p in _dense_paths("q", "kv", "proj")]),
    "performer": Row(
        "performer", {"dim": 64}, _TOKENS, 1,
        [("to_qkv", "kernel_initializer"), ("to_out", "kernel_initializer")],
        ["qkv_projection/kernel", "output_projection/kernel"], deterministic=False),
    "rpc": Row(
        "rpc", {"dim": 64}, _TOKENS, 1,
        [("to_qkv", "kernel_initializer"), ("to_out", "kernel_initializer")],
        ["qkv_projection/kernel", "output_projection/kernel"]),
    "shared_weights_cross": Row(
        "shared_weights_cross", {"dim": 64}, _TOKENS, 1,
        [("qkv_dense", "kernel_initializer"), ("proj_dense", "kernel_initializer")],
        _dense_paths("qkv", "proj"), call=False),
    "single_window": Row(
        "single_window", {"dim": 64, "window_size": 7, "num_heads": 8}, _TOKENS, 3,
        [("qkv", "kernel_initializer"), ("proj", "kernel_initializer")],
        ["relative_position_bias_table"] + _dense_paths("qkv", "proj")),
    "wave_field": Row(
        "wave_field", {"dim": 64}, _TOKENS, 3,
        [("qkv_proj", "kernel_initializer"), ("output_proj", "kernel_initializer")],
        ["wave_frequency", "wave_damping", "wave_phase", "field_coupling"]
        + _dense_paths("qkv_proj", "output_proj", "gate_proj")),
    "window": Row(
        "window", _WINDOW, _TOKENS, 3,
        [("attention.qkv", "kernel_initializer"), ("attention.proj", "kernel_initializer")],
        ["single_window_attention/relative_position_bias_table"]
        + [f"single_window_attention/{p}" for p in _dense_paths("qkv", "proj")],
        owner="attention", offset=1),
    "window_band": Row(
        "window_band", _WINDOW, _TOKENS, 1,
        [("attention.qkv", "kernel_initializer"), ("attention.proj", "kernel_initializer")],
        [f"single_window_attention/{p}" for p in _dense_paths("qkv", "proj")],
        owner="attention", offset=1),
    "window_zigzag": Row(
        "window_zigzag", _WINDOW, _TOKENS, 1,
        [("attention.qkv", "kernel_initializer"), ("attention.proj", "kernel_initializer")],
        [f"single_window_attention/{p}" for p in _dense_paths("qkv", "proj")],
        owner="attention", offset=1),
}


def _sub(layer: keras.layers.Layer, dotted: str):
    target = layer
    for part in filter(None, dotted.split(".")):
        target = getattr(target, part)
    return target


def _site_initializer(layer: keras.layers.Layer, dotted: str, attribute: str):
    return getattr(_sub(layer, dotted), attribute)


def _paths(layer: keras.layers.Layer) -> List[str]:
    """Weight paths without the layer's own (auto-numbered) root segment."""
    return [w.path.split("/", 1)[1] if "/" in w.path else w.path for w in layer.weights]


ROW_IDS = list(ROWS)

# ---------------------------------------------------------------------


class TestNoProjectionReplaysAnotherOnesDraw:

    @pytest.mark.parametrize("row_id", ROW_IDS)
    def test_the_default_arm_replays_no_draw(self, row_id: str) -> None:
        keras.utils.set_random_seed(1234)
        layer = ROWS[row_id].build()
        pairs, compared = replayed_pairs(layer)
        assert compared == ROWS[row_id].compared, f"{row_id}: compared {compared} pairs"
        assert not pairs, f"{len(pairs)} of {compared} kernel pairs replay one draw: {pairs[:3]}"

    @pytest.mark.parametrize("row_id", ROW_IDS)
    def test_a_seedless_instance_replays_no_draw(self, row_id: str) -> None:
        keras.utils.set_random_seed(1234)
        layer = ROWS[row_id].build(kernel_initializer=keras.initializers.HeNormal())
        pairs, compared = replayed_pairs(layer)
        assert compared == ROWS[row_id].compared, f"{row_id}: compared {compared} pairs"
        assert not pairs, f"{len(pairs)} of {compared} kernel pairs replay one draw: {pairs[:3]}"

    @pytest.mark.parametrize("row_id", ROW_IDS)
    def test_a_layer_rebuilt_from_its_config_replays_no_draw(self, row_id: str) -> None:
        keras.utils.set_random_seed(1234)
        row = ROWS[row_id]
        layer = row.build()
        clone = type(layer).from_config(layer.get_config())
        x = np.random.default_rng(0).normal(size=row.shape).astype("float32")
        if row.call:
            clone(x)
        else:
            clone.build(row.shape)
        pairs, compared = replayed_pairs(clone)
        assert compared == row.compared
        assert not pairs, f"{len(pairs)} of {compared} kernel pairs replay one draw: {pairs[:3]}"


class TestTheFirstProjectionKeepsTheLayersInitializer:
    """Carve-out C-8: one site stays on the layer's own instance, the others each clone it."""

    @pytest.mark.parametrize("row_id", ROW_IDS)
    def test_the_first_site_is_the_layers_own_instance(self, row_id: str) -> None:
        layer = ROWS[row_id].build()
        first = _site_initializer(layer, *ROWS[row_id].sites[0])
        assert first is _sub(layer, ROWS[row_id].owner).kernel_initializer, f"{row_id}: {ROWS[row_id].sites[0]}"

    @pytest.mark.parametrize("row_id", ROW_IDS)
    def test_every_other_site_has_its_own_instance_of_the_same_class(self, row_id: str) -> None:
        layer = ROWS[row_id].build()
        own = _sub(layer, ROWS[row_id].owner).kernel_initializer
        others = [_site_initializer(layer, *site) for site in ROWS[row_id].sites[1:]]
        assert others, "a row with one site proves nothing"
        for site, initializer in zip(ROWS[row_id].sites[1:], others):
            assert initializer is not own, f"{row_id}: {site} still shares the layer's instance"
            assert type(initializer) is type(own), f"{row_id}: {site} is {type(initializer).__name__}"
        for (site_a, a), (site_b, b) in itertools.combinations(zip(ROWS[row_id].sites[1:], others), 2):
            assert a is not b, f"{row_id}: {site_a} and {site_b} share one instance"

    @pytest.mark.parametrize("row_id", ROW_IDS)
    def test_the_seeds_of_the_sites_are_distinct(self, row_id: str) -> None:
        keras.utils.set_random_seed(7)
        layer = ROWS[row_id].build()
        seeds = [_site_initializer(layer, *site).seed for site in ROWS[row_id].sites]
        assert len(set(seeds)) == len(seeds), f"{row_id}: {seeds}"

    @pytest.mark.parametrize("row_id", ROW_IDS)
    def test_the_first_site_takes_the_first_default_seed_and_the_next_ones_follow(self, row_id: str) -> None:
        """``qkv`` (or its row's first site) keeps the draw it always had; each clone takes the next seed."""
        keras.utils.set_random_seed(7)
        first = ROWS[row_id].offset
        stream = [keras.initializers.HeNormal().seed for _ in range(first + len(ROWS[row_id].sites) + 2)]
        keras.utils.set_random_seed(7)
        layer = ROWS[row_id].build()
        seeds = [_site_initializer(layer, *site).seed for site in ROWS[row_id].sites]
        assert seeds[0] == stream[first], f"{row_id}: the first site moved off its default seed"
        assert seeds[1:] == stream[first + 1:first + len(seeds)], f"{row_id}: {seeds} against {stream}"


class TestTheKanKeyModeSpendsNoSeedOnProj:
    """In ``kan_key`` mode nothing else holds the layer's instance (``query`` and ``value`` draw from their own
    default strings), so ``proj`` keeps it: no clone, no default seed spent, and still no replayed pair."""

    def _layer(self) -> keras.layers.Layer:
        layer = create_attention_layer(
            "single_window", dim=64, window_size=7, num_heads=8, attention_mode="kan_key")
        layer(np.random.default_rng(0).normal(size=_TOKENS).astype("float32"))
        return layer

    def test_proj_keeps_the_layers_own_instance(self) -> None:
        layer = self._layer()
        assert layer.proj.kernel_initializer is layer.kernel_initializer

    def test_no_pair_replays_one_draw(self) -> None:
        keras.utils.set_random_seed(1234)
        pairs, compared = replayed_pairs(self._layer())
        assert compared > 0
        assert not pairs, f"{len(pairs)} of {compared} kernel pairs replay one draw: {pairs[:3]}"


class TestTheOracleAndTheContractAreNotBlind:

    def test_multi_head_at_its_default_still_replays_the_one_pair(self) -> None:
        """Control: the deliberate D-160 default of ``MultiHeadAttention`` is the replay the oracle must see."""
        keras.utils.set_random_seed(1234)
        layer = create_attention_layer("multi_head", dim=64, num_heads=4)
        layer(np.random.default_rng(0).normal(size=_TOKENS).astype("float32"))
        pairs, compared = replayed_pairs(layer)
        assert (len(pairs), compared) == (1, 1)

    def test_multi_head_with_the_output_keyword_replays_nothing(self) -> None:
        keras.utils.set_random_seed(1234)
        layer = create_attention_layer("multi_head", dim=64, num_heads=4, output_kernel_initializer="glorot_uniform")
        layer(np.random.default_rng(0).normal(size=_TOKENS).astype("float32"))
        pairs, compared = replayed_pairs(layer)
        assert compared == 1 and not pairs

    @pytest.mark.parametrize("row_id", ROW_IDS)
    def test_a_seeded_instance_replays_by_contract(self, row_id: str) -> None:
        """``clone_initializer`` reproduces an explicit seed on purpose: seed 7 at every site is one sample."""
        keras.utils.set_random_seed(1234)
        layer = ROWS[row_id].build(kernel_initializer=keras.initializers.HeNormal(seed=7))
        pairs, compared = replayed_pairs(layer)
        assert compared == ROWS[row_id].compared
        assert pairs, f"{row_id}: a seeded initializer must replay (clone.py exemption 1)"


class TestTheWeightPathsAreUnchanged:
    """Archives are keyed by path: the fix moves seeded draws only."""

    @pytest.mark.parametrize("row_id", ROW_IDS)
    def test_the_layer_owns_exactly_these_weights(self, row_id: str) -> None:
        layer = ROWS[row_id].build()
        assert _paths(layer) == ROWS[row_id].paths

    @pytest.mark.parametrize("row_id", ROW_IDS)
    def test_weights_copied_by_path_reproduce_the_outputs(self, row_id: str) -> None:
        """A second layer, other seeds, the first one's weights copied by path: equal output at atol 0."""
        row = ROWS[row_id]
        keras.utils.set_random_seed(1)
        source = row.build()
        keras.utils.set_random_seed(2)
        target = row.build()
        by_path = {p: w for p, w in zip(_paths(source), source.weights)}
        for path, weight in zip(_paths(target), target.weights):
            weight.assign(by_path[path])
        if not row.call or not row.deterministic:
            # No plain call (shared_weights_cross needs split sizes) or a call that redraws on its own
            # (performer's random features): the weights equal by path is the whole claim.
            for a, b in zip(source.weights, target.weights):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
            return
        x = np.random.default_rng(0).normal(size=row.shape).astype("float32")
        np.testing.assert_array_equal(np.asarray(source(x)), np.asarray(target(x)))
