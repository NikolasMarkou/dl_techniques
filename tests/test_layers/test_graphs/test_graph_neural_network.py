"""Tests for the configurable multi-paradigm GraphNeuralNetworkLayer."""

import os
import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.graphs.graph_neural_network import GraphNeuralNetworkLayer

B, N, D = 2, 5, 16


@pytest.fixture
def graph_inputs():
    rng = np.random.default_rng(1)
    nodes = rng.standard_normal((B, N, D)).astype("float32")
    adj = (rng.uniform(size=(B, N, N)) > 0.5).astype("float32")
    return nodes, adj


class TestGraphNeuralNetworkLayer:

    def test_construction(self):
        layer = GraphNeuralNetworkLayer(concept_dim=D, num_layers=2)
        assert layer.concept_dim == D
        assert layer.num_layers == 2

    @pytest.mark.parametrize("bad", [
        {"concept_dim": 0},
        {"concept_dim": D, "num_layers": 0},
        {"concept_dim": D, "dropout_rate": 1.5},
        {"concept_dim": D, "num_attention_heads": 0},
        {"concept_dim": D, "message_passing": "bogus"},
        {"concept_dim": D, "aggregation": "bogus"},
    ])
    def test_invalid_args_raise(self, bad):
        with pytest.raises(ValueError):
            GraphNeuralNetworkLayer(**bad)

    @pytest.mark.parametrize("mp", ["gcn", "graphsage", "gat", "gin"])
    def test_forward_pass(self, graph_inputs, mp):
        nodes, adj = graph_inputs
        layer = GraphNeuralNetworkLayer(
            concept_dim=D, num_layers=2, message_passing=mp, aggregation="none",
            num_attention_heads=4,
        )
        out = layer((nodes, adj))
        assert tuple(out.shape) == (B, N, D)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    @pytest.mark.parametrize("agg,expected_nodes", [
        ("none", N), ("mean", 1), ("max", 1), ("sum", 1), ("attention", N),
    ])
    def test_compute_output_shape(self, agg, expected_nodes):
        layer = GraphNeuralNetworkLayer(
            concept_dim=D, num_layers=1, aggregation=agg, num_attention_heads=4,
        )
        shape = layer.compute_output_shape([(B, N, D), (B, N, N)])
        assert shape == (B, expected_nodes, D)

    def test_compute_output_shape_matches_call(self, graph_inputs):
        nodes, adj = graph_inputs
        layer = GraphNeuralNetworkLayer(
            concept_dim=D, num_layers=2, aggregation="mean", num_attention_heads=4,
        )
        out = layer((nodes, adj))
        computed = layer.compute_output_shape([nodes.shape, adj.shape])
        assert tuple(out.shape) == tuple(computed)

    def test_serialization_round_trip(self, graph_inputs, tmp_path):
        nodes, adj = graph_inputs
        node_in = keras.Input(shape=(N, D), name="nodes")
        adj_in = keras.Input(shape=(N, N), name="adj")
        out = GraphNeuralNetworkLayer(
            concept_dim=D, num_layers=2, message_passing="gcn",
            aggregation="none", num_attention_heads=4, name="gnn",
        )((node_in, adj_in))
        model = keras.Model([node_in, adj_in], out)
        y0 = model([nodes, adj])

        path = os.path.join(tmp_path, "gnn.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"GraphNeuralNetworkLayer": GraphNeuralNetworkLayer}
        )
        y1 = loaded([nodes, adj])
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0),
            keras.ops.convert_to_numpy(y1),
            rtol=1e-6, atol=1e-6,
        )

    def test_get_config_round_trip(self):
        layer = GraphNeuralNetworkLayer(concept_dim=D, num_layers=2, message_passing="gin")
        config = layer.get_config()
        rebuilt = GraphNeuralNetworkLayer.from_config(config)
        assert rebuilt.concept_dim == D
        assert rebuilt.message_passing == "gin"


# ---------------------------------------------------------------------
# The stack must run at a hidden width that differs from the input width.
#
# Every test above pins concept_dim == D == 16, so `node_shape` and the
# running block width coincide and the build-time shape contract is
# structurally invisible. These cases separate the two.
# ---------------------------------------------------------------------

MESSAGE_PASSING = ["gcn", "graphsage", "gat", "gin"]
NORMALIZATION = ["none", "layer", "rms", "batch"]
# 8 < D, 16 == D (the currently-passing arm), 32 > D.
HIDDEN_WIDTHS = [8, 16, 32]


class TestTheGnnStackRunsAtAHiddenWidth:
    """The layer must run for any ``concept_dim``, not only ``concept_dim == D``."""

    @pytest.mark.parametrize("norm", NORMALIZATION)
    @pytest.mark.parametrize("mp", MESSAGE_PASSING)
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("concept_dim", HIDDEN_WIDTHS)
    def test_forward_pass_at_any_hidden_width(
            self, graph_inputs, concept_dim, num_layers, mp, norm
    ):
        nodes, adj = graph_inputs
        layer = GraphNeuralNetworkLayer(
            concept_dim=concept_dim,
            num_layers=num_layers,
            message_passing=mp,
            normalization=norm,
            aggregation="none",
            num_attention_heads=4,
        )
        out = layer((nodes, adj))
        assert tuple(out.shape) == (B, N, concept_dim)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    @pytest.mark.parametrize("norm", NORMALIZATION)
    @pytest.mark.parametrize("mp", MESSAGE_PASSING)
    @pytest.mark.parametrize("concept_dim", HIDDEN_WIDTHS)
    def test_gradient_step_at_any_hidden_width(self, graph_inputs, concept_dim, mp, norm):
        nodes, adj = graph_inputs
        layer = GraphNeuralNetworkLayer(
            concept_dim=concept_dim,
            num_layers=2,
            message_passing=mp,
            normalization=norm,
            aggregation="none",
            num_attention_heads=4,
        )
        with tf.GradientTape() as tape:
            out = layer((nodes, adj), training=True)
            loss = keras.ops.mean(keras.ops.square(out))
        grads = tape.gradient(loss, layer.trainable_variables)
        assert len(layer.trainable_variables) > 0
        assert any(g is not None for g in grads)

    @pytest.mark.parametrize("agg,expected_nodes", [
        ("none", N), ("mean", 1), ("max", 1), ("sum", 1), ("attention", N),
    ])
    @pytest.mark.parametrize("mp", MESSAGE_PASSING)
    @pytest.mark.parametrize("concept_dim", HIDDEN_WIDTHS)
    def test_compute_output_shape_agrees_with_call_at_any_hidden_width(
            self, graph_inputs, concept_dim, mp, agg, expected_nodes
    ):
        nodes, adj = graph_inputs
        layer = GraphNeuralNetworkLayer(
            concept_dim=concept_dim,
            num_layers=2,
            message_passing=mp,
            aggregation=agg,
            num_attention_heads=4,
        )
        # UNBUILT: compute_output_shape must answer before any call.
        computed_unbuilt = layer.compute_output_shape([(B, N, D), (B, N, N)])
        assert tuple(computed_unbuilt) == (B, expected_nodes, concept_dim)

        out = layer((nodes, adj))
        assert tuple(out.shape) == tuple(computed_unbuilt)
        assert tuple(out.shape) == tuple(
            layer.compute_output_shape([nodes.shape, adj.shape])
        )


# ---------------------------------------------------------------------
# Composite initializer fan-out — plan-2026-09-07T183458-be1c267e step 3.
#
# One seedless `Initializer` INSTANCE handed to several child-layer
# constructors replays the same underlying sample at every later site, so two
# weights start life as the same random numbers. The per-site oracle below is
# the one `plan.md` § S-1 mandates: build the layer, then draw from the
# layer's own STILL-SHARED `self.kernel_initializer` at that weight's OWN
# shape, and assert the created weight is not bit-equal to that replay. A
# pairwise comparison between two sites is deliberately NOT used: cloning
# either member of a pair decorrelates it, so a one-line revert would stay
# green.
#
# Scope of the claim, exactly (copied from the corrected canonical wording in
# `src/dl_techniques/initializers/clone.py`'s module docstring): independence
# holds for a RANDOM SEEDLESS initializer. Three exemptions, all correct
# behaviour, none a defect --
#   1. a caller-supplied SEEDED instance (e.g. `GlorotUniform(seed=7)`)
#      replays deliberately and by contract, ACROSS DIFFERING SHAPES TOO;
#   2. a DETERMINISTIC initializer (`'zeros'`/`'ones'`/`Constant`, and
#      `Identity` only where the weight is 2-D -- it raises on rank 3+) holds
#      no random state, so every site is bit-identical and that is what it is
#      meant to do;
#   3. a CUSTOM initializer whose `get_config()`/`from_config()` round trip
#      raises falls back to `copy.deepcopy`, which copies the ALREADY-RESOLVED
#      seed rather than drawing a new one, so such a site silently stays tied.
# All three exemptions are asserted below as positive controls, so this module
# never states an absolute it has not measured. Exemption 3 applies to the
# CLONED sites -- `gcn_dense_{i}`, `sage_*`, and `gin_mlp_{i}` through
# `MLPBlock` -- and NOT to `gat_attention_{i}` / `aggregation_attention`,
# which never reach `clone_initializer`: see
# `TestACustomInitializerThatCannotSerialize`.
# ---------------------------------------------------------------------

from dl_techniques.layers.ffn.mlp import MLPBlock
from dl_techniques.initializers.clone import clone_initializer


def _np(x):
    return keras.ops.convert_to_numpy(x)


def _weight(layer, suffix):
    """Return the single weight whose path ends with ``suffix``."""
    hits = [w for w in layer.weights if w.path.endswith(suffix)]
    assert len(hits) == 1, f"expected exactly one weight ending {suffix!r}, got {[w.path for w in hits]}"
    return hits[0]


def _replay(initializer, shape):
    """Draw from the still-shared initializer instance at ``shape``."""
    return _np(initializer(tuple(shape), dtype="float32"))


def _assert_not_the_shared_replay(layer, suffix, initializer):
    w = _weight(layer, suffix)
    replay = _replay(initializer, w.shape)
    actual = _np(w)
    assert not np.array_equal(replay, actual), (
        f"{w.path} {tuple(w.shape)} is bit-identical to a fresh draw from the "
        f"layer's shared initializer instance -- the site was not cloned"
    )


def _built_gnn(message_passing, kernel_initializer, bias_initializer, graph_inputs,
               num_layers=2, aggregation="none"):
    nodes, adj = graph_inputs
    layer = GraphNeuralNetworkLayer(
        concept_dim=D,
        num_layers=num_layers,
        message_passing=message_passing,
        aggregation=aggregation,
        num_attention_heads=4,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )
    layer((nodes, adj))
    return layer


class TestTheGnnInitializerDoesNotFanOut:
    """One test per MEASURED-LIVE source site, so a one-line revert reddens one test.

    Six sites, three `keras.layers.Dense` constructions: ``gcn_dense_{i}``,
    ``sage_self_{i}`` and ``sage_neighbor_{i}``, kernel and bias each. The
    other three constructions in `__init__` (``gat_attention_{i}``,
    ``gin_mlp_{i}``, ``aggregation_attention``) are deliberately NOT cloned --
    see `TestTheCalleeReClonesTheSharedInitializer` below and the source
    anchors at those three sites.
    """

    @pytest.mark.parametrize("num_layers", [1, 2])
    def test_the_gcn_dense_kernel_is_not_the_shared_replay(self, graph_inputs, num_layers):
        ki = keras.initializers.GlorotUniform()
        layer = _built_gnn("gcn", ki, "zeros", graph_inputs, num_layers=num_layers)
        for i in range(num_layers):
            _assert_not_the_shared_replay(layer, f"gcn_dense_{i}/kernel", layer.kernel_initializer)

    @pytest.mark.parametrize("num_layers", [1, 2])
    def test_the_gcn_dense_bias_is_not_the_shared_replay(self, graph_inputs, num_layers):
        bi = keras.initializers.RandomNormal(stddev=0.05)
        layer = _built_gnn("gcn", "glorot_uniform", bi, graph_inputs, num_layers=num_layers)
        for i in range(num_layers):
            _assert_not_the_shared_replay(layer, f"gcn_dense_{i}/bias", layer.bias_initializer)

    @pytest.mark.parametrize("num_layers", [1, 2])
    def test_the_sage_self_kernel_is_not_the_shared_replay(self, graph_inputs, num_layers):
        ki = keras.initializers.GlorotUniform()
        layer = _built_gnn("graphsage", ki, "zeros", graph_inputs, num_layers=num_layers)
        for i in range(num_layers):
            _assert_not_the_shared_replay(layer, f"sage_self_{i}/kernel", layer.kernel_initializer)

    @pytest.mark.parametrize("num_layers", [1, 2])
    def test_the_sage_self_bias_is_not_the_shared_replay(self, graph_inputs, num_layers):
        bi = keras.initializers.RandomNormal(stddev=0.05)
        layer = _built_gnn("graphsage", "glorot_uniform", bi, graph_inputs, num_layers=num_layers)
        for i in range(num_layers):
            _assert_not_the_shared_replay(layer, f"sage_self_{i}/bias", layer.bias_initializer)

    @pytest.mark.parametrize("num_layers", [1, 2])
    def test_the_sage_neighbor_kernel_is_not_the_shared_replay(self, graph_inputs, num_layers):
        ki = keras.initializers.GlorotUniform()
        layer = _built_gnn("graphsage", ki, "zeros", graph_inputs, num_layers=num_layers)
        for i in range(num_layers):
            _assert_not_the_shared_replay(layer, f"sage_neighbor_{i}/kernel", layer.kernel_initializer)

    @pytest.mark.parametrize("num_layers", [1, 2])
    def test_the_sage_neighbor_bias_is_not_the_shared_replay(self, graph_inputs, num_layers):
        bi = keras.initializers.RandomNormal(stddev=0.05)
        layer = _built_gnn("graphsage", "glorot_uniform", bi, graph_inputs, num_layers=num_layers)
        for i in range(num_layers):
            _assert_not_the_shared_replay(layer, f"sage_neighbor_{i}/bias", layer.bias_initializer)

    def test_the_sage_self_and_neighbor_kernels_differ_from_each_other(self, graph_inputs):
        """The architecturally sharpest pair: self-transform vs neighbour-transform."""
        ki = keras.initializers.GlorotUniform()
        layer = _built_gnn("graphsage", ki, "zeros", graph_inputs)
        for i in range(2):
            a = _np(_weight(layer, f"sage_self_{i}/kernel"))
            b = _np(_weight(layer, f"sage_neighbor_{i}/kernel"))
            assert not np.array_equal(a, b)


class TestTheExemptionsThisModuleDoesNotClaimAway:
    """Positive controls for exemptions 1 and 2 of `clone.py` § "Scope of the claim, exactly".

    Cloning must NOT break either. These assert that identical weights are the
    CORRECT outcome in both cases, so no reader mistakes the guards above for
    an absolute "initializers can never coincide" claim.
    """

    def test_a_deterministic_zeros_bias_is_identical_at_every_site_and_that_is_correct(
            self, graph_inputs
    ):
        # Exemption 2. `'zeros'` is the class default and holds no random
        # state: every bias is the same because that is what `'zeros'` means.
        layer = _built_gnn("graphsage", "glorot_uniform", "zeros", graph_inputs)
        biases = [w for w in layer.weights if w.path.endswith("/bias")]
        assert len(biases) == 4
        for b in biases:
            assert float(np.max(np.abs(_np(b)))) == 0.0

    def test_a_seeded_initializer_stays_reproducible_across_two_instances(self, graph_inputs):
        # Exemption 1 + invariant I-2: cloning reproduces an explicit seed by
        # contract, so two separately-constructed layers still agree exactly.
        paths_and_values = []
        for _ in range(2):
            layer = _built_gnn(
                "graphsage",
                keras.initializers.GlorotUniform(seed=7),
                keras.initializers.RandomNormal(stddev=0.05, seed=11),
                graph_inputs,
            )
            paths_and_values.append(
                {w.path.split("/", 1)[1]: _np(w) for w in layer.weights}
            )
        a, b = paths_and_values
        assert set(a) == set(b)
        for key in a:
            assert np.array_equal(a[key], b[key]), f"{key} not reproducible under an explicit seed"


class TestTheCalleeReClonesTheSharedInitializer:
    """MONITOR the stock-callee behaviour that steps 3's narrow scope depends on.

    These tests are GREEN today AND were green before the fix. That is
    deliberate and is the whole point: they are not guards for a change we
    made, they are a tripwire on somebody else's code.

    `graph_neural_network.py` deliberately does NOT wrap `clone_initializer`
    around the initializers it hands to ``gat_attention_{i}``,
    ``aggregation_attention`` (both stock `keras.layers.MultiHeadAttention`)
    or ``gin_mlp_{i}`` (`MLPBlock`), because those callees ALREADY re-clone
    per sub-layer -- measured, not assumed. If a future Keras or a future
    `MLPBlock` drops that re-clone, those three sites silently start aliasing
    again and no other test in this repo would notice. These two tests go RED
    and name the reason.

    DO NOT DELETE THESE AS VACUOUS. A test that cannot fail today is exactly
    what an upstream-behaviour tripwire looks like. See decisions.md D-007.
    """

    def test_stock_multi_head_attention_re_clones_per_sublayer(self):
        # `MultiHeadAttention._get_common_kwargs_for_sublayer` runs
        # `initializer.__class__.from_config(initializer.get_config())` for
        # every sub-layer. `GlorotUniform().get_config()` reports
        # `{'seed': None}` even when the live instance has a RESOLVED `.seed`,
        # so `from_config` self-assigns a fresh seed and the tie breaks. The
        # untying depends on the round trip SUCCEEDING. `clone.py` exemption 3
        # -- a raising `get_config()` falls back to `copy.deepcopy` and the
        # site stays tied -- describes `clone_initializer`, which is NOT on
        # this code path, so do not paste it in here: at THIS site a raising
        # `get_config()` propagates and the layer fails to build.
        # `TestACustomInitializerThatCannotSerialize` below measures that.
        ki = keras.initializers.GlorotUniform()
        mha = keras.layers.MultiHeadAttention(num_heads=2, key_dim=4, kernel_initializer=ki)
        x = np.zeros((1, 3, 8), dtype="float32")
        mha(x, x)

        q = _weight(mha, "query/kernel")
        k = _weight(mha, "key/kernel")
        assert tuple(q.shape) == tuple(k.shape)
        assert not np.array_equal(_np(q), _np(k)), (
            "stock MultiHeadAttention no longer re-clones its kernel_initializer "
            "per sub-layer -- graph_neural_network.py's gat_attention_{i} and "
            "aggregation_attention sites must now clone at the site (see D-007)"
        )
        assert not np.array_equal(_replay(ki, q.shape), _np(q))

    def test_the_shared_seedless_instance_really_does_replay(self):
        # The control the test above rests on: without a re-cloning callee,
        # one shared seedless instance DOES hand two Dense layers the same
        # numbers. If this ever goes green-by-vacuity the oracle is broken.
        ki = keras.initializers.GlorotUniform()
        a = keras.layers.Dense(4, kernel_initializer=ki, name="a")
        b = keras.layers.Dense(4, kernel_initializer=ki, name="b")
        x = np.zeros((1, 6), dtype="float32")
        a(x)
        b(x)
        assert np.array_equal(_np(a.kernel), _np(b.kernel))

    def test_mlp_block_clones_its_own_kernel_initializer(self):
        # `layers/ffn/mlp.py` calls `clone_initializer` for fc1 and fc2, which
        # is why `gin_mlp_{i}` is not cloned at the GNN site.
        ki = keras.initializers.GlorotUniform()
        bi = keras.initializers.RandomNormal(stddev=0.05)
        block = MLPBlock(hidden_dim=8, output_dim=4, kernel_initializer=ki, bias_initializer=bi)
        block(np.zeros((1, 3, 4), dtype="float32"))

        for suffix, init in (("fc1/kernel", ki), ("fc2/kernel", ki),
                             ("fc1/bias", bi), ("fc2/bias", bi)):
            w = _weight(block, suffix)
            assert not np.array_equal(_replay(init, w.shape), _np(w)), (
                f"MLPBlock no longer clones its initializer for {suffix} -- "
                "graph_neural_network.py's gin_mlp_{i} site must now clone (see D-007)"
            )


@keras.saving.register_keras_serializable(package="test_gnn_anchor_probe")
class _RefusesToSerialize(keras.initializers.GlorotUniform):
    """A CUSTOM initializer whose ``get_config()`` round trip RAISES.

    The exact shape `clone.py` § "Scope of the claim, exactly" exemption 3
    is about. Defined at module scope and registered so it is picklable and
    serializable-by-name everywhere except the one method that raises.
    """

    def get_config(self):
        raise RuntimeError("this initializer refuses to serialize")


class TestACustomInitializerThatCannotSerialize:
    """Pin what ACTUALLY happens at the three deliberately non-cloned sites.

    plan-2026-09-07T183458-be1c267e step 8.1 / D-019. The D-007 anchor used to
    carry a copy-paste of `clone.py`'s three-exemption paragraph, whose third
    clause says a custom initializer failing the `get_config()` round trip
    "falls back to `copy.deepcopy`, keeping the resolved seed". That is true of
    `clone_initializer`. It is FALSE at `gat_attention_{i}` and
    `aggregation_attention`, which do not call `clone_initializer` at all --
    stock `MultiHeadAttention._get_common_kwargs_for_sublayer` lets the
    exception out and the layer never finishes building.

    SC-10 checked that the three exemptions were NAMED, not that the named
    mechanism EXISTS at the site, so correct prose pasted into an inapplicable
    context passed the gate. These tests are what makes the corrected sentence
    MONITORED instead of merely asserted.

    HOW THEY GO RED, measured, because half of this class is a TRIPWIRE on
    somebody else's code and no revert inside this repo can redden that half --
    exactly like `TestTheCalleeReClonesTheSharedInitializer` above, and for the
    same reason. Do not delete either half as vacuous:

    * the three `clone_initializer` arms (`..._builds_fine_and_stays_tied`,
      `..._deepcopy_fallback_really_does_keep_the_tie`,
      `..._two_gcn_kernels_..._bit_identical`) redden on a REPO-SOURCE
      mutation: delete the `except Exception: return copy.deepcopy(resolved)`
      fallback in `initializers/clone.py`. MEASURED: 5 failed / 330 passed.
    * the five stock-`MultiHeadAttention` arms redden only on a CALLEE
      mutation: give `MultiHeadAttention._get_common_kwargs_for_sublayer` the
      deepcopy fallback that `clone_initializer` has -- i.e. construct the
      exact world the deleted sentence described. MEASURED in-process:
      5 failed / 5 passed, all five failures in this class.

    Wrapping `clone_initializer(...)` around the two `MultiHeadAttention`
    sites does NOT redden anything here, and that is itself the point: the
    deepcopy of a `get_config()`-raising initializer still raises, so the
    "fix" the D-007 anchor forbids would not even change this behaviour.
    """

    _NODES = np.zeros((B, N, D), dtype="float32")
    _ADJ = np.ones((B, N, N), dtype="float32")

    def _run(self, **kwargs):
        layer = GraphNeuralNetworkLayer(
            concept_dim=D,
            num_layers=1,
            kernel_initializer=_RefusesToSerialize(),
            bias_initializer="zeros",
            **kwargs,
        )
        return layer, layer((self._NODES, self._ADJ))

    @pytest.mark.parametrize("kwargs", [
        {"message_passing": "gat", "aggregation": "none"},
        {"message_passing": "gat", "aggregation": "attention"},
        {"message_passing": "gcn", "aggregation": "attention"},
        {"message_passing": "gin", "aggregation": "attention"},
    ])
    def test_a_stock_multi_head_attention_site_fails_loudly_rather_than_staying_tied(
            self, kwargs
    ):
        """Whenever a stock `MultiHeadAttention` is in the sub-layer set."""
        with pytest.raises(RuntimeError, match="refuses to serialize"):
            self._run(**kwargs)

    def test_construction_itself_still_succeeds_the_raise_comes_from_build(self):
        """Name the boundary precisely: `__init__` is fine, `build()` is not.

        The anchor says "`__init__` succeeds, then `build()` raises". If a
        future Keras moved the round trip into `MultiHeadAttention.__init__`
        the anchor would be wrong in a way the parametrized test above cannot
        see, because it wraps both calls.
        """
        layer = GraphNeuralNetworkLayer(
            concept_dim=D, num_layers=1, message_passing="gat", aggregation="none",
            kernel_initializer=_RefusesToSerialize(), bias_initializer="zeros",
        )
        assert not layer.built
        with pytest.raises(RuntimeError, match="refuses to serialize"):
            layer.build(((B, N, D), (B, N, N)))

    @pytest.mark.parametrize("message_passing", ["gcn", "graphsage", "gin"])
    def test_a_cloned_or_mlp_block_site_builds_fine_and_stays_tied(self, message_passing):
        """The other half, and the reason the two must not share one sentence.

        `gcn_dense_{i}` / `sage_*` go through `clone_initializer` and
        `gin_mlp_{i}` goes through `MLPBlock`, which also calls it -- so those
        sites really DO take the `copy.deepcopy` branch, really DO build, and
        really DO stay tied. Exemption 3 applies here and only here.
        """
        layer, out = self._run(message_passing=message_passing, aggregation="none")
        assert layer.built
        assert tuple(out.shape) == (B, N, D)

    def test_the_deepcopy_fallback_really_does_keep_the_tie(self):
        """The control the previous test rests on.

        `clone_initializer` on this initializer returns a distinct object that
        replays the SAME numbers, which is what "keeps the resolved seed"
        means. Without this the test above would be green by vacuity.
        """
        shared = _RefusesToSerialize()
        first = clone_initializer(shared)
        second = clone_initializer(shared)
        assert first is not shared and second is not shared
        assert np.array_equal(
            _np(first((4, 4), dtype="float32")), _np(second((4, 4), dtype="float32"))
        )

    def test_two_gcn_kernels_under_that_initializer_are_bit_identical(self):
        """Exemption 3, end to end, on the real layer.

        Two `gcn_dense_{i}` kernels come out identical even though the site IS
        cloned -- because the clone is a deepcopy of a resolved seed. This is
        the correct outcome, and it is why the fan-out guards elsewhere in this
        module use a seedless `GlorotUniform` and not this class.
        """
        layer = GraphNeuralNetworkLayer(
            concept_dim=D, num_layers=2, message_passing="gcn", aggregation="none",
            kernel_initializer=_RefusesToSerialize(), bias_initializer="zeros",
        )
        layer((self._NODES, self._ADJ))
        a = _np(_weight(layer, "gcn_dense_0/kernel"))
        b = _np(_weight(layer, "gcn_dense_1/kernel"))
        assert np.array_equal(a, b)


class TestTheSecondaryContractsThisLayerAdvertises:
    """The four measured secondary defects fixed in plan step 8.

    Each was MEASURED on the shipped layer before the fix, at `(2, 5, 16)`
    nodes and an all-ones `(2, 5, 5)` adjacency:

    1. `num_attention_heads` is DOCUMENTED as "Must divide ``concept_dim``"
       (`graph_neural_network.py` `:param num_attention_heads:`) and was never
       validated -- `concept_dim=16, num_attention_heads=3` constructed and ran
       with a silently floor-divided `key_dim=5`, so the GAT branch used
       `3 * 5 = 15` of the 16 advertised channels.
    2. The final aggregation attention hardcoded `num_heads=4` /
       `key_dim=concept_dim // 4`, ignoring `num_attention_heads` entirely --
       measured `aggregation_attention.num_heads == 4` at
       `num_attention_heads=8`.
    3. Adjacency shape was unvalidated: a `(1, 5, 5)` adjacency against
       `(2, 5, 16)` nodes was silently BROADCAST and returned a plausible
       `(2, 5, 16)`, and a non-square `(2, 5, 7)` died inside `call()` with a
       raw `InvalidArgumentError: Incompatible shapes: [2,5,7] vs. [2,1,5]`
       that never named the adjacency argument.
    4. The `'layer'` and `'batch'` normalization branches took Keras' stock
       `epsilon=1e-3` while the sibling `'rms'` branch three lines below ran at
       `1e-6` -- a 1000x difference in every denominator, in the class DEFAULT
       configuration, with no shape symptom (D-004).
    """

    # -- 1. num_attention_heads divisibility ---------------------------------

    @pytest.mark.parametrize("heads", [3, 5, 7])
    def test_num_attention_heads_must_divide_concept_dim(self, heads):
        with pytest.raises(ValueError) as excinfo:
            GraphNeuralNetworkLayer(concept_dim=D, num_attention_heads=heads)
        message = str(excinfo.value)
        assert "num_attention_heads" in message
        assert "concept_dim" in message
        assert str(heads) in message
        assert str(D) in message

    @pytest.mark.parametrize("heads", [1, 2, 4, 8, 16])
    def test_a_divisor_head_count_still_constructs(self, heads):
        layer = GraphNeuralNetworkLayer(concept_dim=D, num_attention_heads=heads)
        assert layer.num_attention_heads == heads

    def test_the_divisibility_raise_fires_for_every_message_passing_mode(self):
        """The contract is on the pair, not on whether GAT happens to be on.

        `num_attention_heads` also sizes the aggregation attention, which is
        the DEFAULT aggregation, so a non-divisor is wrong at `'gcn'` too.
        """
        for mode in ("gcn", "graphsage", "gat", "gin"):
            with pytest.raises(ValueError, match="num_attention_heads"):
                GraphNeuralNetworkLayer(
                    concept_dim=D, message_passing=mode, num_attention_heads=3
                )

    # -- 2. the aggregation attention honours num_attention_heads ------------

    @pytest.mark.parametrize("heads", [1, 2, 8, 16])
    def test_the_aggregation_attention_uses_the_configured_head_count(self, heads):
        layer = GraphNeuralNetworkLayer(
            concept_dim=D, num_attention_heads=heads, aggregation="attention"
        )
        assert layer.aggregation_attention.num_heads == heads
        assert layer.aggregation_attention.key_dim == D // heads

    def test_the_default_head_count_is_unchanged(self):
        """At the default `num_attention_heads=4` the swap is bit-identical.

        Pinned so a future edit cannot quietly move the default config's
        weight shapes while claiming to be wiring a knob through.
        """
        layer = GraphNeuralNetworkLayer(concept_dim=D, aggregation="attention")
        assert layer.aggregation_attention.num_heads == 4
        assert layer.aggregation_attention.key_dim == D // 4

    def test_the_aggregation_attention_actually_runs_at_a_non_default_head_count(
        self, graph_inputs
    ):
        """A shape assertion alone would pass on a layer that cannot run."""
        nodes, adj = graph_inputs
        layer = GraphNeuralNetworkLayer(
            concept_dim=D, num_layers=1, num_attention_heads=8, aggregation="attention"
        )
        out = layer((nodes, adj))
        assert tuple(out.shape) == (B, N, D)

    # -- 3. adjacency shape validation ---------------------------------------

    def test_a_broadcastable_adjacency_batch_is_refused(self, graph_inputs):
        """The measured silent case: `(1, N, N)` against `(2, N, D)` nodes."""
        nodes, _ = graph_inputs
        layer = GraphNeuralNetworkLayer(concept_dim=D, num_layers=1, aggregation="none")
        with pytest.raises(ValueError) as excinfo:
            layer((nodes, np.ones((1, N, N), dtype="float32")))
        message = str(excinfo.value)
        assert "adjacency" in message
        assert "(1, 5, 5)" in message or "(1, 5, 5)".replace(" ", "") in message

    def test_a_non_square_adjacency_is_refused_by_name(self, graph_inputs):
        nodes, _ = graph_inputs
        layer = GraphNeuralNetworkLayer(concept_dim=D, num_layers=1, aggregation="none")
        with pytest.raises(ValueError) as excinfo:
            layer((nodes, np.ones((B, N, N + 2), dtype="float32")))
        assert "adjacency" in str(excinfo.value)

    def test_an_adjacency_of_the_wrong_rank_is_refused(self, graph_inputs):
        nodes, _ = graph_inputs
        layer = GraphNeuralNetworkLayer(concept_dim=D, num_layers=1, aggregation="none")
        with pytest.raises(ValueError, match="adjacency"):
            layer((nodes, np.ones((N, N), dtype="float32")))

    def test_an_adjacency_whose_node_count_disagrees_with_the_nodes_is_refused(
        self, graph_inputs
    ):
        nodes, _ = graph_inputs
        layer = GraphNeuralNetworkLayer(concept_dim=D, num_layers=1, aggregation="none")
        with pytest.raises(ValueError, match="adjacency"):
            layer((nodes, np.ones((B, N + 1, N + 1), dtype="float32")))

    @pytest.mark.parametrize("adjacency_shape", [
        (None, N, N),
        (B, None, None),
        (None, None, None),
    ])
    def test_a_symbolic_adjacency_axis_does_NOT_raise(self, adjacency_shape):
        """None-safety, the way `layers/tabular/nlinear.py` does it.

        An unknown (symbolic) axis carries no disagreement to detect, so the
        guard must skip it and check only the RANK. Without this arm the raise
        would break every functional-API caller that builds on
        `keras.Input(shape=(None, None))`.
        """
        layer = GraphNeuralNetworkLayer(concept_dim=D, num_layers=1, aggregation="none")
        layer.build(((None, N, D), adjacency_shape))
        assert layer.built

    def test_the_matching_adjacency_still_builds_and_runs(self, graph_inputs):
        """Positive control: the guard must not reject the valid case."""
        nodes, adj = graph_inputs
        layer = GraphNeuralNetworkLayer(concept_dim=D, num_layers=2, aggregation="none")
        out = layer((nodes, adj))
        assert tuple(out.shape) == (B, N, D)

    def test_a_functional_model_with_symbolic_node_and_edge_counts_still_builds(self):
        node_input = keras.Input(shape=(None, D))
        adjacency_input = keras.Input(shape=(None, None))
        out = GraphNeuralNetworkLayer(
            concept_dim=D, num_layers=1, aggregation="none"
        )((node_input, adjacency_input))
        model = keras.Model([node_input, adjacency_input], out)
        rng = np.random.default_rng(3)
        nodes = rng.standard_normal((B, N, D)).astype("float32")
        adj = np.ones((B, N, N), dtype="float32")
        assert tuple(model([nodes, adj]).shape) == (B, N, D)

    # -- 4. normalization epsilon --------------------------------------------

    @pytest.mark.parametrize("normalization", ["layer", "batch", "rms"])
    def test_every_normalization_branch_runs_at_the_house_epsilon(self, normalization):
        """D-004: all three branches at 1e-6, matching `norms/factory.py`.

        `layers/CLAUDE.md` § Layer Reuse Policy rule 5: Keras' stock `1e-3` is
        1000x the factory's `1e-6` with no shape symptom and no warning. There
        is no cited reference for `1e-3` here -- it was the stock default taken
        by accident, and this file's own `'rms'` branch already disagreed with
        it.
        """
        layer = GraphNeuralNetworkLayer(
            concept_dim=D, num_layers=1, normalization=normalization
        )
        assert layer.norm_layers[0].epsilon == pytest.approx(1e-6, rel=0, abs=0)

    def test_the_none_branch_still_creates_no_normalization_layer(self):
        layer = GraphNeuralNetworkLayer(
            concept_dim=D, num_layers=2, normalization="none"
        )
        assert layer.norm_layers == [None, None]

    @pytest.mark.parametrize("normalization", ["layer", "batch", "rms"])
    def test_the_normalization_layer_keeps_its_explicit_name(self, normalization):
        """Routing through the factory must not drop `name=`.

        A dropped `name=` would fall back to Keras' process-global
        auto-increment counter and make weight paths depend on import order --
        the `_SEWeights` defect (D-011) in a different file.
        """
        layer = GraphNeuralNetworkLayer(
            concept_dim=D, num_layers=2, normalization=normalization
        )
        expected = {"layer": "layer_norm", "batch": "batch_norm", "rms": "rms_norm"}[
            normalization
        ]
        for index in range(2):
            assert layer.norm_layers[index].name == f"gnn_{expected}_{index}"
