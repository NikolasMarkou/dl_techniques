"""Tests for the TabM building blocks (efficient tabular ensembles).

These five layers each live in their own module under
``src/dl_techniques/layers/tabular/``: ``scale_ensemble.py``,
``linear_efficient_ensemble.py``, ``nlinear.py``, ``tabm_mlp_block.py`` and
``tabm_backbone.py``. They shared a single module until
``plan-2026-09-07-b821967f`` split it one class per file; this file keeps its
historical name (D-006) and covers all five modules together, because the
layers compose into one stack and the round-trip guards exercise them jointly.
"""

import json
import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.activations.golu import GoLU

from dl_techniques.layers.tabular.scale_ensemble import ScaleEnsemble
from dl_techniques.layers.tabular.linear_efficient_ensemble import (
    LinearEfficientEnsemble,
)
from dl_techniques.layers.tabular.nlinear import NLinear
from dl_techniques.layers.tabular.tabm_mlp_block import TabMMLPBlock
from dl_techniques.layers.tabular.tabm_backbone import TabMBackbone

B, K, D = 2, 3, 6


def _f32(*shape):
    return np.random.default_rng(0).standard_normal(shape).astype("float32")


def _roundtrip(layer, input_shape, data, name, tmp_path, cls):
    inp = keras.Input(shape=input_shape)
    out = layer(inp)
    model = keras.Model(inp, out)
    y0 = model(data, training=False)
    path = os.path.join(tmp_path, f"{name}.keras")
    model.save(path)
    loaded = keras.models.load_model(path, custom_objects={cls.__name__: cls})
    y1 = loaded(data, training=False)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
        rtol=1e-5, atol=1e-5,
    )


class TestScaleEnsemble:
    def test_forward_and_shape(self):
        layer = ScaleEnsemble(k=K, input_dim=D)
        out = layer(_f32(B, K, D))
        assert tuple(out.shape) == (B, K, D)
        assert layer.compute_output_shape((B, K, D)) == (B, K, D)

    def test_serialization(self, tmp_path):
        _roundtrip(ScaleEnsemble(k=K, input_dim=D, name="se"), (K, D), _f32(B, K, D),
                   "se", tmp_path, ScaleEnsemble)

    @pytest.mark.parametrize("kwargs, bad", [
        (dict(k=0, input_dim=D), "0"),
        (dict(k=-1, input_dim=D), "-1"),
        (dict(k=K, input_dim=0), "0"),
        (dict(k=K, input_dim=-2), "-2"),
    ])
    def test_constructor_rejects_non_positive(self, kwargs, bad):
        with pytest.raises(ValueError) as exc:
            ScaleEnsemble(**kwargs)
        # The message must name the offending value, not just the argument.
        assert bad in str(exc.value)

    def test_constructor_accepts_valid_values(self):
        # Positive control: the guards must not fire on the shipped configuration.
        layer = ScaleEnsemble(k=K, input_dim=D)
        assert (layer.k, layer.input_dim) == (K, D)

    def test_build_rejects_k_mismatch(self):
        # `k` is a REQUIRED constructor argument that names axis 1, but build()
        # used to read nothing at all from `input_shape`.
        layer = ScaleEnsemble(k=K, input_dim=D)
        with pytest.raises(ValueError) as exc:
            layer.build((None, K + 4, D))
        msg = str(exc.value)
        assert str(K) in msg and str(K + 4) in msg
        assert str(tuple((None, K + 4, D))) in msg

    def test_build_rejects_input_dim_mismatch(self):
        # The measured reproduction: `input_dim=1` against a (B, K, D) input was
        # accepted and then absorbed by the broadcast in call(), which returned a
        # (B, K, D) tensor scaled by the WRONG (K, 1) weight and no error at all.
        layer = ScaleEnsemble(k=K, input_dim=1)
        with pytest.raises(ValueError) as exc:
            layer.build((B, K, D))
        msg = str(exc.value)
        assert "1" in msg and str(D) in msg
        assert str(tuple((B, K, D))) in msg

    def test_build_accepts_matching_shape(self):
        # Positive control: the guards must not reject the contract shape.
        layer = ScaleEnsemble(k=K, input_dim=D)
        layer.build((B, K, D))
        assert tuple(layer.weight.shape) == (K, D)

    @pytest.mark.parametrize("shape", [(None, None, D), (None, K, None)])
    def test_build_accepts_unknown_axes(self, shape):
        # The `is not None` sub-condition on both guards exists for this: a
        # symbolic/unknown axis carries no information and must not raise.
        # Sub-layers here are built via a direct `.build(input_shape)` call, not
        # always through a fully concrete functional trace.
        layer = ScaleEnsemble(k=K, input_dim=D)
        layer.build(shape)
        assert tuple(layer.weight.shape) == (K, D)


class TestLinearEfficientEnsemble:
    def test_forward_and_shape(self):
        layer = LinearEfficientEnsemble(units=5, k=K)
        out = layer(_f32(B, K, D))
        assert tuple(out.shape) == (B, K, 5)

    def test_serialization(self, tmp_path):
        _roundtrip(LinearEfficientEnsemble(units=5, k=K, name="lee"), (K, D), _f32(B, K, D),
                   "lee", tmp_path, LinearEfficientEnsemble)

    @pytest.mark.parametrize("kwargs, bad", [
        (dict(units=0, k=K), "0"),
        (dict(units=-4, k=K), "-4"),
        (dict(units=5, k=0), "0"),
        (dict(units=5, k=-1), "-1"),
    ])
    def test_constructor_rejects_non_positive(self, kwargs, bad):
        with pytest.raises(ValueError) as exc:
            LinearEfficientEnsemble(**kwargs)
        # The message must name the offending value, not just the argument.
        assert bad in str(exc.value)

    def test_constructor_accepts_valid_values(self):
        # Positive control: the guards must not fire on the shipped configuration.
        layer = LinearEfficientEnsemble(units=5, k=K)
        assert (layer.units, layer.k) == (5, K)

    def test_all_gates_off_forward_pass(self):
        # `r` / `s` / `bias` are created in build() only when their flag is set,
        # and read in call() behind the identical flag. With all three off the
        # layer must still run: an asymmetry between the two sides would surface
        # here as an AttributeError, and this arm is otherwise unexercised.
        layer = LinearEfficientEnsemble(
            units=5, k=K, use_bias=False,
            ensemble_scaling_in=False, ensemble_scaling_out=False,
        )
        out = layer(_f32(B, K, D))
        assert tuple(out.shape) == (B, K, 5)
        assert not hasattr(layer, "r") and not hasattr(layer, "s")
        assert not hasattr(layer, "bias")

    @pytest.mark.parametrize("gate", ["ensemble_scaling_in", "ensemble_scaling_out", "use_bias"])
    def test_each_gate_off_forward_pass(self, gate):
        # Same lockstep contract, one flag at a time.
        layer = LinearEfficientEnsemble(units=5, k=K, **{gate: False})
        assert tuple(layer(_f32(B, K, D)).shape) == (B, K, 5)

    @pytest.mark.parametrize("dist", ["random-signs", "normal"])
    def test_r_and_s_differ_at_equal_widths(self, dist):
        # `r` (pre-matmul, shape (k, input_dim)) and `s` (post-matmul, shape
        # (k, units)) are architecturally distinct roles, so a stochastic
        # `init_distribution` must draw them INDEPENDENTLY. A single seedless
        # Keras initializer instance self-assigns a seed and replays it, so
        # sharing one instance across both `add_weight` calls produced
        # bit-identical vectors at every shape where `input_dim == units`.
        # The widths here are deliberately EQUAL: at unequal widths the shapes
        # differ and the vectors trivially differ, which is the bug's hiding
        # place, so such a test would be green against the defect.
        layer = LinearEfficientEnsemble(units=8, k=K, init_distribution=dist)
        layer.build((None, K, 8))
        r = keras.ops.convert_to_numpy(layer.r)
        s = keras.ops.convert_to_numpy(layer.s)
        # Anti-vacuity: assert the shapes MATCH first. Equality must be
        # possible for the inequality below to carry any information.
        assert r.shape == s.shape == (K, 8)
        assert not np.array_equal(r, s)

    def test_ones_distribution_keeps_r_and_s_identical(self):
        # Positive control, not a diversity arm: with `init_distribution='ones'`
        # both vectors are all-ones BY DESIGN, so equality is the correct
        # assertion here. Cloning a deterministic `Ones()` still yields ones,
        # so the fix must not perturb this branch.
        layer = LinearEfficientEnsemble(units=8, k=K, init_distribution="ones")
        layer.build((None, K, 8))
        r = keras.ops.convert_to_numpy(layer.r)
        s = keras.ops.convert_to_numpy(layer.s)
        assert r.shape == s.shape == (K, 8)
        assert np.array_equal(r, s)
        assert np.array_equal(r, np.ones_like(r))

    def test_build_rejects_k_mismatch(self):
        # build() only ever read `input_shape[-1]` (the kernel fan-in); an axis-1
        # mismatch was accepted here and surfaced later as an opaque backend
        # InvalidArgumentError from the `x * expand_dims(r, 0)` multiply in call().
        layer = LinearEfficientEnsemble(units=5, k=K)
        with pytest.raises(ValueError) as exc:
            layer.build((B, K + 4, D))
        msg = str(exc.value)
        assert str(K) in msg and str(K + 4) in msg
        assert str(tuple((B, K + 4, D))) in msg

    def test_build_accepts_matching_k(self):
        # Positive control: the contract shape still builds, weights unchanged.
        layer = LinearEfficientEnsemble(units=5, k=K)
        layer.build((B, K, D))
        assert tuple(layer.kernel.shape) == (D, 5)
        assert tuple(layer.r.shape) == (K, D)

    def test_build_accepts_unknown_k_axis(self):
        # `is not None` sub-condition (H-2): an unknown axis 1 must build, and
        # the weights are still sized from the STORED `k`, not from the input.
        layer = LinearEfficientEnsemble(units=5, k=K)
        layer.build((None, None, D))
        assert tuple(layer.r.shape) == (K, D)

    def test_compute_output_shape_derives_from_k(self):
        # The input here is DELIBERATELY inconsistent -- axis 1 is K+4, not k --
        # because that is the only way to see which SOURCE the answer came from.
        # `r`/`s`/`bias` are all shaped from the stored `k`, so `k` is what
        # call() actually produces; reading `input_shape[1]` made this method
        # disagree with call() and with TabMMLPBlock on the same nominal input
        # ((None, 7, 5) vs (None, 3, 5)). Guide v2 3.4 requires the answer to
        # come from stored config on an UNBUILT layer, which is why nothing is
        # built here. build() would now REJECT this shape outright
        # (test_build_rejects_k_mismatch), so no layer that could be built can
        # ever produce a (_, K+4, 5) tensor.
        layer = LinearEfficientEnsemble(units=5, k=K)
        assert not layer.built
        assert layer.compute_output_shape((None, K + 4, D)) == (None, K, 5)
        # ... and it now agrees with the block that wraps it, on that same input.
        assert (
            layer.compute_output_shape((None, K + 4, D))
            == TabMMLPBlock(units=5, k=K).compute_output_shape((None, K + 4, D))
        )


class TestNLinear:
    def test_forward_and_shape(self):
        layer = NLinear(n=K, input_dim=D, output_dim=5)
        out = layer(_f32(B, K, D))
        assert tuple(out.shape) == (B, K, 5)
        assert layer.compute_output_shape((B, K, D)) == (B, K, 5)

    def test_serialization(self, tmp_path):
        _roundtrip(NLinear(n=K, input_dim=D, output_dim=5, name="nl"), (K, D), _f32(B, K, D),
                   "nl", tmp_path, NLinear)

    @pytest.mark.parametrize("kwargs, bad", [
        (dict(n=0, input_dim=D, output_dim=5), "0"),
        (dict(n=-1, input_dim=D, output_dim=5), "-1"),
        (dict(n=K, input_dim=D, output_dim=0), "0"),
        (dict(n=K, input_dim=D, output_dim=-3), "-3"),
        (dict(n=K, input_dim=0, output_dim=5), "0"),
        (dict(n=K, input_dim=-2, output_dim=5), "-2"),
    ])
    def test_constructor_rejects_non_positive(self, kwargs, bad):
        with pytest.raises(ValueError) as exc:
            NLinear(**kwargs)
        # The message must name the offending value, not just the argument.
        assert bad in str(exc.value)

    def test_constructor_accepts_valid_values(self):
        # Positive control: the guards must not fire on the shipped configuration.
        assert NLinear(n=K, input_dim=D, output_dim=5).n == K
        assert NLinear(n=1, input_dim=None, output_dim=1).input_dim is None

    def test_build_rejects_input_dim_mismatch(self):
        layer = NLinear(n=K, input_dim=D, output_dim=5)
        with pytest.raises(ValueError) as exc:
            layer.build((None, K, D + 4))
        msg = str(exc.value)
        assert str(D) in msg and str(D + 4) in msg

    def test_build_rejects_n_mismatch(self):
        # `n` names axis 1 and sizes `kernels`, but build() only ever validated
        # the LAST axis: an axis-1 mismatch was accepted here,
        # compute_output_shape still answered (None, n, output_dim), and the
        # failure surfaced only as an opaque backend error from the einsum in
        # call(). The guard sits ABOVE the input_dim block, in axis order.
        layer = NLinear(n=K, input_dim=D, output_dim=5)
        with pytest.raises(ValueError) as exc:
            layer.build((None, K + 4, D))
        msg = str(exc.value)
        assert str(K) in msg and str(K + 4) in msg
        assert str(tuple((None, K + 4, D))) in msg

    def test_build_accepts_matching_input_dim(self):
        # Positive control for the shape contract.
        layer = NLinear(n=K, input_dim=D, output_dim=5)
        layer.build((None, K, D))
        assert tuple(layer.kernels.shape) == (K, D, 5)

    def test_build_accepts_unknown_n_axis(self):
        # `is not None` sub-condition (H-2): an unknown axis 1 carries no
        # information and must build, with `kernels` still sized from `n`.
        layer = NLinear(n=K, input_dim=D, output_dim=5)
        layer.build((None, None, D))
        assert tuple(layer.kernels.shape) == (K, D, 5)

    def test_deferred_input_dim_builds_and_roundtrips(self, tmp_path):
        # G-3: `input_dim=None` is the correct deferred fan-in idiom, not a
        # round-trip defect -- build() fills in the concrete value and that is
        # what get_config() serializes. This is the path TabMMLPBlock(packed) uses.
        layer = NLinear(n=K, input_dim=None, output_dim=5, name="nl_deferred")
        layer.build((None, K, D))
        assert layer.input_dim == D
        assert tuple(layer.kernels.shape) == (K, D, 5)
        assert layer.get_config()["input_dim"] == D

        _roundtrip(NLinear(n=K, input_dim=None, output_dim=5, name="nld"), (K, D),
                   _f32(B, K, D), "nld", tmp_path, NLinear)


class TestTabMMLPBlock:
    def test_forward_no_ensemble(self):
        layer = TabMMLPBlock(units=8)
        out = layer(_f32(B, 10))
        assert tuple(out.shape) == (B, 8)
        assert layer.compute_output_shape((B, 10)) == (B, 8)

    def test_forward_ensemble(self):
        layer = TabMMLPBlock(units=8, k=K)
        out = layer(_f32(B, K, 10))
        assert tuple(out.shape) == (B, K, 8)

    def test_serialization(self, tmp_path):
        _roundtrip(TabMMLPBlock(units=8, name="mlp"), (10,), _f32(B, 10), "mlp", tmp_path, TabMMLPBlock)

    @pytest.mark.parametrize("kwargs, bad", [
        (dict(units=0), "0"),
        (dict(units=-3), "-3"),
        (dict(units=8, k=0), "0"),
        (dict(units=8, k=-2), "-2"),
        (dict(units=8, dropout_rate=1.5), "1.5"),
        (dict(units=8, dropout_rate=-0.1), "-0.1"),
        # k=None means there is no ensemble, so the `if self.k is None` branch wins
        # and a plain Dense is built. Before this arm existed, `ensemble_type` was
        # then SILENTLY discarded while `get_config()` still reported 'packed'.
        (dict(units=8, ensemble_type='packed'), "packed"),
    ])
    def test_constructor_rejects_out_of_range(self, kwargs, bad):
        with pytest.raises(ValueError) as exc:
            TabMMLPBlock(**kwargs)
        # The message must name the offending value, not just the argument.
        assert bad in str(exc.value)

    def test_packed_with_k_none_is_rejected_not_silently_discarded(self):
        # The discriminating observable is the DISAGREEMENT the raise now prevents:
        # against the pre-fix code `type(block.linear).__name__ == 'Dense'` while
        # `get_config()['ensemble_type'] == 'packed'`. Asserting only that a
        # ValueError is raised would not say what it protects, so name both halves.
        with pytest.raises(ValueError) as exc:
            TabMMLPBlock(units=8, ensemble_type='packed')
        msg = str(exc.value)
        assert "k is None" in msg
        assert "packed" in msg
        # And the raise belongs to this layer, not to a sub-layer constructor.
        assert os.path.basename(str(exc.traceback[-1].path)) == "tabm_mlp_block.py"

    @pytest.mark.parametrize("kwargs", [
        dict(units=8),
        dict(units=8, k=K),
        dict(units=8, dropout_rate=0.0),
        dict(units=8, dropout_rate=1.0),
        # The shipped 'plain' TabM path: ARCH_SPECS['plain'] passes the DEFAULT
        # ensemble_type together with k=None, so the guard above must not fire on it.
        dict(units=8, ensemble_type='efficient'),
        # 'packed' is legal the moment there is actually an ensemble to pack.
        dict(units=8, k=K, ensemble_type='packed'),
    ])
    def test_constructor_accepts_valid(self, kwargs):
        # Positive controls: the guards above must not fire on shipped configs.
        assert TabMMLPBlock(**kwargs) is not None

    def test_ensemble_type_reaches_the_built_sublayer(self):
        # `get_config()` echoing a knob proves nothing about what was built; assert
        # the sub-layer TYPE, which is the only thing that distinguishes the two
        # ensemble realizations.
        assert type(TabMMLPBlock(units=8, k=K, ensemble_type='packed').linear) is NLinear
        assert type(
            TabMMLPBlock(units=8, k=K, ensemble_type='efficient').linear
        ) is LinearEfficientEnsemble
        assert type(TabMMLPBlock(units=8).linear) is keras.layers.Dense

    def test_activation_is_stored_verbatim_not_eagerly_resolved(self):
        # RED against the pre-split implementation, which did
        # `self.activation = keras.activations.get(activation)` and therefore
        # stored a FUNCTION. `deserialize_activation` returns a string unchanged,
        # so the key survives into `get_config()` verbatim -- which is the whole
        # point of the pair (a Keras-unknown factory key would otherwise be
        # destroyed on the way in). `activation_fn` carries the live callable.
        layer = TabMMLPBlock(units=8, activation="relu")
        assert layer.activation == "relu"
        assert not callable(layer.activation)
        assert callable(layer.activation_fn)
        assert layer.get_config()["activation"] == "relu"

    @pytest.mark.parametrize("activation", ["relu", "mish", keras.activations.gelu])
    def test_activation_config_is_json_serializable(self, activation):
        # The observable the activation_serialization module docstring names as
        # discriminating: `get_config()` must be JSON-safe for every accepted form.
        json.dumps(TabMMLPBlock(units=8, activation=activation).get_config())

    @pytest.mark.parametrize("activation, tag", [
        ("relu", "act_str"),
        ("mish", "act_mish"),
        (keras.activations.gelu, "act_callable"),
    ])
    def test_activation_roundtrip(self, activation, tag, tmp_path):
        # `mish` is a dl_techniques activation key; the live callable is the form
        # TabMBackbone hands down when a caller passes one. These two forms plus a
        # plain name string are the ONLY supported ones -- a Layer is rejected, see
        # `test_activation_layer_is_rejected`.
        _roundtrip(TabMMLPBlock(units=8, activation=activation, name=tag), (10,),
                   _f32(B, 10), tag, tmp_path, TabMMLPBlock)

    @pytest.mark.parametrize("act_layer, tag", [
        (keras.layers.PReLU(), "prelu"),   # parameterised: owns a variable
        (GoLU(), "golu"),                  # stateless: owns none
    ])
    def test_activation_layer_is_rejected(self, act_layer, tag):
        # Replaces the previous `GoLU()` ROUND-TRIP arm, which could not fail:
        # MEASURED, GoLU has zero weights and is shape-agnostic, so it was blind to
        # both real defects of the Layer path. Those defects, MEASURED at 255400a0f:
        #   (a) `build()` builds only `self.linear` and `self.dropout`, never the
        #       activation, so a functional model wrapping
        #       `TabMMLPBlock(units=4, k=3, activation=keras.layers.PReLU())` had
        #       4 weights at `model.save()` time and 5 after the first forward pass
        #       -- the archive silently omitted `alpha`;
        #   (b) `TabMBackbone` hands ONE instance to every block, so
        #       `TabMBackbone(hidden_dims=[8, 6], k=3, activation=keras.layers.PReLU())`
        #       raised `InvalidArgumentError: Incompatible shapes: [3,8] vs [2,3,6]`.
        # The stateless arm is included deliberately: the guard is on the TYPE, not
        # on whether the instance happens to own variables today, so a future
        # stateful rewrite of a currently-stateless activation Layer cannot slip in.
        with pytest.raises(ValueError) as exc:
            TabMMLPBlock(units=8, activation=act_layer)
        msg = str(exc.value)
        assert type(act_layer).__name__ in msg   # names what was passed
        assert "'relu'" in msg                   # names the remedy: a name string
        assert "keras.activations.gelu" in msg   # ... or a stateless callable

    def test_activation_layer_config_dict_is_rejected_too(self):
        # The `from_config` route: a config written before the guard existed carries
        # the Layer as a dict, and `deserialize_activation` turns it back into a live
        # Layer. Validating the RAW argument instead of the deserialized value would
        # leave that route open, so the guard must sit after the deserialize call.
        cfg = keras.saving.serialize_keras_object(GoLU())
        assert isinstance(cfg, dict)
        with pytest.raises(ValueError, match="Layer instance"):
            TabMMLPBlock(units=8, activation=cfg)

    @pytest.mark.parametrize("rate, tag", [(0.0, "drop0"), (0.5, "drop5")])
    def test_dropout_object_exists_at_every_rate(self, rate, tag, tmp_path):
        # v2 s1.3 / fix (b): the Dropout object is created unconditionally, so the
        # object graph and sibling auto-names do not shift with the rate. RED
        # against the pre-split implementation, which set `self.dropout = None`
        # at rate 0. Only `build()` and `call()` gate on the rate.
        layer = TabMMLPBlock(units=8, dropout_rate=rate, name=tag)
        out = layer(_f32(B, 10), training=False)
        assert tuple(out.shape) == (B, 8)
        assert layer.dropout is not None
        assert isinstance(layer.dropout, keras.layers.Dropout)
        assert layer.dropout.rate == rate
        _roundtrip(TabMMLPBlock(units=8, dropout_rate=rate, name=f"{tag}_rt"), (10,),
                   _f32(B, 10), f"{tag}_rt", tmp_path, TabMMLPBlock)


class TestTabMBackbone:
    def test_forward_and_shape(self):
        layer = TabMBackbone(hidden_dims=[8, 6])
        out = layer(_f32(B, 10))
        assert tuple(out.shape) == (B, 6)

    def test_serialization(self, tmp_path):
        _roundtrip(TabMBackbone(hidden_dims=[8, 6], name="backbone"), (10,), _f32(B, 10),
                   "backbone", tmp_path, TabMBackbone)

    @pytest.mark.parametrize("kwargs, expected", [
        # An EMPTY hidden_dims is the arm that is RED against the pre-split code:
        # zero blocks were constructed, so nothing downstream ever complained and
        # the backbone silently became an identity map.
        (dict(hidden_dims=[]), ("hidden_dims", "[]")),
        # A bad ENTRY must be attributed to its index. TabMMLPBlock's own `units`
        # guard would also fire here, but its message cannot name the position --
        # asserting on "index N" is what makes this arm the backbone's guard.
        (dict(hidden_dims=[8, 0]), ("index 1", "0")),
        (dict(hidden_dims=[0, 8]), ("index 0", "0")),
        (dict(hidden_dims=[8, -4]), ("index 1", "-4")),
        (dict(hidden_dims=[8], dropout_rate=1.5), ("dropout_rate", "1.5")),
        (dict(hidden_dims=[8], dropout_rate=-0.1), ("dropout_rate", "-0.1")),
    ])
    def test_constructor_rejects_out_of_range(self, kwargs, expected):
        with pytest.raises(ValueError) as exc:
            TabMBackbone(**kwargs)
        # The message must name the offending value -- and, for a bad entry, its index.
        for token in expected:
            assert token in str(exc.value)

    @pytest.mark.parametrize("rate", [1.5, -0.1])
    def test_dropout_rate_is_rejected_by_the_backbone_itself(self, rate):
        # TabMMLPBlock raises a message-identical ValueError for the same value, so
        # the message alone cannot tell the two guards apart. What discriminates is
        # WHERE the raise happens: the backbone must reject before it constructs any
        # block, i.e. the innermost frame is this module, not tabm_mlp_block.py.
        with pytest.raises(ValueError) as exc:
            TabMBackbone(hidden_dims=[8, 6], dropout_rate=rate)
        assert os.path.basename(str(exc.traceback[-1].path)) == "tabm_backbone.py"

    @pytest.mark.parametrize("kwargs", [
        dict(hidden_dims=[8]),
        dict(hidden_dims=[8, 6]),
        dict(hidden_dims=[8, 6], k=K),
        dict(hidden_dims=[8], dropout_rate=0.0),
        dict(hidden_dims=[8], dropout_rate=1.0),
    ])
    def test_constructor_accepts_valid(self, kwargs):
        # Positive controls: the guards above must not fire on shipped configs.
        assert TabMBackbone(**kwargs) is not None

    def test_activation_is_handed_down_verbatim(self):
        # The backbone stores the SERIALIZABLE value and never calls it; each block
        # resolves its own live callable. Pins step 6's activation/activation_fn
        # split at the hand-down boundary: a string must arrive at the block as a
        # string, or `serialize_activation` in the block's `get_config` would be
        # handed a resolved function instead of the factory key.
        layer = TabMBackbone(hidden_dims=[8, 8], activation="mish")
        assert layer.activation == "mish"
        assert not callable(layer.activation)
        assert layer.get_config()["activation"] == "mish"
        for block in layer.blocks:
            assert block.activation == "mish"
            assert callable(block.activation_fn)
            assert block.get_config()["activation"] == "mish"

    def test_activation_roundtrip(self, tmp_path):
        # `mish` is a dl_techniques-reachable key, not a Keras builtin object, so a
        # backbone that eagerly resolved it would round-trip a different function.
        _roundtrip(TabMBackbone(hidden_dims=[8, 8], activation="mish", name="bb_mish"),
                   (10,), _f32(B, 10), "bb_mish", tmp_path, TabMBackbone)

    def test_activation_layer_is_rejected_before_any_block_is_usable(self):
        # The backbone hands ONE `activation` object to every block, so a stateful
        # activation Layer would be SHARED across blocks of different widths.
        # MEASURED at 255400a0f: `TabMBackbone(hidden_dims=[8, 6], k=3,
        # activation=keras.layers.PReLU())` raised `InvalidArgumentError:
        # Incompatible shapes: [3,8] vs [2,3,6]` on the forward pass -- a shape error
        # from deep inside PReLU, arbitrarily far from the constructor that caused it.
        # The guard lives in TabMMLPBlock (one copy, not two that can drift), so the
        # raise is a ValueError from that frame, at CONSTRUCTION time.
        with pytest.raises(ValueError) as exc:
            TabMBackbone(hidden_dims=[8, 6], k=K, activation=keras.layers.PReLU())
        assert "Layer instance" in str(exc.value)
        assert os.path.basename(str(exc.traceback[-1].path)) == "tabm_mlp_block.py"

    def test_packed_with_k_none_is_rejected_through_the_blocks(self):
        # Same delegation, second knob: the backbone does not re-implement the
        # ensemble_type/k interaction, it inherits the block's guard.
        with pytest.raises(ValueError) as exc:
            TabMBackbone(hidden_dims=[8, 6], ensemble_type='packed')
        assert "k is None" in str(exc.value)
        assert os.path.basename(str(exc.traceback[-1].path)) == "tabm_mlp_block.py"


class TestRegistrationKeys:
    """The five keys this package's classes are deserialized by, pinned literally.

    A ``.keras`` round trip can NEVER catch a wrong ``package=`` string: the archive
    is written and read through the SAME in-process registry, so a typo is
    self-consistent and loads fine. The string is hand-authored and consumed
    literally, with no derivation step, so nothing mechanical catches a stale one
    either -- an assertion on the exact key text is the only instrument that works.
    That is why these five arms spell the key out instead of asserting merely that
    some key exists, and why they are collected tests rather than a one-shot grep.

    ``registration_contract`` (tests/conftest.py) asserts the shared half: the key is
    package-qualified, owned by ``dl_techniques``, resolves back to this exact class,
    and its legacy ``Custom>{__name__}`` alias still resolves to the same object. It
    returns the key; the ``==`` below is the module-path half it deliberately does
    not hard-code. Same shape as
    ``tests/test_initializers/test_random_signs.py::test_registered_name_resolves``.

    All five keys were REWRITTEN by ``plan-2026-09-07-b821967f`` when the classes
    moved out of ``layers/tabular/tabm_blocks.py`` (D-003: clean break, no
    ``legacy_packages=`` alias), so a regression here is not hypothetical -- it is
    the edit that plan already made once.
    """

    @pytest.mark.parametrize("cls, expected_key", [
        (ScaleEnsemble,
         "dl_techniques.layers.tabular.scale_ensemble>ScaleEnsemble"),
        (LinearEfficientEnsemble,
         "dl_techniques.layers.tabular.linear_efficient_ensemble"
         ">LinearEfficientEnsemble"),
        (NLinear,
         "dl_techniques.layers.tabular.nlinear>NLinear"),
        (TabMMLPBlock,
         "dl_techniques.layers.tabular.tabm_mlp_block>TabMMLPBlock"),
        (TabMBackbone,
         "dl_techniques.layers.tabular.tabm_backbone>TabMBackbone"),
    ], ids=["ScaleEnsemble", "LinearEfficientEnsemble", "NLinear",
            "TabMMLPBlock", "TabMBackbone"])
    def test_registered_name_resolves(self, cls, expected_key,
                                      registration_contract):
        key = registration_contract(cls)
        assert key == expected_key


class TestOutputShapeContract:
    """`compute_output_shape` must equal what `call()` actually produces.

    N-8 was a DISAGREEMENT, not a crash: `LinearEfficientEnsemble` derived axis 1
    from `input_shape[1]` while `TabMMLPBlock` derived it from `self.k`, so the
    same nominal input got two different answers and neither site was obviously
    wrong on its own. A per-class unit assertion cannot see that -- only a
    contract applied to every class at once can, which is why these nine arms
    cover all five classes in plain mode and in BOTH ensemble types.

    `TabMBackbone.build()` threads `current_shape = block.compute_output_shape(
    current_shape)` from block to block, so a wrong source does not stay local:
    it propagates through the whole stack while every reported shape stays
    self-consistent. The backbone arms are the ones that measure that.

    Each arm asserts two things:

    * concrete -- `compute_output_shape((B,) + shape)` equals the shape of a REAL
      forward pass, so the prediction is checked against the tensor, not against
      another prediction;
    * symbolic -- `compute_output_shape((None,) + shape)` keeps the batch axis
      `None` and agrees with the concrete output on every other axis, so an
      unknown batch size neither leaks into nor is invented for the feature axes.
    """

    SUBJECTS = [
        (lambda: ScaleEnsemble(k=K, input_dim=D), (K, D)),
        (lambda: LinearEfficientEnsemble(units=5, k=K), (K, D)),
        (lambda: NLinear(n=K, input_dim=D, output_dim=5), (K, D)),
        (lambda: TabMMLPBlock(units=8), (D,)),
        (lambda: TabMMLPBlock(units=8, k=K, ensemble_type='efficient'), (K, D)),
        (lambda: TabMMLPBlock(units=8, k=K, ensemble_type='packed'), (K, D)),
        (lambda: TabMBackbone(hidden_dims=[8, 6]), (D,)),
        (lambda: TabMBackbone(hidden_dims=[8, 6], k=K, ensemble_type='efficient'), (K, D)),
        (lambda: TabMBackbone(hidden_dims=[8, 6], k=K, ensemble_type='packed'), (K, D)),
    ]

    @pytest.mark.parametrize("make, shape", SUBJECTS, ids=[
        "ScaleEnsemble",
        "LinearEfficientEnsemble",
        "NLinear",
        "TabMMLPBlock-plain",
        "TabMMLPBlock-efficient",
        "TabMMLPBlock-packed",
        "TabMBackbone-plain",
        "TabMBackbone-efficient",
        "TabMBackbone-packed",
    ])
    def test_compute_output_shape_matches_call(self, make, shape):
        layer = make()

        # Concrete arm: the prediction is checked against a real tensor.
        actual = tuple(layer(_f32(B, *shape)).shape)
        assert tuple(layer.compute_output_shape((B,) + shape)) == actual

        # Symbolic arm: an unknown batch axis stays unknown and must not change
        # any feature axis.
        symbolic = tuple(layer.compute_output_shape((None,) + shape))
        assert symbolic[0] is None
        assert symbolic[1:] == actual[1:]
