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

    def test_build_accepts_matching_input_dim(self):
        # Positive control for the shape contract.
        layer = NLinear(n=K, input_dim=D, output_dim=5)
        layer.build((None, K, D))
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
    ])
    def test_constructor_rejects_out_of_range(self, kwargs, bad):
        with pytest.raises(ValueError) as exc:
            TabMMLPBlock(**kwargs)
        # The message must name the offending value, not just the argument.
        assert bad in str(exc.value)

    @pytest.mark.parametrize("kwargs", [
        dict(units=8),
        dict(units=8, k=K),
        dict(units=8, dropout_rate=0.0),
        dict(units=8, dropout_rate=1.0),
    ])
    def test_constructor_accepts_valid(self, kwargs):
        # Positive controls: the guards above must not fire on shipped configs.
        assert TabMMLPBlock(**kwargs) is not None

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
        (GoLU(), "act_layer"),
    ])
    def test_activation_roundtrip(self, activation, tag, tmp_path):
        # `mish` and the GoLU layer are dl_techniques activations; the live
        # callable is the form TabMBackbone hands down when a caller passes one.
        _roundtrip(TabMMLPBlock(units=8, activation=activation, name=tag), (10,),
                   _f32(B, 10), tag, tmp_path, TabMMLPBlock)

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
