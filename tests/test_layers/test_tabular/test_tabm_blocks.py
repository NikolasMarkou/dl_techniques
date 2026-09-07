"""Tests for the TabM building blocks (efficient tabular ensembles)."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.tabular.tabm_blocks import (
    ScaleEnsemble,
    LinearEfficientEnsemble,
    NLinear,
    TabMMLPBlock,
    TabMBackbone,
)

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


class TestLinearEfficientEnsemble:
    def test_forward_and_shape(self):
        layer = LinearEfficientEnsemble(units=5, k=K)
        out = layer(_f32(B, K, D))
        assert tuple(out.shape) == (B, K, 5)

    def test_serialization(self, tmp_path):
        _roundtrip(LinearEfficientEnsemble(units=5, k=K, name="lee"), (K, D), _f32(B, K, D),
                   "lee", tmp_path, LinearEfficientEnsemble)


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


class TestTabMBackbone:
    def test_forward_and_shape(self):
        layer = TabMBackbone(hidden_dims=[8, 6])
        out = layer(_f32(B, 10))
        assert tuple(out.shape) == (B, 6)

    def test_serialization(self, tmp_path):
        _roundtrip(TabMBackbone(hidden_dims=[8, 6], name="backbone"), (10,), _f32(B, 10),
                   "backbone", tmp_path, TabMBackbone)
