"""
Test suite for the SD3 AdaLN modulation trio.

Covers, for each of :class:`AdaLayerNormZero`, :class:`AdaLayerNormZeroX`,
:class:`AdaLayerNormContinuous`: instantiation + ctor validation, forward
return shapes (``x_norm`` ``(B,N,dim)``; chunks ``(B,dim)``), modulation
broadcast across ``N>1``, AdaLN-Zero identity-at-init (``x_norm`` equals a
plain no-affine LayerNorm of ``x`` because the modulation Dense is
zero-initialized), ``compute_output_shape`` pre/post build, ``get_config`` /
``from_config`` round-trip, a full ``.keras`` save/load round-trip via a
two-input Functional model, and variable batch / variable N.
"""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.layers.transformers.sd3_adaln import (
    AdaLayerNormZero,
    AdaLayerNormZeroX,
    AdaLayerNormContinuous,
)


DIM = 64
N = 16
BATCH = 2
EPS = 1e-6


def _no_affine_layernorm(x_np: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Reference no-affine LayerNorm over the last axis."""
    mean = x_np.mean(axis=-1, keepdims=True)
    var = x_np.var(axis=-1, keepdims=True)
    return (x_np - mean) / np.sqrt(var + eps)


@pytest.fixture
def sample():
    keras.utils.set_random_seed(42)
    x = keras.random.normal((BATCH, N, DIM))
    cond = keras.random.normal((BATCH, DIM))
    return x, cond


# =====================================================================
# AdaLayerNormZero
# =====================================================================


class TestAdaLayerNormZero:

    @pytest.fixture
    def layer(self):
        return AdaLayerNormZero(dim=DIM, eps=EPS)

    def test_initialization(self, layer):
        assert layer.dim == DIM
        assert layer.eps == EPS
        assert layer.linear.units == 6 * DIM
        assert layer.norm.center is False
        assert layer.norm.scale is False

    def test_ctor_raises_on_bad_dim(self):
        with pytest.raises(ValueError):
            AdaLayerNormZero(dim=0)

    def test_ctor_raises_on_bad_eps(self):
        with pytest.raises(ValueError):
            AdaLayerNormZero(dim=DIM, eps=0.0)

    def test_forward_shapes(self, layer, sample):
        x, cond = sample
        x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = layer([x, cond])
        assert tuple(x_norm.shape) == (BATCH, N, DIM)
        for chunk in (gate_msa, shift_mlp, scale_mlp, gate_mlp):
            assert tuple(chunk.shape) == (BATCH, DIM)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(x_norm)))

    def test_modulation_broadcast_across_n(self, layer):
        """Modulation (B,dim) -> (B,1,dim) must broadcast over arbitrary N."""
        keras.utils.set_random_seed(1)
        for n in (1, 5, 33):
            x = keras.random.normal((BATCH, n, DIM))
            cond = keras.random.normal((BATCH, DIM))
            x_norm = layer([x, cond])[0]
            assert tuple(x_norm.shape) == (BATCH, n, DIM)

    def test_identity_at_init(self, layer, sample):
        """Zero-init Dense => x_norm == no-affine LayerNorm(x); gates ~0."""
        x, cond = sample
        x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = layer([x, cond])
        ref = _no_affine_layernorm(keras.ops.convert_to_numpy(x))
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(x_norm), ref, atol=1e-5
        )
        for chunk in (gate_msa, shift_mlp, scale_mlp, gate_mlp):
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(chunk), 0.0, atol=1e-6
            )

    def test_compute_output_shape_before_build(self):
        layer = AdaLayerNormZero(dim=DIM)
        out = layer.compute_output_shape([(BATCH, N, DIM), (BATCH, DIM)])
        assert out == [
            (BATCH, N, DIM),
            (BATCH, DIM),
            (BATCH, DIM),
            (BATCH, DIM),
            (BATCH, DIM),
        ]

    def test_compute_output_shape_matches_actual(self, layer, sample):
        x, cond = sample
        outs = layer([x, cond])
        computed = layer.compute_output_shape([tuple(x.shape), tuple(cond.shape)])
        for c, o in zip(computed, outs):
            assert c == tuple(o.shape)

    def test_get_config_round_trip(self):
        layer = AdaLayerNormZero(dim=128, eps=1e-5)
        cfg = layer.get_config()
        assert cfg["dim"] == 128
        assert cfg["eps"] == 1e-5
        rebuilt = AdaLayerNormZero.from_config(cfg)
        assert rebuilt.dim == 128
        assert rebuilt.eps == 1e-5

    def test_keras_serialization_round_trip(self, sample):
        x, cond = sample
        x_in = keras.Input(shape=(N, DIM), name="x")
        c_in = keras.Input(shape=(DIM,), name="cond")
        outs = AdaLayerNormZero(dim=DIM)([x_in, c_in])
        model = keras.Model([x_in, c_in], list(outs))
        inputs = {
            "x": keras.ops.convert_to_numpy(x),
            "cond": keras.ops.convert_to_numpy(cond),
        }
        before = model.predict(inputs, verbose=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "adaln_zero.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            after = reloaded.predict(inputs, verbose=0)
        for a, b in zip(before, after):
            np.testing.assert_allclose(a, b, atol=1e-6)

    def test_variable_batch(self, layer):
        keras.utils.set_random_seed(7)
        for b in (1, 3, 5):
            x = keras.random.normal((b, N, DIM))
            cond = keras.random.normal((b, DIM))
            x_norm, gate_msa = layer([x, cond])[0], layer([x, cond])[1]
            assert tuple(x_norm.shape) == (b, N, DIM)
            assert tuple(gate_msa.shape) == (b, DIM)


# =====================================================================
# AdaLayerNormZeroX
# =====================================================================


class TestAdaLayerNormZeroX:

    @pytest.fixture
    def layer(self):
        return AdaLayerNormZeroX(dim=DIM, eps=EPS)

    def test_initialization(self, layer):
        assert layer.dim == DIM
        assert layer.linear.units == 9 * DIM

    def test_ctor_raises_on_bad_dim(self):
        with pytest.raises(ValueError):
            AdaLayerNormZeroX(dim=-1)

    def test_forward_shapes(self, layer, sample):
        x, cond = sample
        (x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp,
         x_norm2, gate_msa2) = layer([x, cond])
        assert tuple(x_norm.shape) == (BATCH, N, DIM)
        assert tuple(x_norm2.shape) == (BATCH, N, DIM)
        for chunk in (gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_msa2):
            assert tuple(chunk.shape) == (BATCH, DIM)

    def test_modulation_broadcast_across_n(self, layer):
        keras.utils.set_random_seed(2)
        for n in (1, 5, 33):
            x = keras.random.normal((BATCH, n, DIM))
            cond = keras.random.normal((BATCH, DIM))
            outs = layer([x, cond])
            assert tuple(outs[0].shape) == (BATCH, n, DIM)
            assert tuple(outs[5].shape) == (BATCH, n, DIM)

    def test_identity_at_init(self, layer, sample):
        """Both x_norm and x_norm2 == no-affine LayerNorm(x); gates ~0."""
        x, cond = sample
        (x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp,
         x_norm2, gate_msa2) = layer([x, cond])
        ref = _no_affine_layernorm(keras.ops.convert_to_numpy(x))
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(x_norm), ref, atol=1e-5
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(x_norm2), ref, atol=1e-5
        )
        # x_norm and x_norm2 share norm(x) at init => identical.
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(x_norm),
            keras.ops.convert_to_numpy(x_norm2),
            atol=1e-6,
        )
        for chunk in (gate_msa, shift_mlp, scale_mlp, gate_mlp, gate_msa2):
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(chunk), 0.0, atol=1e-6
            )

    def test_compute_output_shape_before_build(self):
        layer = AdaLayerNormZeroX(dim=DIM)
        out = layer.compute_output_shape([(BATCH, N, DIM), (BATCH, DIM)])
        assert out == [
            (BATCH, N, DIM),
            (BATCH, DIM),
            (BATCH, DIM),
            (BATCH, DIM),
            (BATCH, DIM),
            (BATCH, N, DIM),
            (BATCH, DIM),
        ]

    def test_compute_output_shape_matches_actual(self, layer, sample):
        x, cond = sample
        outs = layer([x, cond])
        computed = layer.compute_output_shape([tuple(x.shape), tuple(cond.shape)])
        for c, o in zip(computed, outs):
            assert c == tuple(o.shape)

    def test_get_config_round_trip(self):
        layer = AdaLayerNormZeroX(dim=96, eps=1e-4)
        cfg = layer.get_config()
        assert cfg["dim"] == 96
        assert cfg["eps"] == 1e-4
        rebuilt = AdaLayerNormZeroX.from_config(cfg)
        assert rebuilt.dim == 96
        assert rebuilt.eps == 1e-4

    def test_keras_serialization_round_trip(self, sample):
        x, cond = sample
        x_in = keras.Input(shape=(N, DIM), name="x")
        c_in = keras.Input(shape=(DIM,), name="cond")
        outs = AdaLayerNormZeroX(dim=DIM)([x_in, c_in])
        model = keras.Model([x_in, c_in], list(outs))
        inputs = {
            "x": keras.ops.convert_to_numpy(x),
            "cond": keras.ops.convert_to_numpy(cond),
        }
        before = model.predict(inputs, verbose=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "adaln_zerox.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            after = reloaded.predict(inputs, verbose=0)
        for a, b in zip(before, after):
            np.testing.assert_allclose(a, b, atol=1e-6)

    def test_variable_batch(self, layer):
        keras.utils.set_random_seed(8)
        for b in (1, 3, 5):
            x = keras.random.normal((b, N, DIM))
            cond = keras.random.normal((b, DIM))
            outs = layer([x, cond])
            assert tuple(outs[0].shape) == (b, N, DIM)
            assert tuple(outs[5].shape) == (b, N, DIM)


# =====================================================================
# AdaLayerNormContinuous
# =====================================================================


class TestAdaLayerNormContinuous:

    @pytest.fixture
    def layer(self):
        return AdaLayerNormContinuous(dim=DIM, eps=EPS)

    def test_initialization(self, layer):
        assert layer.dim == DIM
        assert layer.linear.units == 2 * DIM
        assert layer.norm.center is False
        assert layer.norm.scale is False

    def test_ctor_raises_on_bad_dim(self):
        with pytest.raises(ValueError):
            AdaLayerNormContinuous(dim=0)

    def test_forward_shape_single_tensor(self, layer, sample):
        x, cond = sample
        out = layer([x, cond])
        assert not isinstance(out, (list, tuple))
        assert tuple(out.shape) == (BATCH, N, DIM)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_modulation_broadcast_across_n(self, layer):
        keras.utils.set_random_seed(3)
        for n in (1, 5, 33):
            x = keras.random.normal((BATCH, n, DIM))
            cond = keras.random.normal((BATCH, DIM))
            out = layer([x, cond])
            assert tuple(out.shape) == (BATCH, n, DIM)

    def test_identity_at_init(self, layer, sample):
        """Zero-init Dense => output == no-affine LayerNorm(x)."""
        x, cond = sample
        out = layer([x, cond])
        ref = _no_affine_layernorm(keras.ops.convert_to_numpy(x))
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(out), ref, atol=1e-5
        )

    def test_compute_output_shape_before_build(self):
        layer = AdaLayerNormContinuous(dim=DIM)
        out = layer.compute_output_shape([(BATCH, N, DIM), (BATCH, DIM)])
        assert out == (BATCH, N, DIM)

    def test_compute_output_shape_matches_actual(self, layer, sample):
        x, cond = sample
        out = layer([x, cond])
        computed = layer.compute_output_shape([tuple(x.shape), tuple(cond.shape)])
        assert computed == tuple(out.shape)

    def test_get_config_round_trip(self):
        layer = AdaLayerNormContinuous(dim=80, eps=1e-5)
        cfg = layer.get_config()
        assert cfg["dim"] == 80
        assert cfg["eps"] == 1e-5
        rebuilt = AdaLayerNormContinuous.from_config(cfg)
        assert rebuilt.dim == 80
        assert rebuilt.eps == 1e-5

    def test_keras_serialization_round_trip(self, sample):
        x, cond = sample
        x_in = keras.Input(shape=(N, DIM), name="x")
        c_in = keras.Input(shape=(DIM,), name="cond")
        out = AdaLayerNormContinuous(dim=DIM)([x_in, c_in])
        model = keras.Model([x_in, c_in], out)
        inputs = {
            "x": keras.ops.convert_to_numpy(x),
            "cond": keras.ops.convert_to_numpy(cond),
        }
        before = model.predict(inputs, verbose=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "adaln_continuous.keras")
            model.save(path)
            reloaded = keras.models.load_model(path)
            after = reloaded.predict(inputs, verbose=0)
        np.testing.assert_allclose(before, after, atol=1e-6)

    def test_variable_batch(self, layer):
        keras.utils.set_random_seed(9)
        for b in (1, 3, 5):
            x = keras.random.normal((b, N, DIM))
            cond = keras.random.normal((b, DIM))
            out = layer([x, cond])
            assert tuple(out.shape) == (b, N, DIM)
