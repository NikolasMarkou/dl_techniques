"""Tests for the DenseBlock layer (relocated from test_standard_blocks.py).

Hardened per plan-2026-09-14T165315-47f9d575 step 11 to match the
`test_conv_blocks/` precedent's rigor (see step 10's `test_basic_block.py`):
exact (`rtol=0, atol=0`) `.keras` round trip, pre-call weight values,
no-sub-layer-built assertions for `normalization_type=None` and
`dropout_rate=0.0`, a per-variable gradient-flow assertion, `isfinite` on
every forward pass, a degenerate (1-unit / 1-feature) sweep, dtype-policy
arms with a float32 control, and XLA-vs-eager agreement.
"""

import os
import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.dense_block import DenseBlock

B = 2


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _roundtrip(layer, input_shape, data, name, tmp_path):
    inp = keras.Input(shape=input_shape)
    out = layer(inp)
    model = keras.Model(inp, out)
    y0 = model(data, training=False)
    path = os.path.join(tmp_path, f"{name}.keras")
    model.save(path)
    loaded = keras.models.load_model(path)
    y1 = loaded(data, training=False)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
        rtol=1e-5, atol=1e-5,
    )


class TestDenseBlock:
    def test_forward_and_shape(self):
        x = np.random.default_rng(0).standard_normal((B, 10)).astype("float32")
        layer = DenseBlock(units=8)
        out = layer(x)
        assert tuple(out.shape) == (B, 8)
        assert layer.compute_output_shape((B, 10)) == (B, 8)
        assert bool(keras.ops.all(keras.ops.isfinite(out)))

    def test_serialization(self, tmp_path):
        x = np.random.default_rng(0).standard_normal((B, 10)).astype("float32")
        _roundtrip(DenseBlock(units=8, name="dense"), (10,), x, "dense", tmp_path)


class TestInvalidArgs:
    @pytest.mark.parametrize("ctor", [
        lambda: DenseBlock(units=0),
    ])
    def test_invalid_args_raise(self, ctor):
        with pytest.raises(ValueError):
            ctor()


# ---------------------------------------------------------------------
# Weight values BEFORE the first call (construction + explicit build())
# ---------------------------------------------------------------------

class TestWeightsBeforeFirstCall:
    def test_dense_bias_is_exact_before_any_call(self):
        """`build()` alone (never `call()`) must populate the exact
        zero-initialized bias — the default `bias_initializer='zeros'`.
        """
        layer = DenseBlock(units=8, name="preweights")
        layer.build((B, 10))

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(layer.dense.bias), 0.0, rtol=0.0, atol=0.0
        )
        assert layer.dense.kernel.shape == (10, 8)


# ---------------------------------------------------------------------
# No-sub-layer-built assertions for None/False config branches
# ---------------------------------------------------------------------

class TestDisabledBranchesBuildNothing:
    def test_norm_is_none_when_normalization_type_is_none(self):
        layer = DenseBlock(units=8, normalization_type=None, name="no_norm")
        layer.build((B, 10))

        assert layer.norm is None
        tracked = [v for v in vars(layer).values() if isinstance(v, keras.layers.Layer)]
        norm_like = [t for t in tracked if t.name == f"{layer.name}_norm"]
        assert not norm_like, "a normalization sub-layer survived despite normalization_type=None"

    def test_dropout_is_none_when_rate_is_zero(self):
        layer = DenseBlock(units=8, dropout_rate=0.0, name="no_dropout")
        layer.build((B, 10))

        assert layer.dropout is None
        tracked = [v for v in vars(layer).values() if isinstance(v, keras.layers.Layer)]
        dropout_like = [t for t in tracked if isinstance(t, keras.layers.Dropout)]
        assert not dropout_like, "a Dropout sub-layer survived despite dropout_rate=0.0"


# ---------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------

class TestGradientFlow:
    def test_every_trainable_weight_receives_a_gradient(self, rng):
        layer = DenseBlock(
            units=8, normalization_type="layer_norm", dropout_rate=0.0, name="grad"
        )
        x = tf.constant(rng.standard_normal((B, 10)).astype("float32"))

        with tf.GradientTape() as tape:
            y = layer(x, training=True)
            loss = keras.ops.mean(keras.ops.square(y))

        grads = tape.gradient(loss, layer.trainable_weights)
        assert len(layer.trainable_weights) > 0
        for weight, grad in zip(layer.trainable_weights, grads):
            assert grad is not None, f"{weight.path} received no gradient"
            grad_np = keras.ops.convert_to_numpy(grad)
            assert np.isfinite(grad_np).all(), f"{weight.path} gradient is non-finite"
            assert np.max(np.abs(grad_np)) > 0.0, f"{weight.path} gradient is identically zero"


# ---------------------------------------------------------------------
# Degenerate feature dimension
# ---------------------------------------------------------------------

class TestDegenerateFeatureDimension:
    def test_single_unit_output(self, rng):
        x = rng.standard_normal((B, 10)).astype("float32")
        layer = DenseBlock(units=1, name="degenerate_units")
        out = layer(x)
        assert tuple(out.shape) == (B, 1)
        assert bool(keras.ops.all(keras.ops.isfinite(out)))

    def test_single_feature_input(self, rng):
        x = rng.standard_normal((B, 1)).astype("float32")
        layer = DenseBlock(units=8, name="degenerate_input")
        out = layer(x)
        assert tuple(out.shape) == (B, 8)
        assert bool(keras.ops.all(keras.ops.isfinite(out)))


# ---------------------------------------------------------------------
# dtype policy (float32 control + mixed_float16 + float64)
# ---------------------------------------------------------------------

class TestDtypePolicy:
    def test_forward_pass_is_finite_under_every_policy(self, dtype_policy, rng):
        x = rng.standard_normal((B, 10)).astype("float32")
        layer = DenseBlock(units=8, name=f"dtype_{dtype_policy}")
        out = layer(x, training=False)
        out_np = keras.ops.convert_to_numpy(out)
        assert np.isfinite(out_np).all()
        assert (layer.compute_dtype != layer.variable_dtype) == (dtype_policy == "mixed_float16")


# ---------------------------------------------------------------------
# XLA (jit_compile=True) versus eager
# ---------------------------------------------------------------------

class TestXlaVersusEager:
    def test_jit_compiled_output_matches_eager(self, rng):
        x = rng.standard_normal((B, 10)).astype("float32")
        layer = DenseBlock(units=8, name="jit_dense_block")
        y_eager = keras.ops.convert_to_numpy(layer(x, training=False))

        @tf.function(jit_compile=True)
        def compiled(t):
            return layer(t, training=False)

        y_jit = np.asarray(compiled(tf.constant(x)))

        assert y_jit.shape == y_eager.shape
        assert np.all(np.isfinite(y_jit))
        peak = float(np.max(np.abs(y_eager)))
        assert peak > 0.0, "output is degenerate, comparison is vacuous"
        # One 10-wide Dense accumulation: bound conservatively on float32 eps
        # scaled by the accumulation length and output magnitude.
        n_accumulations = 10
        atol = n_accumulations * float(np.finfo(np.float32).eps) * max(1.0, peak)
        np.testing.assert_allclose(y_jit, y_eager, rtol=0.0, atol=atol)


# ---------------------------------------------------------------------
# .keras round trip at rtol=0
# ---------------------------------------------------------------------

class TestKerasRoundTripExact:
    def test_round_trip_is_bit_identical(self, tmp_path, rng):
        x = rng.standard_normal((B, 10)).astype("float32")
        inp = keras.Input(shape=(10,))
        out = DenseBlock(units=8, name="exact_rt")(inp)
        model = keras.Model(inp, out)
        y0 = model(x, training=False)

        path = os.path.join(tmp_path, "exact_rt.keras")
        model.save(path)
        loaded = keras.models.load_model(path)

        original = {w.path: keras.ops.convert_to_numpy(w) for w in model.weights}
        restored = {w.path: keras.ops.convert_to_numpy(w) for w in loaded.weights}
        assert set(original) == set(restored)
        for key in original:
            np.testing.assert_allclose(
                original[key], restored[key], rtol=0.0, atol=0.0,
                err_msg=f"weight {key} changed across the .keras round trip",
            )

        y1 = loaded(x, training=False)
        assert bool(keras.ops.all(keras.ops.isfinite(y1)))
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=0.0, atol=0.0,
            err_msg="reloaded model is not bit-identical to the original",
        )
