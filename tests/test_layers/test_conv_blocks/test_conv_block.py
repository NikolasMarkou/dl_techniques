"""Tests for ConvBlock (relocated from test_standard_blocks.py's TestConvBlock).

Hardened per plan-2026-09-14T165315-47f9d575 step 10 to match the
`test_conv_blocks/` precedent's rigor (see `test_gabor_depthwise_separable_block.py`
as the exemplar): exact (`rtol=0, atol=0`) `.keras` round trip, pre-call weight
values, build-parity / no-sub-layer-built assertions for the `None`/`False`
branches, a per-variable gradient-flow assertion, `isfinite` on every forward
pass, a degenerate (1x1) spatial-extent sweep, dtype-policy arms with a float32
control, and XLA-vs-eager agreement.
"""

import os
import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.conv_blocks.conv_block import ConvBlock

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


class TestConvBlock:
    def test_forward_and_shape(self):
        x = np.random.default_rng(0).standard_normal((B, 8, 8, 4)).astype("float32")
        layer = ConvBlock(filters=8, kernel_size=3)
        out = layer(x)
        assert tuple(out.shape) == (B, 8, 8, 8)
        assert layer.compute_output_shape((B, 8, 8, 4)) == (B, 8, 8, 8)
        assert bool(keras.ops.all(keras.ops.isfinite(out)))

    def test_serialization(self, tmp_path):
        x = np.random.default_rng(0).standard_normal((B, 8, 8, 4)).astype("float32")
        _roundtrip(ConvBlock(filters=8, kernel_size=3, name="conv"), (8, 8, 4), x, "conv", tmp_path)


class TestInvalidArgs:
    @pytest.mark.parametrize("ctor", [
        lambda: ConvBlock(filters=0, kernel_size=3),
    ])
    def test_invalid_args_raise(self, ctor):
        with pytest.raises(ValueError):
            ctor()


# ---------------------------------------------------------------------
# Weight values BEFORE the first call (construction + explicit build())
# ---------------------------------------------------------------------

class TestWeightsBeforeFirstCall:
    def test_zero_initializer_weights_are_exact_before_any_call(self, rng):
        """`build()` alone (never `call()`) must populate exact initial values.

        `kernel_initializer='zeros'` and Keras' default bias/BatchNorm inits
        (bias zeros, gamma ones, beta zeros, moving_mean zeros,
        moving_variance ones) are all deterministic, so this is a real
        value comparison at `atol=0.0`, not a shape-only check.
        """
        layer = ConvBlock(
            filters=6, kernel_size=3, kernel_initializer="zeros", name="preweights"
        )
        layer.build((B, 8, 8, 4))

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(layer.conv.kernel), 0.0, rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(layer.conv.bias), 0.0, rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(layer.norm.gamma), 1.0, rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(layer.norm.beta), 0.0, rtol=0.0, atol=0.0
        )
        assert layer.conv.kernel.shape == (3, 3, 4, 6)


# ---------------------------------------------------------------------
# Build parity + no-sub-layer-built assertions for None/False branches
# ---------------------------------------------------------------------

class TestNoneFalseBranchesBuildNothing:
    def test_dropout_and_pool_are_none_when_disabled(self):
        """`dropout_rate=0.0` and `use_pooling=False` (the defaults) must
        create neither sub-layer -- not merely skip calling them.
        """
        layer = ConvBlock(filters=6, kernel_size=3, name="no_extras")
        layer.build((B, 8, 8, 4))

        assert layer.dropout is None
        assert layer.pool is None
        # No dropout/pool instance survives anywhere in the layer's own
        # __dict__ (the only place a Keras Layer auto-tracks a sub-layer
        # attribute) -- a stray un-nulled reference would show up here even
        # though `self.dropout`/`self.pool` themselves read None.
        tracked = [v for v in vars(layer).values() if isinstance(v, keras.layers.Layer)]
        assert not any(isinstance(t, keras.layers.Dropout) for t in tracked)
        assert not any(
            isinstance(t, (keras.layers.MaxPooling2D, keras.layers.AveragePooling2D))
            for t in tracked
        )

# ---------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------

class TestGradientFlow:
    def test_every_trainable_weight_receives_a_gradient(self, rng):
        layer = ConvBlock(filters=6, kernel_size=3, name="grad")
        x = tf.constant(rng.standard_normal((B, 8, 8, 4)).astype("float32"))

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
# Degenerate spatial extent
# ---------------------------------------------------------------------

class TestDegenerateSpatialExtent:
    def test_one_by_one_input(self, rng):
        """A literal 0-length spatial dim is not constructible (Conv2D and
        every pooling op require at least one spatial position); 1x1 is the
        smallest valid degenerate case and, with the default 'same' padding,
        is a legal Conv2D input at every stride/kernel combination tested here.
        """
        x = rng.standard_normal((B, 1, 1, 4)).astype("float32")
        layer = ConvBlock(filters=6, kernel_size=3, padding="same", name="degenerate")
        out = layer(x)
        assert tuple(out.shape) == (B, 1, 1, 6)
        assert bool(keras.ops.all(keras.ops.isfinite(out)))


# ---------------------------------------------------------------------
# dtype policy (float32 control + mixed_float16 + float64)
# ---------------------------------------------------------------------

class TestDtypePolicy:
    def test_forward_pass_is_finite_under_every_policy(self, dtype_policy, rng):
        x = rng.standard_normal((B, 8, 8, 4)).astype("float32")
        layer = ConvBlock(filters=6, kernel_size=3, name=f"dtype_{dtype_policy}")
        out = layer(x, training=False)
        out_np = keras.ops.convert_to_numpy(out)
        assert np.isfinite(out_np).all()
        assert (layer.compute_dtype != layer.variable_dtype) == (dtype_policy == "mixed_float16")


# ---------------------------------------------------------------------
# XLA (jit_compile=True) versus eager
# ---------------------------------------------------------------------

class TestXlaVersusEager:
    def test_jit_compiled_output_matches_eager(self, rng):
        x = rng.standard_normal((B, 8, 8, 4)).astype("float32")
        layer = ConvBlock(filters=6, kernel_size=3, name="jit_conv_block")
        y_eager = keras.ops.convert_to_numpy(layer(x, training=False))

        @tf.function(jit_compile=True)
        def compiled(t):
            return layer(t, training=False)

        y_jit = np.asarray(compiled(tf.constant(x)))

        assert y_jit.shape == y_eager.shape
        assert np.all(np.isfinite(y_jit))
        peak = float(np.max(np.abs(y_eager)))
        assert peak > 0.0, "output is degenerate, comparison is vacuous"
        # Conv (3x3x4 = 36 taps) into BatchNorm into ReLU: bound conservatively
        # on float32 eps scaled by the accumulation length and output magnitude.
        atol = 36 * float(np.finfo(np.float32).eps) * max(1.0, peak)
        np.testing.assert_allclose(y_jit, y_eager, rtol=0.0, atol=atol)


# ---------------------------------------------------------------------
# .keras round trip at rtol=0
# ---------------------------------------------------------------------

class TestKerasRoundTripExact:
    def test_round_trip_is_bit_identical(self, tmp_path, rng):
        x = rng.standard_normal((B, 8, 8, 4)).astype("float32")
        inp = keras.Input(shape=(8, 8, 4))
        out = ConvBlock(filters=6, kernel_size=3, name="exact_rt")(inp)
        model = keras.Model(inp, out)
        y0 = model(x, training=False)

        path = os.path.join(tmp_path, "exact_rt.keras")
        model.save(path)
        loaded = keras.models.load_model(path)

        # Weight comparison FIRST, before the loaded model has been called.
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
