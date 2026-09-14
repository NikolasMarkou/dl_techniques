"""Tests for BottleneckBlock (relocated from test_standard_blocks.py's TestBottleneckBlock).

Hardened per plan-2026-09-14T165315-47f9d575 step 10 to match the
`test_conv_blocks/` precedent's rigor: exact (`rtol=0, atol=0`) `.keras` round
trip, pre-call weight values, build-parity / no-sub-layer-built assertions for
`use_projection=False`, a per-variable gradient-flow assertion, `isfinite` on
every forward pass, a degenerate (1x1) spatial-extent sweep, dtype-policy arms
with a float32 control, and XLA-vs-eager agreement.
"""

import os
import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.conv_blocks.bottleneck_block import BottleneckBlock

# The XLA-vs-eager tolerance below was derived CPU-only; on a TF32-capable GPU
# the tensor-core matmul path misses it by ~50x (measured, review-iter-1.md
# concern 1). Opt into the module-scoped disable/restore fixture rather than
# widening the tolerance.
pytestmark = pytest.mark.usefixtures("tf32_disabled")

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


class TestBottleneckBlock:
    def test_forward_and_shape(self):
        x = np.random.default_rng(0).standard_normal((B, 8, 8, 32)).astype("float32")
        layer = BottleneckBlock(filters=8)
        out = layer(x)
        assert tuple(out.shape) == (B, 8, 8, 32)
        assert bool(keras.ops.all(keras.ops.isfinite(out)))

    def test_serialization(self, tmp_path):
        x = np.random.default_rng(0).standard_normal((B, 8, 8, 32)).astype("float32")
        _roundtrip(BottleneckBlock(filters=8, name="bottleneck"), (8, 8, 32), x, "bottleneck", tmp_path)


class TestInvalidArgs:
    @pytest.mark.parametrize("ctor", [
        lambda: BottleneckBlock(filters=0),
    ])
    def test_invalid_args_raise(self, ctor):
        with pytest.raises(ValueError):
            ctor()


# ---------------------------------------------------------------------
# Weight values BEFORE the first call (construction + explicit build())
# ---------------------------------------------------------------------

class TestWeightsBeforeFirstCall:
    def test_batch_norm_weights_are_exact_before_any_call(self):
        """`build()` alone (never `call()`) must populate exact initial
        BatchNorm values: gamma=1, beta=0, moving_mean=0, moving_variance=1.
        """
        layer = BottleneckBlock(filters=8, name="preweights")
        layer.build((B, 8, 8, 32))

        for bn in (layer.bn1, layer.bn2, layer.bn3):
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(bn.gamma), 1.0, rtol=0.0, atol=0.0
            )
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(bn.beta), 0.0, rtol=0.0, atol=0.0
            )
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(bn.moving_mean), 0.0, rtol=0.0, atol=0.0
            )
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(bn.moving_variance), 1.0, rtol=0.0, atol=0.0
            )
        assert layer.conv1.kernel.shape == (1, 1, 32, 8)
        assert layer.conv2.kernel.shape == (3, 3, 8, 8)
        assert layer.conv3.kernel.shape == (1, 1, 8, 32)  # filters * expansion(4) = 32


# ---------------------------------------------------------------------
# Build parity + no-sub-layer-built assertions for use_projection=False
# ---------------------------------------------------------------------

class TestNoProjectionBranchBuildsNothing:
    def test_shortcut_conv_and_bn_are_none_when_no_projection(self):
        layer = BottleneckBlock(filters=8, use_projection=False, name="no_proj")
        layer.build((B, 8, 8, 32))

        assert layer.shortcut_conv is None
        assert layer.shortcut_bn is None
        tracked = [v for v in vars(layer).values() if isinstance(v, keras.layers.Layer)]
        shortcut_convs = [
            t for t in tracked
            if isinstance(t, keras.layers.Conv2D) and "shortcut" in t.name
        ]
        assert not shortcut_convs, "a shortcut Conv2D survived despite use_projection=False"


# ---------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------

class TestGradientFlow:
    def test_every_trainable_weight_receives_a_gradient(self, rng):
        layer = BottleneckBlock(filters=8, use_projection=True, stride=2, name="grad")
        x = tf.constant(rng.standard_normal((B, 8, 8, 32)).astype("float32"))

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
        """0-length spatial extent is not constructible (every conv in the
        block requires >= 1 spatial position); 1x1 with 'same' padding and
        stride 1 is the smallest valid degenerate case.
        """
        x = rng.standard_normal((B, 1, 1, 32)).astype("float32")
        layer = BottleneckBlock(filters=8, name="degenerate")
        out = layer(x)
        assert tuple(out.shape) == (B, 1, 1, 32)
        assert bool(keras.ops.all(keras.ops.isfinite(out)))


# ---------------------------------------------------------------------
# dtype policy (float32 control + mixed_float16 + float64)
# ---------------------------------------------------------------------

class TestDtypePolicy:
    def test_forward_pass_is_finite_under_every_policy(self, dtype_policy, rng):
        x = rng.standard_normal((B, 8, 8, 32)).astype("float32")
        layer = BottleneckBlock(filters=8, name=f"dtype_{dtype_policy}")
        out = layer(x, training=False)
        out_np = keras.ops.convert_to_numpy(out)
        assert np.isfinite(out_np).all()
        assert (layer.compute_dtype != layer.variable_dtype) == (dtype_policy == "mixed_float16")


# ---------------------------------------------------------------------
# XLA (jit_compile=True) versus eager
# ---------------------------------------------------------------------

class TestXlaVersusEager:
    def test_jit_compiled_output_matches_eager(self, rng):
        x = rng.standard_normal((B, 8, 8, 32)).astype("float32")
        layer = BottleneckBlock(filters=8, name="jit_bottleneck_block")
        y_eager = keras.ops.convert_to_numpy(layer(x, training=False))

        @tf.function(jit_compile=True)
        def compiled(t):
            return layer(t, training=False)

        y_jit = np.asarray(compiled(tf.constant(x)))

        assert y_jit.shape == y_eager.shape
        assert np.all(np.isfinite(y_jit))
        peak = float(np.max(np.abs(y_eager)))
        assert peak > 0.0, "output is degenerate, comparison is vacuous"
        # 1x1x32 (32 taps) + 3x3x8 (72 taps) + 1x1x8 (8 taps) chained.
        n_accumulations = 32 + 72 + 8
        atol = n_accumulations * float(np.finfo(np.float32).eps) * max(1.0, peak)
        np.testing.assert_allclose(y_jit, y_eager, rtol=0.0, atol=atol)


# ---------------------------------------------------------------------
# .keras round trip at rtol=0
# ---------------------------------------------------------------------

class TestKerasRoundTripExact:
    def test_round_trip_is_bit_identical(self, tmp_path, rng):
        x = rng.standard_normal((B, 8, 8, 32)).astype("float32")
        inp = keras.Input(shape=(8, 8, 32))
        out = BottleneckBlock(filters=8, use_projection=True, name="exact_rt")(inp)
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
