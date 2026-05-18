"""
Refined comprehensive test suite for the BandRMS layer.

Mirrors the Canonical-B test layout used by `test_adaptive_band_rms.py` and
extends the previously-thin 10-fn test module to canonical rigor (~35 tests
across `TestBandRMS`, `TestBandRMSIntegration`, `TestBandRMSEdgeCases`).

BandRMS contract:
- output = (x / RMS) * scale
- RMS = max(sqrt(mean(x^2) + eps), eps)
- scale = (1 - alpha) + alpha * sigmoid(5 * band_param) ∈ [1 - alpha, 1]
- band_param is a single trainable scalar, shape ()
- mixed_float16: inputs are cast to fp32 internally; band_param is read+cast to
  fp32 explicitly (DECISION plan_2026-05-14_3764496e/D-002 at band_rms.py:266).
"""

import os
import tempfile
from typing import Any, Dict, Tuple

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.layers.norms.band_rms import BandRMS
from dl_techniques.utils.logger import logger


# =============================================================================
# TestBandRMS
# =============================================================================


class TestBandRMS:
    """Comprehensive test suite for the BandRMS layer."""

    # -------------------------------------------------------------------------
    # FIXTURES
    # -------------------------------------------------------------------------

    @pytest.fixture
    def input_tensor_2d(self) -> np.ndarray:
        """2D input tensor (batch, features) for dense scenarios."""
        np.random.seed(42)
        return np.random.randn(4, 64).astype(np.float32)

    @pytest.fixture
    def input_tensor_3d(self) -> np.ndarray:
        """3D input tensor (batch, seq, features) for transformer scenarios."""
        np.random.seed(42)
        return np.random.randn(4, 20, 32).astype(np.float32)

    @pytest.fixture
    def input_tensor_4d(self) -> np.ndarray:
        """4D input tensor (batch, H, W, C) for convolutional scenarios."""
        np.random.seed(42)
        return np.random.randn(4, 16, 16, 32).astype(np.float32)

    @pytest.fixture
    def input_tensor_5d(self) -> np.ndarray:
        """5D input tensor for 3D-convolutional scenarios."""
        np.random.seed(42)
        return np.random.randn(2, 4, 8, 8, 16).astype(np.float32)

    @pytest.fixture
    def default_params(self) -> Dict[str, Any]:
        return {
            "max_band_width": 0.1,
            "axis": -1,
            "epsilon": 1e-7,
            "band_initializer": "zeros",
        }

    @pytest.fixture
    def custom_params(self) -> Dict[str, Any]:
        return {
            "max_band_width": 0.2,
            "axis": 1,
            "epsilon": 1e-6,
            "band_initializer": "zeros",
            "band_regularizer": keras.regularizers.L2(1e-4),
        }

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------

    def test_initialization_defaults(self):
        layer = BandRMS()
        assert layer.max_band_width == 0.1
        assert layer.axis == -1
        assert layer.epsilon == 1e-7
        assert isinstance(layer.band_initializer, keras.initializers.Zeros)
        # When None is passed, BandRMS substitutes L2(1e-5).
        assert layer.band_regularizer is not None
        assert layer.band_param is None  # created in build()

    def test_initialization_custom(self, custom_params):
        layer = BandRMS(**custom_params)
        assert layer.max_band_width == custom_params["max_band_width"]
        assert layer.axis == custom_params["axis"]
        assert layer.epsilon == custom_params["epsilon"]
        assert isinstance(layer.band_initializer, keras.initializers.Zeros)
        assert layer.band_regularizer is custom_params["band_regularizer"]

    @pytest.mark.parametrize("bad_alpha", [-0.1, 0.0, 1.0, 1.5])
    def test_invalid_max_band_width(self, bad_alpha):
        with pytest.raises(ValueError, match="max_band_width must be between 0 and 1"):
            BandRMS(max_band_width=bad_alpha)

    @pytest.mark.parametrize("bad_eps", [-1.0, -1e-7, 0.0])
    def test_invalid_epsilon(self, bad_eps):
        with pytest.raises(ValueError, match="epsilon must be positive"):
            BandRMS(epsilon=bad_eps)

    # -------------------------------------------------------------------------
    # BUILD / SHAPE
    # -------------------------------------------------------------------------

    def test_build_2d(self, input_tensor_2d):
        layer = BandRMS()
        _ = layer(input_tensor_2d)
        assert layer.built is True
        assert layer.band_param is not None
        assert tuple(layer.band_param.shape) == ()

    def test_build_3d(self, input_tensor_3d):
        layer = BandRMS()
        _ = layer(input_tensor_3d)
        assert layer.built is True
        assert tuple(layer.band_param.shape) == ()

    def test_build_4d(self, input_tensor_4d):
        layer = BandRMS()
        _ = layer(input_tensor_4d)
        assert layer.built is True
        assert tuple(layer.band_param.shape) == ()

    def test_output_shape_2d(self, input_tensor_2d):
        layer = BandRMS()
        out = layer(input_tensor_2d)
        assert tuple(out.shape) == input_tensor_2d.shape

    def test_output_shape_3d(self, input_tensor_3d):
        layer = BandRMS()
        out = layer(input_tensor_3d)
        assert tuple(out.shape) == input_tensor_3d.shape

    def test_output_shape_4d(self, input_tensor_4d):
        layer = BandRMS()
        out = layer(input_tensor_4d)
        assert tuple(out.shape) == input_tensor_4d.shape

    def test_compute_output_shape(self, input_tensor_3d):
        layer = BandRMS()
        assert layer.compute_output_shape(input_tensor_3d.shape) == input_tensor_3d.shape

    def test_single_scalar_band_param(self, input_tensor_3d):
        """Exactly ONE trainable scalar parameter regardless of feature dim."""
        layer = BandRMS()
        _ = layer(input_tensor_3d)
        trainable_count = sum(int(np.prod(w.shape)) for w in layer.trainable_weights)
        assert trainable_count == 1

    def test_param_count_independent_of_feature_dim(self):
        for d in (8, 64, 512, 4096):
            layer = BandRMS()
            _ = layer(tf.zeros((2, d), dtype=tf.float32))
            count = sum(int(np.prod(w.shape)) for w in layer.trainable_weights)
            assert count == 1, f"d={d} should yield 1 param, got {count}"

    # -------------------------------------------------------------------------
    # NORMALIZATION CORRECTNESS
    # -------------------------------------------------------------------------

    def test_normalization_bound_invariant_at_init(self, input_tensor_2d):
        """
        At init band_param=0 → sigmoid(0)=0.5 → scale = (1-α) + α*0.5 = 1 - α/2.
        Per-sample RMS must lie in [1-α, 1].
        """
        alpha = 0.1
        layer = BandRMS(max_band_width=alpha, epsilon=1e-7, band_initializer="zeros")
        out = layer(input_tensor_2d)
        out_np = keras.ops.convert_to_numpy(out)
        per_sample_rms = np.sqrt(np.mean(out_np ** 2, axis=-1))
        assert np.all(per_sample_rms >= (1.0 - alpha) - 1e-4)
        assert np.all(per_sample_rms <= 1.0 + 1e-4)

    @pytest.mark.parametrize("band_param_value", [-5.0, 0.0, 5.0])
    def test_normalization_bound_invariant_after_assign(
        self, input_tensor_2d, band_param_value: float
    ):
        """
        After assigning extreme values to band_param (sigmoid -> {0, 0.5, 1}),
        per-sample RMS must still lie in [1-α, 1].
        """
        alpha = 0.2
        layer = BandRMS(max_band_width=alpha, epsilon=1e-7, band_initializer="zeros")
        _ = layer(input_tensor_2d)  # build
        layer.band_param.assign(band_param_value)
        out = layer(input_tensor_2d)
        out_np = keras.ops.convert_to_numpy(out)
        per_sample_rms = np.sqrt(np.mean(out_np ** 2, axis=-1))
        assert np.all(per_sample_rms >= (1.0 - alpha) - 1e-4), (
            f"min={per_sample_rms.min()} below band for band_param={band_param_value}"
        )
        assert np.all(per_sample_rms <= 1.0 + 1e-4), (
            f"max={per_sample_rms.max()} above band for band_param={band_param_value}"
        )

    def test_mathematical_correctness_at_init(self, input_tensor_2d):
        """
        Closed-form check: output == (x / RMS) * (1 - α/2) at init.
        """
        alpha = 0.1
        eps = 1e-7
        layer = BandRMS(
            max_band_width=alpha,
            epsilon=eps,
            band_initializer="zeros",
        )
        out = keras.ops.convert_to_numpy(layer(input_tensor_2d))

        x = input_tensor_2d
        rms = np.maximum(
            np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps), eps
        )
        normalized = x / rms
        scale = (1.0 - alpha) + alpha * 0.5  # sigmoid(0)=0.5
        expected = normalized * scale

        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)

    def test_multi_axis_correctness(self, input_tensor_4d):
        """axis=(-2, -1) should normalize over the last two dims."""
        layer = BandRMS(axis=(-2, -1), max_band_width=0.2)
        out = layer(input_tensor_4d)
        out_np = keras.ops.convert_to_numpy(out)
        rms = np.sqrt(np.mean(out_np ** 2, axis=(-2, -1)))
        assert np.all(rms >= 0.8 - 1e-4)
        assert np.all(rms <= 1.0 + 1e-4)

    def test_training_vs_inference(self, input_tensor_3d):
        layer = BandRMS()
        train_out = keras.ops.convert_to_numpy(layer(input_tensor_3d, training=True))
        eval_out = keras.ops.convert_to_numpy(layer(input_tensor_3d, training=False))
        np.testing.assert_allclose(train_out, eval_out, rtol=1e-6, atol=1e-7)

    def test_deterministic(self, input_tensor_2d):
        layer = BandRMS()
        a = keras.ops.convert_to_numpy(layer(input_tensor_2d))
        b = keras.ops.convert_to_numpy(layer(input_tensor_2d))
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)

    # -------------------------------------------------------------------------
    # SERIALIZATION
    # -------------------------------------------------------------------------

    def test_get_config_completeness(self, custom_params):
        layer = BandRMS(**custom_params)
        cfg = layer.get_config()
        for key in (
            "max_band_width",
            "axis",
            "epsilon",
            "band_initializer",
            "band_regularizer",
        ):
            assert key in cfg
        assert cfg["max_band_width"] == custom_params["max_band_width"]
        assert cfg["axis"] == custom_params["axis"]
        assert cfg["epsilon"] == custom_params["epsilon"]

    def test_serialization_cycle_2d(self, input_tensor_2d):
        layer = BandRMS(max_band_width=0.15, epsilon=1e-6)
        out_before = keras.ops.convert_to_numpy(layer(input_tensor_2d))

        cfg = layer.get_config()
        restored = BandRMS.from_config(cfg)
        # Restored layer must produce same output (band_param=0 at init).
        out_after = keras.ops.convert_to_numpy(restored(input_tensor_2d))
        np.testing.assert_allclose(out_before, out_after, rtol=1e-5, atol=1e-6)

    def test_serialization_cycle_3d(self, input_tensor_3d):
        layer = BandRMS(max_band_width=0.2, axis=-1, epsilon=1e-7)
        out_before = keras.ops.convert_to_numpy(layer(input_tensor_3d))

        cfg = layer.get_config()
        restored = BandRMS.from_config(cfg)
        out_after = keras.ops.convert_to_numpy(restored(input_tensor_3d))
        np.testing.assert_allclose(out_before, out_after, rtol=1e-5, atol=1e-6)

    def test_save_load_keras_format(self, input_tensor_3d):
        inputs = keras.Input(shape=input_tensor_3d.shape[1:])
        outputs = BandRMS(max_band_width=0.1, epsilon=1e-7)(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        original = keras.ops.convert_to_numpy(model(input_tensor_3d, training=False))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "band_rms.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            reloaded = keras.ops.convert_to_numpy(loaded(input_tensor_3d, training=False))
        np.testing.assert_allclose(original, reloaded, rtol=1e-6, atol=1e-7)

    def test_save_load_weights(self, input_tensor_2d):
        inputs = keras.Input(shape=input_tensor_2d.shape[1:])
        outputs = BandRMS(max_band_width=0.1, epsilon=1e-7)(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        before = keras.ops.convert_to_numpy(model(input_tensor_2d, training=False))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "band_rms.weights.h5")
            model.save_weights(path)
            model.load_weights(path)
            after = keras.ops.convert_to_numpy(model(input_tensor_2d, training=False))
        np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-7)

    # -------------------------------------------------------------------------
    # GRADIENT FLOW / TRAINING
    # -------------------------------------------------------------------------

    def test_gradient_flow_band_param(self, input_tensor_3d):
        """Grad w.r.t. the scalar `band_param` must be non-None and non-zero."""
        layer = BandRMS(max_band_width=0.1)
        x = tf.constant(input_tensor_3d)
        with tf.GradientTape() as tape:
            out = layer(x, training=True)
            loss = ops.mean(out ** 2)
        grads = tape.gradient(loss, layer.trainable_variables)
        assert len(grads) == 1
        assert grads[0] is not None
        assert not bool(keras.ops.any(keras.ops.isnan(grads[0])))
        # The gradient should be measurably non-zero for non-degenerate input.
        assert float(keras.ops.abs(grads[0])) > 0.0

    def test_gradient_flow_through_input(self, input_tensor_2d):
        layer = BandRMS(max_band_width=0.1)
        var_x = tf.Variable(input_tensor_2d)
        with tf.GradientTape() as tape:
            out = layer(var_x, training=True)
            loss = ops.mean(out)
        g = tape.gradient(loss, var_x)
        assert g is not None
        assert not bool(keras.ops.any(keras.ops.isnan(g)))

    def test_band_param_updates_during_training(self, input_tensor_2d):
        layer = BandRMS(max_band_width=0.1)
        _ = layer(input_tensor_2d)
        initial = float(layer.band_param.numpy())
        optimizer = keras.optimizers.SGD(learning_rate=0.1)
        with tf.GradientTape() as tape:
            out = layer(input_tensor_2d, training=True)
            loss = ops.mean(out ** 2)
        grads = tape.gradient(loss, layer.trainable_variables)
        optimizer.apply_gradients(zip(grads, layer.trainable_variables))
        updated = float(layer.band_param.numpy())
        assert updated != initial

    # -------------------------------------------------------------------------
    # MIXED PRECISION (D-002 anchor at band_rms.py:266)
    # -------------------------------------------------------------------------

    def test_mixed_precision_compatibility(self, input_tensor_2d):
        """
        Under `mixed_float16`, `band_param` is auto-cast on read. The
        explicit fp32 cast at band_rms.py:266 prevents a crash when the
        multiply against fp32 `normalized` would otherwise dtype-mismatch.
        """
        original_policy = keras.mixed_precision.global_policy()
        keras.mixed_precision.set_global_policy("mixed_float16")
        try:
            layer = BandRMS(max_band_width=0.1, epsilon=1e-5)
            x_fp16 = tf.cast(input_tensor_2d, tf.float16)

            # Forward must succeed and be finite.
            out = layer(x_fp16)
            assert out.dtype == tf.float16
            assert not bool(keras.ops.any(keras.ops.isnan(out)))
            assert bool(keras.ops.all(keras.ops.isfinite(out)))

            # Backward must also be finite.
            with tf.GradientTape() as tape:
                tape.watch(x_fp16)
                out = layer(x_fp16, training=True)
                loss = ops.mean(ops.cast(out, "float32") ** 2)
            grads = tape.gradient(loss, layer.trainable_variables)
            assert grads[0] is not None
            assert not bool(keras.ops.any(keras.ops.isnan(grads[0])))
        finally:
            keras.mixed_precision.set_global_policy(original_policy)

    # -------------------------------------------------------------------------
    # REGULARIZATION
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "regularizer",
        [None, keras.regularizers.L2(1e-4), keras.regularizers.L1(1e-5)],
    )
    def test_band_regularizer_options(self, input_tensor_2d, regularizer):
        layer = BandRMS(max_band_width=0.1, band_regularizer=regularizer)
        out = layer(input_tensor_2d)
        assert tuple(out.shape) == input_tensor_2d.shape
        if regularizer is None:
            # BandRMS substitutes L2(1e-5) when None is passed.
            assert layer.band_regularizer is not None

    # -------------------------------------------------------------------------
    # DYNAMIC SHAPES / NUMERICS
    # -------------------------------------------------------------------------

    def test_dynamic_batch_size(self):
        inputs = keras.Input(shape=(10,))
        outputs = BandRMS(max_band_width=0.1)(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        for bs in (1, 4, 17, 32):
            x = tf.random.normal((bs, 10))
            y = model(x)
            assert int(y.shape[0]) == bs

    def test_zero_input(self):
        layer = BandRMS(max_band_width=0.1, epsilon=1e-6)
        out = layer(tf.zeros((2, 16), dtype=tf.float32))
        out_np = keras.ops.convert_to_numpy(out)
        assert not np.any(np.isnan(out_np))
        # All-zero input → RMS ~ epsilon → scaled to 0.
        assert np.allclose(out_np, 0.0, atol=1e-3)

    def test_constant_input(self):
        layer = BandRMS(max_band_width=0.1, epsilon=1e-7)
        # Constant non-zero input → RMS == |c|; output sample-wise constant within band.
        c = 3.5
        x = tf.constant(np.full((2, 16), c, dtype=np.float32))
        out = keras.ops.convert_to_numpy(layer(x))
        per_sample_rms = np.sqrt(np.mean(out ** 2, axis=-1))
        # band_param=0 → scale = 1 - alpha/2 = 0.95
        np.testing.assert_allclose(per_sample_rms, 0.95, atol=1e-4)

    def test_extreme_input_values(self):
        layer = BandRMS(max_band_width=0.1, epsilon=1e-6)
        for magnitude in (1e-8, 1e8):
            x = tf.constant(np.full((2, 32), magnitude, dtype=np.float32))
            out = keras.ops.convert_to_numpy(layer(x))
            assert not np.any(np.isnan(out))
            assert np.all(np.isfinite(out))


# =============================================================================
# TestBandRMSIntegration
# =============================================================================


class TestBandRMSIntegration:
    """Model-integration smoke tests."""

    @pytest.fixture
    def input_tensor_3d(self) -> np.ndarray:
        np.random.seed(0)
        return np.random.randn(8, 20, 32).astype(np.float32)

    def test_integration_with_layer_norm(self, input_tensor_3d):
        """LayerNorm → BandRMS chain runs end-to-end."""
        inputs = keras.Input(shape=input_tensor_3d.shape[1:])
        x = keras.layers.LayerNormalization()(inputs)
        x = BandRMS(max_band_width=0.1)(x)
        model = keras.Model(inputs=inputs, outputs=x)
        out = model(input_tensor_3d)
        assert tuple(out.shape) == input_tensor_3d.shape

    def test_integration_with_dense_stack(self):
        inputs = keras.Input(shape=(20, 10))
        x = keras.layers.Flatten()(inputs)
        x = keras.layers.Dense(16)(x)
        x = BandRMS(max_band_width=0.1)(x)
        x = keras.layers.Dense(8)(x)
        x = BandRMS(max_band_width=0.2)(x)
        outputs = keras.layers.Dense(1)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse")
        x_train = tf.random.normal((64, 20, 10))
        y_train = tf.random.normal((64, 1))
        hist = model.fit(x_train, y_train, epochs=3, batch_size=16, verbose=0)
        assert len(hist.history["loss"]) == 3
        # Should make some progress on a trivial regression task.
        assert hist.history["loss"][0] >= hist.history["loss"][-1] - 1e-3

    def test_transformer_integration(self, input_tensor_3d):
        """BandRMS used as a normalization stage inside a 2-block transformer."""
        d_model = input_tensor_3d.shape[-1]
        inputs = keras.Input(shape=input_tensor_3d.shape[1:])
        x = inputs
        for _ in range(2):
            # pre-norm attention sub-block
            n = BandRMS(max_band_width=0.1)(x)
            attn = keras.layers.MultiHeadAttention(num_heads=4, key_dim=d_model // 4)(n, n)
            x = keras.layers.Add()([x, attn])
            # pre-norm FFN sub-block
            n = BandRMS(max_band_width=0.1)(x)
            ffn = keras.layers.Dense(d_model * 2, activation="gelu")(n)
            ffn = keras.layers.Dense(d_model)(ffn)
            x = keras.layers.Add()([x, ffn])
        model = keras.Model(inputs=inputs, outputs=x)
        out = model(input_tensor_3d)
        assert tuple(out.shape) == input_tensor_3d.shape

    def test_higher_dim_inputs(self):
        """5D input is handled correctly along the default last axis."""
        x = tf.random.normal((2, 4, 8, 8, 16))
        layer = BandRMS(max_band_width=0.1)
        out = keras.ops.convert_to_numpy(layer(x))
        assert out.shape == (2, 4, 8, 8, 16)
        per_sample_rms = np.sqrt(np.mean(out ** 2, axis=-1))
        # band_param=0 → scale = 0.95 → per-sample RMS in [0.9, 1.0]
        assert np.all(per_sample_rms >= 0.9 - 1e-4)
        assert np.all(per_sample_rms <= 1.0 + 1e-4)


# =============================================================================
# TestBandRMSEdgeCases
# =============================================================================


class TestBandRMSEdgeCases:
    """Edge-case behaviors and corner inputs."""

    def test_single_element_tensor(self):
        """Single feature dim: RMS=|x|, output = sign(x) * scale."""
        layer = BandRMS(max_band_width=0.1, epsilon=1e-7)
        x = tf.constant([[3.0], [-2.0]], dtype=tf.float32)
        out = keras.ops.convert_to_numpy(layer(x))
        # Single-element axis: RMS = |value|, so x/RMS = ±1. scale = 0.95.
        np.testing.assert_allclose(np.abs(out), 0.95, atol=1e-4)
        assert out[0, 0] > 0
        assert out[1, 0] < 0

    def test_very_large_tensor(self):
        """Memory-stress smoke: 1M elements, must complete without OOM."""
        x = tf.random.normal((4, 256 * 1024))
        layer = BandRMS(max_band_width=0.1)
        out = layer(x)
        assert tuple(out.shape) == (4, 256 * 1024)

    def test_negative_axis(self):
        """axis=-1 (default) must equal axis=2 for rank-3 input."""
        x = tf.random.normal((2, 4, 8))
        layer_neg = BandRMS(max_band_width=0.1, axis=-1)
        layer_pos = BandRMS(max_band_width=0.1, axis=2)
        out_neg = keras.ops.convert_to_numpy(layer_neg(x))
        out_pos = keras.ops.convert_to_numpy(layer_pos(x))
        np.testing.assert_allclose(out_neg, out_pos, rtol=1e-6, atol=1e-7)

    def test_band_param_initializer_random(self):
        """Non-zero band initializer still yields per-sample RMS in band."""
        layer = BandRMS(
            max_band_width=0.1,
            band_initializer=keras.initializers.RandomNormal(stddev=2.0, seed=7),
        )
        x = tf.random.normal((4, 64))
        out = keras.ops.convert_to_numpy(layer(x))
        per_sample_rms = np.sqrt(np.mean(out ** 2, axis=-1))
        assert np.all(per_sample_rms >= 0.9 - 1e-4)
        assert np.all(per_sample_rms <= 1.0 + 1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
