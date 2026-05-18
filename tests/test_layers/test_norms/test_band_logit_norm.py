"""
Comprehensive test suite for the BandLogitNorm layer.

BandLogitNorm contract (distinct from other band-family variants):
- Output L2 norm (NOT RMS norm) lies in [1 - max_band_width, 1].
- Pipeline: x_norm = x / ||x||_2; ln = LayerNorm(||x||_2); s = tanh(4 * ln);
  scale = (1 - alpha) + alpha * (s + 1) / 2; output = x_norm * scale.
- The inner LayerNormalization is applied to a singleton last-axis (the L2
  scalar), so before training the gamma/beta are at defaults (1, 0) and
  LayerNorm of a singleton dim is exactly 0 → scale = 1 - alpha/2 deterministically.
- This variant is designed for classification logits; on residual-stream
  contexts it is off-label (handled at the report-level, not in the layer).

Layout mirrors `test_band_rms.py` / `test_adaptive_band_rms.py` (Canonical-B).
"""

import os
import tempfile
from typing import Any, Dict

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.layers.norms.band_logit_norm import BandLogitNorm
from dl_techniques.utils.logger import logger


# =============================================================================
# TestBandLogitNorm
# =============================================================================


class TestBandLogitNorm:
    """Comprehensive test suite for the BandLogitNorm layer."""

    # -------------------------------------------------------------------------
    # FIXTURES
    # -------------------------------------------------------------------------

    @pytest.fixture
    def input_tensor_2d(self) -> np.ndarray:
        np.random.seed(42)
        return np.random.randn(4, 64).astype(np.float32)

    @pytest.fixture
    def input_tensor_3d(self) -> np.ndarray:
        np.random.seed(42)
        return np.random.randn(4, 16, 32).astype(np.float32)

    @pytest.fixture
    def input_tensor_4d(self) -> np.ndarray:
        np.random.seed(42)
        return np.random.randn(2, 8, 8, 32).astype(np.float32)

    @pytest.fixture
    def default_params(self) -> Dict[str, Any]:
        return {"max_band_width": 0.01, "axis": -1, "epsilon": 1e-7}

    @pytest.fixture
    def custom_params(self) -> Dict[str, Any]:
        return {"max_band_width": 0.1, "axis": -1, "epsilon": 1e-6}

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------

    def test_initialization_defaults(self):
        layer = BandLogitNorm()
        # Default alpha is 0.01 (tighter than band_rms default 0.1).
        assert layer.max_band_width == 0.01
        assert layer.axis == -1
        assert layer.epsilon == 1e-7
        # Inner LN is created in build()
        assert layer.norm is None

    def test_initialization_custom(self, custom_params):
        layer = BandLogitNorm(**custom_params)
        assert layer.max_band_width == custom_params["max_band_width"]
        assert layer.axis == custom_params["axis"]
        assert layer.epsilon == custom_params["epsilon"]

    @pytest.mark.parametrize("bad_alpha", [-0.1, 0.0, 1.0, 1.5])
    def test_invalid_max_band_width(self, bad_alpha):
        with pytest.raises(ValueError, match="max_band_width must be between 0 and 1"):
            BandLogitNorm(max_band_width=bad_alpha)

    @pytest.mark.parametrize("bad_eps", [-1.0, -1e-7, 0.0])
    def test_invalid_epsilon(self, bad_eps):
        with pytest.raises(ValueError, match="epsilon must be positive"):
            BandLogitNorm(epsilon=bad_eps)

    # -------------------------------------------------------------------------
    # BUILD / SUBLAYER CONTRACT
    # -------------------------------------------------------------------------

    def test_build_2d(self, input_tensor_2d):
        layer = BandLogitNorm()
        _ = layer(input_tensor_2d)
        assert layer.built is True
        assert layer.norm is not None
        # Inner LN must be built
        assert layer.norm.built is True

    def test_build_3d(self, input_tensor_3d):
        layer = BandLogitNorm()
        _ = layer(input_tensor_3d)
        assert layer.built is True
        assert layer.norm.built is True

    def test_inner_layer_norm_variables(self, input_tensor_2d):
        """Inner LayerNormalization has the expected gamma/beta with shape (1,)."""
        layer = BandLogitNorm()
        _ = layer(input_tensor_2d)
        ln = layer.norm
        # Inner LN was built on shape [..., 1] (the per-sample L2 scalar)
        # so its gamma/beta have shape (1,)
        gamma = ln.gamma
        beta = ln.beta
        assert tuple(gamma.shape) == (1,)
        assert tuple(beta.shape) == (1,)
        # Defaults: gamma=1, beta=0
        np.testing.assert_allclose(keras.ops.convert_to_numpy(gamma), 1.0, atol=1e-7)
        np.testing.assert_allclose(keras.ops.convert_to_numpy(beta), 0.0, atol=1e-7)

    def test_inner_ln_name(self, input_tensor_2d):
        layer = BandLogitNorm(name="my_blogit")
        _ = layer(input_tensor_2d)
        assert layer.norm.name == "my_blogit_layer_norm"

    # -------------------------------------------------------------------------
    # SHAPE PRESERVATION
    # -------------------------------------------------------------------------

    def test_output_shape_2d(self, input_tensor_2d):
        out = BandLogitNorm()(input_tensor_2d)
        assert tuple(out.shape) == input_tensor_2d.shape

    def test_output_shape_3d(self, input_tensor_3d):
        out = BandLogitNorm()(input_tensor_3d)
        assert tuple(out.shape) == input_tensor_3d.shape

    def test_output_shape_4d(self, input_tensor_4d):
        out = BandLogitNorm()(input_tensor_4d)
        assert tuple(out.shape) == input_tensor_4d.shape

    def test_compute_output_shape(self, input_tensor_3d):
        layer = BandLogitNorm()
        assert layer.compute_output_shape(input_tensor_3d.shape) == input_tensor_3d.shape

    # -------------------------------------------------------------------------
    # L2-NOT-RMS INVARIANT (critical test pinning the variant's contract)
    # -------------------------------------------------------------------------

    def test_l2_norm_in_band_not_rms(self, input_tensor_2d):
        """
        Critical: the variant constrains L2 norm to [1-alpha, 1], NOT RMS.
        At default init (gamma=1, beta=0): LN(singleton) = 0 → scale = 1 - alpha/2.
        L2(output) must equal 1 - alpha/2 within fp32 precision.
        """
        alpha = 0.1
        layer = BandLogitNorm(max_band_width=alpha, epsilon=1e-7)
        out = keras.ops.convert_to_numpy(layer(input_tensor_2d))
        l2 = np.sqrt(np.sum(out ** 2, axis=-1))
        expected_l2 = 1.0 - alpha / 2.0
        np.testing.assert_allclose(l2, expected_l2, atol=1e-5)
        # Cross-check: per-sample RMS is *not* in [1-alpha, 1] — it scales as ~ L2/sqrt(D).
        per_sample_rms = np.sqrt(np.mean(out ** 2, axis=-1))
        D = input_tensor_2d.shape[-1]
        expected_rms = expected_l2 / np.sqrt(D)
        np.testing.assert_allclose(per_sample_rms, expected_rms, rtol=1e-3, atol=1e-5)
        # And the RMS is NOT in [1-alpha, 1] for D > 1.
        assert np.all(per_sample_rms < (1.0 - alpha)), (
            f"For D={D}, RMS={per_sample_rms} must be << 1-alpha={1-alpha}; "
            f"if this assertion fires, the layer's L2-not-RMS invariant has silently "
            f"changed (someone may have swapped sum for mean)."
        )

    def test_l2_norm_in_band_after_training_inner_ln(self, input_tensor_2d):
        """
        After training the inner LN's gamma/beta away from their defaults,
        L2 must still lie in [1 - alpha, 1].
        """
        alpha = 0.1
        layer = BandLogitNorm(max_band_width=alpha, epsilon=1e-7)
        _ = layer(input_tensor_2d)
        # Push gamma and beta to a few extreme settings; the tanh keeps us in band.
        for gamma_val, beta_val in [(0.0, 0.0), (10.0, -5.0), (-7.0, 3.0), (50.0, 0.0)]:
            layer.norm.gamma.assign([gamma_val])
            layer.norm.beta.assign([beta_val])
            out = keras.ops.convert_to_numpy(layer(input_tensor_2d))
            l2 = np.sqrt(np.sum(out ** 2, axis=-1))
            assert np.all(l2 >= (1.0 - alpha) - 1e-4), (
                f"L2 below band at gamma={gamma_val}, beta={beta_val}: min={l2.min()}"
            )
            assert np.all(l2 <= 1.0 + 1e-4), (
                f"L2 above band at gamma={gamma_val}, beta={beta_val}: max={l2.max()}"
            )

    def test_dimension_dependence(self):
        """
        L2(output) ∈ [1-alpha, 1] regardless of D, but equivalent RMS scales as 1/sqrt(D).
        """
        alpha = 0.01
        for d in (4, 16, 256, 1024):
            np.random.seed(d)
            x = np.random.randn(4, d).astype(np.float32)
            out = keras.ops.convert_to_numpy(BandLogitNorm(max_band_width=alpha)(x))
            l2 = np.sqrt(np.sum(out ** 2, axis=-1))
            np.testing.assert_allclose(l2, 1.0 - alpha / 2.0, atol=1e-5)
            rms = np.sqrt(np.mean(out ** 2, axis=-1))
            np.testing.assert_allclose(
                rms, (1.0 - alpha / 2.0) / np.sqrt(d), rtol=1e-3, atol=1e-5
            )

    def test_tanh_saturation_at_extreme_layer_norms(self, input_tensor_2d):
        """
        Drive the inner LN output via gamma/beta to ±large values → tanh saturates →
        scale equals exactly the band endpoints (1 - alpha or 1).
        """
        alpha = 0.1
        layer = BandLogitNorm(max_band_width=alpha)
        _ = layer(input_tensor_2d)

        # Saturate positive: gamma=0 but beta=+inf-like → after LN we get +large
        # but LN's gamma=0 zeroes out the signal. Instead manipulate beta which
        # is added AFTER multiplication by gamma. With gamma=0, output=beta.
        layer.norm.gamma.assign([0.0])
        layer.norm.beta.assign([10.0])  # large positive
        out_high = keras.ops.convert_to_numpy(layer(input_tensor_2d))
        l2_high = np.sqrt(np.sum(out_high ** 2, axis=-1))
        # tanh(4 * 10) ~ +1 → scale = (1 + 1)/2 * alpha + (1 - alpha) = 1
        np.testing.assert_allclose(l2_high, 1.0, atol=1e-4)

        layer.norm.beta.assign([-10.0])  # large negative
        out_low = keras.ops.convert_to_numpy(layer(input_tensor_2d))
        l2_low = np.sqrt(np.sum(out_low ** 2, axis=-1))
        # tanh(-40) ~ -1 → scale = 0 * alpha + (1 - alpha) = 1 - alpha
        np.testing.assert_allclose(l2_low, 1.0 - alpha, atol=1e-4)

    # -------------------------------------------------------------------------
    # TRAINING / DETERMINISM
    # -------------------------------------------------------------------------

    def test_training_vs_inference(self, input_tensor_3d):
        layer = BandLogitNorm()
        train_out = keras.ops.convert_to_numpy(layer(input_tensor_3d, training=True))
        eval_out = keras.ops.convert_to_numpy(layer(input_tensor_3d, training=False))
        # LayerNormalization is not stateful → train and eval must match.
        np.testing.assert_allclose(train_out, eval_out, rtol=1e-6, atol=1e-7)

    def test_deterministic(self, input_tensor_2d):
        layer = BandLogitNorm()
        a = keras.ops.convert_to_numpy(layer(input_tensor_2d))
        b = keras.ops.convert_to_numpy(layer(input_tensor_2d))
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)

    # -------------------------------------------------------------------------
    # SERIALIZATION
    # -------------------------------------------------------------------------

    def test_get_config_completeness(self, custom_params):
        layer = BandLogitNorm(**custom_params)
        cfg = layer.get_config()
        for key in ("max_band_width", "axis", "epsilon"):
            assert key in cfg
        assert cfg["max_band_width"] == custom_params["max_band_width"]
        assert cfg["axis"] == custom_params["axis"]
        assert cfg["epsilon"] == custom_params["epsilon"]

    def test_from_config_explicit_method(self):
        """BandLogitNorm has an explicit from_config classmethod — verify it works."""
        original = BandLogitNorm(max_band_width=0.2, axis=-1, epsilon=1e-6)
        cfg = original.get_config()
        restored = BandLogitNorm.from_config(cfg)
        assert restored.max_band_width == original.max_band_width
        assert restored.axis == original.axis
        assert restored.epsilon == original.epsilon

    def test_serialization_cycle_2d(self, input_tensor_2d):
        original = BandLogitNorm(max_band_width=0.1, epsilon=1e-7)
        out_before = keras.ops.convert_to_numpy(original(input_tensor_2d))

        cfg = original.get_config()
        restored = BandLogitNorm.from_config(cfg)
        out_after = keras.ops.convert_to_numpy(restored(input_tensor_2d))
        np.testing.assert_allclose(out_before, out_after, rtol=1e-5, atol=1e-6)

    def test_serialization_cycle_3d(self, input_tensor_3d):
        original = BandLogitNorm(max_band_width=0.05, epsilon=1e-7)
        out_before = keras.ops.convert_to_numpy(original(input_tensor_3d))

        cfg = original.get_config()
        restored = BandLogitNorm.from_config(cfg)
        out_after = keras.ops.convert_to_numpy(restored(input_tensor_3d))
        np.testing.assert_allclose(out_before, out_after, rtol=1e-5, atol=1e-6)

    def test_save_load_keras_format(self, input_tensor_2d):
        inputs = keras.Input(shape=input_tensor_2d.shape[1:])
        outputs = BandLogitNorm(max_band_width=0.1)(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        original = keras.ops.convert_to_numpy(model(input_tensor_2d, training=False))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "band_logit_norm.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            reloaded = keras.ops.convert_to_numpy(loaded(input_tensor_2d, training=False))
        np.testing.assert_allclose(original, reloaded, rtol=1e-6, atol=1e-7)

    def test_save_load_weights(self, input_tensor_2d):
        inputs = keras.Input(shape=input_tensor_2d.shape[1:])
        outputs = BandLogitNorm(max_band_width=0.1)(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        before = keras.ops.convert_to_numpy(model(input_tensor_2d, training=False))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "blogit.weights.h5")
            model.save_weights(path)
            model.load_weights(path)
            after = keras.ops.convert_to_numpy(model(input_tensor_2d, training=False))
        np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-7)

    # -------------------------------------------------------------------------
    # GRADIENT FLOW
    # -------------------------------------------------------------------------

    def test_gradient_flow_through_inner_ln(self, input_tensor_2d):
        """
        Gradient must flow into the inner LN's gamma/beta — the only trainable
        weights of the variant.
        """
        layer = BandLogitNorm(max_band_width=0.1)
        x = tf.constant(input_tensor_2d)
        with tf.GradientTape() as tape:
            out = layer(x, training=True)
            loss = ops.mean(out ** 2)
        grads = tape.gradient(loss, layer.trainable_variables)
        assert len(grads) == 2  # gamma and beta of inner LN
        for g in grads:
            assert g is not None
            assert not bool(keras.ops.any(keras.ops.isnan(g)))

    def test_gradient_flow_through_input(self, input_tensor_2d):
        layer = BandLogitNorm(max_band_width=0.1)
        var_x = tf.Variable(input_tensor_2d)
        with tf.GradientTape() as tape:
            out = layer(var_x, training=True)
            loss = ops.mean(out)
        g = tape.gradient(loss, var_x)
        assert g is not None
        assert not bool(keras.ops.any(keras.ops.isnan(g)))

    def test_inner_ln_trains(self, input_tensor_2d):
        """One SGD step must change the inner LN's beta away from 0."""
        layer = BandLogitNorm(max_band_width=0.1)
        _ = layer(input_tensor_2d)
        initial_beta = float(layer.norm.beta.numpy()[0])
        optimizer = keras.optimizers.SGD(learning_rate=0.5)
        with tf.GradientTape() as tape:
            out = layer(input_tensor_2d, training=True)
            # Loss biased to one side of the band so beta moves.
            loss = ops.mean((out - 1.0) ** 2)
        grads = tape.gradient(loss, layer.trainable_variables)
        optimizer.apply_gradients(zip(grads, layer.trainable_variables))
        updated_beta = float(layer.norm.beta.numpy()[0])
        assert updated_beta != initial_beta

    # -------------------------------------------------------------------------
    # MIXED PRECISION
    # -------------------------------------------------------------------------

    def test_mixed_precision_compatibility(self, input_tensor_2d):
        """
        BandLogitNorm has NO internal fp32 cast (unlike BandRMS). Verify
        forward + backward complete without NaN under mixed_float16 policy.
        """
        original_policy = keras.mixed_precision.global_policy()
        keras.mixed_precision.set_global_policy("mixed_float16")
        try:
            layer = BandLogitNorm(max_band_width=0.1, epsilon=1e-5)
            x_fp16 = tf.cast(input_tensor_2d, tf.float16)
            out = layer(x_fp16)
            # Output dtype follows the global policy's compute dtype.
            assert not bool(keras.ops.any(keras.ops.isnan(out)))
            assert bool(keras.ops.all(keras.ops.isfinite(out)))

            with tf.GradientTape() as tape:
                tape.watch(x_fp16)
                out = layer(x_fp16, training=True)
                loss = ops.mean(ops.cast(out, "float32") ** 2)
            grads = tape.gradient(loss, layer.trainable_variables)
            for g in grads:
                assert g is not None
                assert not bool(keras.ops.any(keras.ops.isnan(g)))
        finally:
            keras.mixed_precision.set_global_policy(original_policy)

    # -------------------------------------------------------------------------
    # NUMERICS / EDGE INPUTS
    # -------------------------------------------------------------------------

    def test_zero_input(self):
        """
        All-zero input is well-defined because of the epsilon floor:
        x_sum_squared = max(0, eps) → x_length = sqrt(eps) → x_normalized = 0/sqrt(eps) = 0.
        """
        layer = BandLogitNorm(max_band_width=0.1, epsilon=1e-6)
        out = keras.ops.convert_to_numpy(layer(tf.zeros((2, 8), dtype=tf.float32)))
        assert not np.any(np.isnan(out))
        # x_normalized = 0 → output = 0 * scale = 0
        np.testing.assert_allclose(out, 0.0, atol=1e-5)

    def test_extreme_input_magnitudes_l2_stays_in_band(self):
        """
        Inputs spanning ~7 orders of magnitude must still produce L2 ∈ [1-alpha, 1].
        """
        alpha = 0.01
        rows = []
        for i in range(8):
            np.random.seed(i)
            rows.append((np.random.randn(32) * 10 ** i).astype(np.float32))
        x = np.stack(rows)
        out = keras.ops.convert_to_numpy(BandLogitNorm(max_band_width=alpha)(x))
        l2 = np.sqrt(np.sum(out ** 2, axis=-1))
        assert np.all(l2 >= 1.0 - alpha - 1e-4)
        assert np.all(l2 <= 1.0 + 1e-4)

    def test_constant_input(self):
        """Constant non-zero input: x_norm = (1/sqrt(D), ..., 1/sqrt(D)) * sign."""
        layer = BandLogitNorm(max_band_width=0.1, epsilon=1e-7)
        x = tf.constant(np.full((2, 16), 5.0, dtype=np.float32))
        out = keras.ops.convert_to_numpy(layer(x))
        l2 = np.sqrt(np.sum(out ** 2, axis=-1))
        # band_init → tanh(0) → scale = 0.95
        np.testing.assert_allclose(l2, 0.95, atol=1e-4)


# =============================================================================
# TestBandLogitNormIntegration
# =============================================================================


class TestBandLogitNormIntegration:
    """Model-integration tests for BandLogitNorm."""

    def test_logit_layer_integration(self):
        """
        Designed-for use case: `Dense → BandLogitNorm → softmax`.
        Softmax must not underflow; class probabilities must be finite and sum to 1.
        """
        num_classes = 10
        inputs = keras.Input(shape=(32,))
        x = keras.layers.Dense(num_classes)(inputs)
        x = BandLogitNorm(max_band_width=0.1)(x)
        outputs = keras.layers.Softmax()(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        x_data = tf.random.normal((16, 32))
        probs = keras.ops.convert_to_numpy(model(x_data))
        assert probs.shape == (16, num_classes)
        assert np.all(np.isfinite(probs))
        np.testing.assert_allclose(probs.sum(axis=-1), 1.0, atol=1e-5)
        # No class probability is exactly 0 (would mean log-softmax underflow).
        assert np.all(probs > 1e-9)

    def test_train_one_step_classification(self):
        inputs = keras.Input(shape=(32,))
        x = keras.layers.Dense(10)(inputs)
        x = BandLogitNorm(max_band_width=0.1)(x)
        outputs = keras.layers.Softmax()(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        x_train = tf.random.normal((64, 32))
        y_train = tf.random.uniform((64,), 0, 10, dtype=tf.int32)
        hist = model.fit(x_train, y_train, epochs=2, batch_size=16, verbose=0)
        assert len(hist.history["loss"]) == 2
        assert np.isfinite(hist.history["loss"][-1])


# =============================================================================
# TestBandLogitNormEdgeCases
# =============================================================================


class TestBandLogitNormEdgeCases:
    """Edge cases and corner inputs."""

    def test_dynamic_batch_size(self):
        inputs = keras.Input(shape=(10,))
        outputs = BandLogitNorm(max_band_width=0.1)(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        for bs in (1, 4, 17, 32):
            x = tf.random.normal((bs, 10))
            y = model(x)
            assert int(y.shape[0]) == bs

    def test_single_class_logit_layer(self):
        """D=1 is a degenerate but legal input."""
        layer = BandLogitNorm(max_band_width=0.1)
        x = tf.constant([[3.0], [-2.0], [0.5]], dtype=tf.float32)
        out = keras.ops.convert_to_numpy(layer(x))
        # x_norm = sign(x); scale = 0.95; output = ±0.95
        np.testing.assert_allclose(np.abs(out), 0.95, atol=1e-4)
        assert out[0, 0] > 0
        assert out[1, 0] < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
