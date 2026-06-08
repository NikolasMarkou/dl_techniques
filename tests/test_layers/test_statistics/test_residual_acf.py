"""Test suite for the ResidualACFLayer + ACFMonitorCallback.

Covers init/config validation, forward pass + output shape + lag-0==1, graph-mode
(`tf.function` / `keras.Model`) execution, short-sequence (seq_length < max_lag)
determinism with a numpy ground-truth ACF, gradient flow, `.keras` serialization
round-trip, training-mode `add_loss`, and a light callback smoke test.
"""

import os
import tempfile

import pytest
import numpy as np
import tensorflow as tf
import keras

from dl_techniques.layers.statistics.residual_acf import (
    ResidualACFLayer,
    ACFMonitorCallback,
)


# ---------------------------------------------------------------------
# Numpy ground-truth biased ACF, matching the layer's estimator exactly:
#   centered = r - mean(r, axis=time)
#   var      = mean(centered**2, axis=time) + epsilon
#   acf(k)   = mean(centered[:-k] * centered[k:], axis=time) / var
#   acf(0)   = 1.0
# Operates on a single (seq, features) array.
# ---------------------------------------------------------------------
def _numpy_acf(residuals_1d, max_lag, epsilon):
    r = np.asarray(residuals_1d, dtype=np.float64)
    seq = r.shape[0]
    mean = r.mean(axis=0, keepdims=True)
    centered = r - mean
    var = (centered ** 2).mean(axis=0, keepdims=True) + epsilon

    acf = np.zeros((max_lag + 1,) + r.shape[1:], dtype=np.float64)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        if lag >= seq:
            acf[lag] = 0.0
        else:
            cov = (centered[:-lag] * centered[lag:]).mean(axis=0)
            acf[lag] = cov / var[0]
    return acf


class TestResidualACFLayer:
    """Test suite for ResidualACFLayer."""

    # ----------------------------- fixtures -----------------------------
    @pytest.fixture
    def seq_length(self):
        return 64

    @pytest.fixture
    def n_features(self):
        return 3

    @pytest.fixture
    def max_lag(self):
        return 10

    @pytest.fixture
    def predictions(self, seq_length, n_features):
        return tf.random.normal([4, seq_length, n_features], seed=42)

    @pytest.fixture
    def targets(self, seq_length, n_features):
        return tf.random.normal([4, seq_length, n_features], seed=43)

    @pytest.fixture
    def layer(self, max_lag):
        return ResidualACFLayer(max_lag=max_lag)

    # --------------------------- init / config --------------------------
    def test_initialization_defaults(self):
        layer = ResidualACFLayer()
        assert layer.max_lag == 40
        assert layer.regularization_weight is None
        assert layer.target_lags == list(range(1, 41))
        assert layer.acf_threshold == 0.1
        assert layer.use_absolute_acf is True
        assert layer.epsilon == 1e-7
        assert layer.acf_values is None

    def test_initialization_custom(self):
        layer = ResidualACFLayer(
            max_lag=5,
            regularization_weight=0.5,
            target_lags=[1, 3],
            acf_threshold=0.2,
            use_absolute_acf=False,
            epsilon=1e-6,
        )
        assert layer.max_lag == 5
        assert layer.regularization_weight == 0.5
        assert layer.target_lags == [1, 3]
        assert layer.acf_threshold == 0.2
        assert layer.use_absolute_acf is False
        assert layer.epsilon == 1e-6

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_lag": 0},
            {"regularization_weight": -1.0},
            {"acf_threshold": -0.1},
            {"max_lag": 5, "target_lags": [0]},
            {"max_lag": 5, "target_lags": [6]},
        ],
    )
    def test_invalid_config_raises(self, kwargs):
        with pytest.raises(ValueError):
            ResidualACFLayer(**kwargs)

    # --------------------------- forward pass ---------------------------
    def test_forward_pass_passthrough_shape(self, layer, predictions, targets):
        out = layer([predictions, targets])
        # Pass-through: output equals predictions, same shape.
        assert out.shape == predictions.shape
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(out),
            keras.ops.convert_to_numpy(predictions),
            atol=1e-6,
        )

    def test_compute_acf_shape_and_lag0(self, layer, predictions, targets, max_lag, n_features):
        residuals = predictions - targets
        acf = layer.compute_acf(residuals)
        acf_np = keras.ops.convert_to_numpy(acf)
        assert acf_np.shape == (4, max_lag + 1, n_features)
        # Lag-0 is exactly 1.0.
        np.testing.assert_allclose(acf_np[:, 0, :], 1.0, atol=1e-6)
        assert np.all(np.isfinite(acf_np))

    def test_acf_matches_numpy_ground_truth(self, max_lag, n_features):
        # Deterministic AR(1)-like residual signal per feature.
        rng = np.random.default_rng(0)
        seq = 80
        batch = 2
        r = np.zeros((batch, seq, n_features), dtype=np.float32)
        for b in range(batch):
            for f in range(n_features):
                phi = 0.5 + 0.1 * f
                noise = rng.standard_normal(seq)
                x = np.zeros(seq)
                for t in range(1, seq):
                    x[t] = phi * x[t - 1] + noise[t]
                r[b, :, f] = x

        layer = ResidualACFLayer(max_lag=max_lag, epsilon=1e-7)
        predictions = tf.constant(r)
        targets = tf.zeros_like(predictions)  # residuals == predictions
        acf = layer.compute_acf(predictions - targets)
        acf_np = keras.ops.convert_to_numpy(acf)

        for b in range(batch):
            gt = _numpy_acf(r[b], max_lag, 1e-7)  # (max_lag+1, features)
            np.testing.assert_allclose(acf_np[b], gt, atol=1e-5)

    # ----------------------------- graph mode ---------------------------
    def test_graph_mode_tf_function(self, max_lag, seq_length, n_features):
        layer = ResidualACFLayer(max_lag=max_lag)

        @tf.function
        def run(p, t):
            return layer([p, t])

        p = tf.random.normal([2, seq_length, n_features])
        t = tf.random.normal([2, seq_length, n_features])
        out = run(p, t)
        assert out.shape == p.shape
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_graph_mode_keras_model(self, max_lag, seq_length, n_features):
        p_in = keras.Input(shape=(seq_length, n_features))
        t_in = keras.Input(shape=(seq_length, n_features))
        out = ResidualACFLayer(max_lag=max_lag)([p_in, t_in])
        model = keras.Model([p_in, t_in], out)

        p = tf.random.normal([3, seq_length, n_features])
        t = tf.random.normal([3, seq_length, n_features])
        y = model([p, t], training=False)
        assert y.shape == p.shape
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(y)))

    # -------------------------- short sequence --------------------------
    def test_short_sequence_no_crash_and_ground_truth(self, n_features):
        # seq_length (5) < max_lag (10): out-of-range lags must be deterministic 0.
        max_lag = 10
        seq = 5
        rng = np.random.default_rng(7)
        r = rng.standard_normal((1, seq, n_features)).astype(np.float32)

        layer = ResidualACFLayer(max_lag=max_lag, epsilon=1e-7)
        predictions = tf.constant(r)
        targets = tf.zeros_like(predictions)
        acf = layer.compute_acf(predictions - targets)
        acf_np = keras.ops.convert_to_numpy(acf)

        assert acf_np.shape == (1, max_lag + 1, n_features)
        assert np.all(np.isfinite(acf_np))

        gt = _numpy_acf(r[0], max_lag, 1e-7)
        np.testing.assert_allclose(acf_np[0], gt, atol=1e-5)
        # Lags >= seq_length are exactly 0.
        np.testing.assert_allclose(acf_np[0, seq:, :], 0.0, atol=1e-7)

    def test_short_sequence_graph_mode(self, n_features):
        max_lag = 10
        seq = 5
        layer = ResidualACFLayer(max_lag=max_lag)

        @tf.function
        def run(p, t):
            return layer.compute_acf(p - t)

        p = tf.random.normal([2, seq, n_features])
        t = tf.random.normal([2, seq, n_features])
        acf = run(p, t)
        acf_np = keras.ops.convert_to_numpy(acf)
        assert acf_np.shape == (2, max_lag + 1, n_features)
        assert np.all(np.isfinite(acf_np))
        np.testing.assert_allclose(acf_np[:, seq:, :], 0.0, atol=1e-7)

    # ---------------------------- gradient flow -------------------------
    def test_gradient_flow(self, max_lag, seq_length, n_features):
        layer = ResidualACFLayer(max_lag=max_lag, regularization_weight=1.0)
        p = tf.Variable(tf.random.normal([2, seq_length, n_features]))
        t = tf.constant(tf.random.normal([2, seq_length, n_features]))

        with tf.GradientTape() as tape:
            out = layer([p, t], training=True)
            loss = tf.reduce_mean(out) + tf.add_n(layer.losses)

        grad = tape.gradient(loss, p)
        assert grad is not None
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(grad)))

    # ----------------------------- training -----------------------------
    def test_training_true_adds_loss(self, max_lag, seq_length, n_features):
        layer = ResidualACFLayer(max_lag=max_lag, regularization_weight=0.5)
        p = tf.random.normal([2, seq_length, n_features])
        t = tf.random.normal([2, seq_length, n_features])
        layer([p, t], training=True)
        assert len(layer.losses) > 0
        assert float(keras.ops.convert_to_numpy(layer.losses[0])) >= 0.0

    def test_training_none_no_loss(self, max_lag, seq_length, n_features):
        layer = ResidualACFLayer(max_lag=max_lag, regularization_weight=0.5)
        p = tf.random.normal([2, seq_length, n_features])
        t = tf.random.normal([2, seq_length, n_features])
        layer([p, t])  # training defaults to None -> no loss
        assert len(layer.losses) == 0

    def test_no_reg_weight_no_loss_even_in_training(self, max_lag, seq_length, n_features):
        layer = ResidualACFLayer(max_lag=max_lag, regularization_weight=None)
        p = tf.random.normal([2, seq_length, n_features])
        t = tf.random.normal([2, seq_length, n_features])
        layer([p, t], training=True)
        assert len(layer.losses) == 0

    # --------------------------- serialization --------------------------
    def test_get_config_round_trip(self):
        layer = ResidualACFLayer(
            max_lag=7,
            regularization_weight=0.3,
            target_lags=[1, 2, 5],
            acf_threshold=0.15,
            use_absolute_acf=False,
            epsilon=1e-6,
        )
        cfg = layer.get_config()
        rebuilt = ResidualACFLayer.from_config(cfg)
        assert rebuilt.max_lag == 7
        assert rebuilt.regularization_weight == 0.3
        assert rebuilt.target_lags == [1, 2, 5]
        assert rebuilt.acf_threshold == 0.15
        assert rebuilt.use_absolute_acf is False
        assert rebuilt.epsilon == 1e-6

    def test_keras_model_serialization_round_trip(self, max_lag, seq_length, n_features):
        p_in = keras.Input(shape=(seq_length, n_features))
        t_in = keras.Input(shape=(seq_length, n_features))
        out = ResidualACFLayer(max_lag=max_lag, name="acf")([p_in, t_in])
        model = keras.Model([p_in, t_in], out)

        p = tf.random.normal([3, seq_length, n_features])
        t = tf.random.normal([3, seq_length, n_features])
        y_before = keras.ops.convert_to_numpy(model([p, t], training=False))

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "acf.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            y_after = keras.ops.convert_to_numpy(loaded([p, t], training=False))

        np.testing.assert_allclose(y_before, y_after, atol=1e-6)

    # ------------------------- acf summary state ------------------------
    def test_get_acf_summary_after_call(self, max_lag, seq_length, n_features):
        layer = ResidualACFLayer(max_lag=max_lag)
        # Before any call -> None.
        assert layer.get_acf_summary() is None
        p = tf.random.normal([2, seq_length, n_features])
        t = tf.random.normal([2, seq_length, n_features])
        layer([p, t])
        summary = layer.get_acf_summary()
        assert summary is not None
        assert "mean_abs_acf" in summary
        assert "max_abs_acf" in summary
        assert "significant_lags" in summary
        assert np.isfinite(summary["mean_abs_acf"])


class TestACFMonitorCallback:
    """Light smoke test for ACFMonitorCallback."""

    def test_init(self):
        cb = ACFMonitorCallback(layer_name="acf", log_frequency=5)
        assert cb.layer_name == "acf"
        assert cb.log_frequency == 5
        assert cb.batch_count == 0

    def test_on_train_batch_end_smoke(self):
        # Build a tiny model containing a named ResidualACFLayer and drive the
        # callback's on_train_batch_end past its log_frequency.
        seq_length, n_features = 16, 2
        p_in = keras.Input(shape=(seq_length, n_features))
        t_in = keras.Input(shape=(seq_length, n_features))
        out = ResidualACFLayer(max_lag=4, name="acf_mon")([p_in, t_in])
        model = keras.Model([p_in, t_in], out)

        # Populate acf_values via a forward pass so get_acf_summary returns data.
        p = tf.random.normal([2, seq_length, n_features])
        t = tf.random.normal([2, seq_length, n_features])
        model([p, t])

        cb = ACFMonitorCallback(layer_name="acf_mon", log_frequency=1)
        cb.set_model(model)
        # Should not raise; batch_count increments and triggers a log at freq 1.
        cb.on_train_batch_end(batch=0, logs={"loss": 0.1})
        assert cb.batch_count == 1
